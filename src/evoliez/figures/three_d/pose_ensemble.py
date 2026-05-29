"""Overlay of all WT Boltz pose samples on a shared protein backbone.

Loads each ``*_model_*.pdb`` from the WT Boltz predictions directory,
aligns them on the protein chain, then hides everything but the ligand
for the secondary models so the user sees one cartoon + N stick poses.
Pose color follows a viridis-style gradient ranked by model index (model
0 = highest confidence, brightest yellow; later models -> darker).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template
from typing import Any, List, Optional, Tuple

from evoliez.figures.style import style_dpi
from evoliez.figures.three_d.pymol_runner import pymol_available, run_pymol_script
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

_LIGAND_RESN = "NDP+NAP+LIG+NAI+NAD+SAH+SAM"

# 8-stop viridis approximation (yellow -> teal -> dark purple).  Used to
# tint pose models in confidence order; PyMOL accepts hex via
# ``color 0xRRGGBB, sel``.
_VIRIDIS_HEX: Tuple[str, ...] = (
    "0xfde725",  # yellow (best)
    "0xa0da39",
    "0x4ac16d",
    "0x1fa187",
    "0x277f8e",
    "0x365c8d",
    "0x46327e",
    "0x440154",  # dark purple (worst)
)

_MODEL_RE = re.compile(r"_model_(?P<idx>\d+)\.pdb$", re.IGNORECASE)


def _find_pose_pdbs(artifacts: ReportArtifacts) -> List[Path]:
    """Return all WT Boltz pose PDBs, sorted by model index (0 first).

    Walks the whole ``complexes/`` tree for the
    ``boltz_results_wt_boltz_input`` result dir rather than hardcoding a
    path: s04 writes WT predictions under ``complexes/boltz/...``
    (``ctx.paths.complexes / "boltz"``), while an older layout put them
    directly under ``complexes/...``. rglob finds either.
    """
    run_dir = getattr(artifacts, "run_dir", None)
    if run_dir is None:
        return []
    complexes = Path(run_dir) / "complexes"
    if not complexes.exists():
        return []

    candidates: List[Path] = []
    # Each match is the WT Boltz result dir under whichever layout; pose
    # PDBs live under its ``predictions/`` subtree.
    for result_dir in sorted(complexes.rglob("boltz_results_wt_boltz_input")):
        if not result_dir.is_dir():
            continue
        preds = result_dir / "predictions"
        search_root = preds if preds.exists() else result_dir
        candidates.extend(search_root.rglob("*_model_*.pdb"))

    # De-dup (rglob can surface the same file via nested matches) while
    # keeping a stable set.
    unique = sorted(set(candidates))

    def _key(p: Path) -> Tuple[int, str]:
        m = _MODEL_RE.search(p.name)
        idx = int(m.group("idx")) if m else 1_000_000
        return (idx, p.name)

    return sorted(unique, key=_key)


def _distinct_model_count(poses: List[Path]) -> int:
    """Number of distinct Boltz model indices among ``poses``.

    Two copies of ``*_model_0.pdb`` (e.g. a mock backend that wrote one
    pose into one predictions dir) count as a single model - not an
    ensemble. Files without a parseable ``_model_<n>`` suffix each count
    as their own (path-keyed) distinct entry.
    """
    seen: set = set()
    for p in poses:
        m = _MODEL_RE.search(p.name)
        seen.add(("idx", int(m.group("idx"))) if m else ("path", str(p)))
    return len(seen)


def _confidence_for(pdb_path: Path) -> Optional[float]:
    """Read the matching ``confidence_*_model_*.json`` next to ``pdb_path``.

    Returns ``None`` if missing or malformed; the caller falls back to the
    model-index gradient.
    """
    m = _MODEL_RE.search(pdb_path.name)
    if not m:
        return None
    idx = m.group("idx")
    parent = pdb_path.parent
    for cand in parent.glob(f"confidence*_model_{idx}.json"):
        try:
            data = json.loads(cand.read_text())
        except (OSError, ValueError):
            continue
        score = data.get("confidence_score") or data.get("complex_plddt")
        if isinstance(score, (int, float)):
            return float(score)
    return None


# String.Template avoids brace conflicts with PyMOL selection syntax.
_HEADER = Template(
    """\
bg_color white
set ray_shadows, 1
set ray_opaque_background, off
"""
)

_LOAD_REF = Template(
    """\
load $pdb_path, $obj_name
hide everything, $obj_name
show cartoon, $obj_name and polymer
color cyan, $obj_name and polymer
show sticks, $obj_name and resn $ligand_resn
color $color, $obj_name and resn $ligand_resn
"""
)

_LOAD_POSE = Template(
    """\
load $pdb_path, $obj_name
hide everything, $obj_name
intra_fit $obj_name and polymer, $ref_name and polymer
show sticks, $obj_name and resn $ligand_resn
color $color, $obj_name and resn $ligand_resn
"""
)

_FOOTER = Template(
    """\
zoom $ref_name and resn $ligand_resn, 10
ray $width, $height
png $out_path, dpi=$dpi
"""
)


def _resolution_for(dpi: int) -> Tuple[int, int]:
    if dpi >= 500:
        return (2400, 1800)
    if dpi >= 250:
        return (2000, 1500)
    return (1600, 1200)


def _color_for(rank: int, n_poses: int) -> str:
    """Pick a viridis stop for the ``rank``-th pose (0 = best)."""
    if n_poses <= 1:
        return _VIRIDIS_HEX[0]
    # Distribute ranks evenly across the palette.
    slot = int(round(rank * (len(_VIRIDIS_HEX) - 1) / max(1, n_poses - 1)))
    slot = max(0, min(len(_VIRIDIS_HEX) - 1, slot))
    return _VIRIDIS_HEX[slot]


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    max_samples: int = 10,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render the WT pose-ensemble overlay PNG.

    Returns ``None`` when PyMOL is unavailable, when no WT pose PDBs were
    found, or when the WT complex PDB itself is missing (the caller then
    falls back to the 3Dmol inline viewer).
    """
    if getattr(artifacts, "wt_complex_pdb", None) is None:
        _LOGGER.info("pose_ensemble: wt_complex_pdb missing, skipping")
        return None
    if not pymol_available():
        _LOGGER.info("pose_ensemble: PyMOL not on PATH, deferring to 3Dmol")
        return None

    poses = _find_pose_pdbs(artifacts)
    if not poses:
        _LOGGER.info("pose_ensemble: no Boltz pose PDBs found, skipping")
        return None

    # Honesty guard: an "ensemble" needs >= 2 distinct model files. A mock
    # backend (or a single-diffusion-sample run) yields one model_0 - which
    # would render as a single pose mislabeled "ensemble". Bail so the
    # builder's 3Dmol single-pose fallback takes over instead.
    if _distinct_model_count(poses) < 2:
        _LOGGER.info(
            "pose_ensemble: only %d distinct model file(s) found "
            "(need >=2 for an ensemble); deferring to 3Dmol single-pose view",
            _distinct_model_count(poses),
        )
        return None

    poses = poses[: max(1, int(max_samples))]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = style_dpi(style)
    width, height = _resolution_for(dpi)

    n = len(poses)
    confidences: List[Optional[float]] = [_confidence_for(p) for p in poses]
    script_parts: List[str] = [_HEADER.substitute()]

    ref_name = "wt_ref"
    script_parts.append(
        _LOAD_REF.substitute(
            pdb_path=str(poses[0].resolve()),
            obj_name=ref_name,
            ligand_resn=_LIGAND_RESN,
            color=_color_for(0, n),
        )
    )
    for rank, pose in enumerate(poses[1:], start=1):
        script_parts.append(
            _LOAD_POSE.substitute(
                pdb_path=str(pose.resolve()),
                obj_name=f"pose_{rank}",
                ref_name=ref_name,
                ligand_resn=_LIGAND_RESN,
                color=_color_for(rank, n),
            )
        )
    script_parts.append(
        _FOOTER.substitute(
            ref_name=ref_name,
            ligand_resn=_LIGAND_RESN,
            width=width,
            height=height,
            out_path=str(out_path.resolve()),
            dpi=dpi,
        )
    )

    ok, msg = run_pymol_script("".join(script_parts))
    if not ok or not out_path.exists():
        _LOGGER.warning("pose_ensemble: PyMOL render failed: %s", msg)
        return None

    return FigureSpec(
        figure_id="04_boltz_pose_ensemble",
        section="boltz",
        title="Boltz pose ensemble (WT)",
        description=(
            f"Overlay of {n} WT Boltz pose samples on a shared cartoon, "
            "colored by confidence rank (viridis)."
        ),
        path=out_path,
        source_files=[Path(p) for p in poses],
        renderer="pymol",
        params={
            "style": style,
            "n_poses": n,
            "max_samples": int(max_samples),
            "dpi": dpi,
            "resolution": [width, height],
            "confidences": [c if c is not None else None for c in confidences],
        },
    )
