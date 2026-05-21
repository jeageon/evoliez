"""Top-K mutation positions colored on the WT cartoon (PyMOL).

Parses ``reports/final_candidates.csv``, takes the top ``top_k`` rows by
order in the file (which the pipeline writes ranked best-first), pulls
their mutated residue numbers, and renders a PyMOL PNG with each mutated
side-chain shown as sticks tinted by ``evidence_color``.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from evoliez.figures.style import evidence_color, style_dpi
from evoliez.figures.three_d.pymol_runner import pymol_available, run_pymol_script
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

_LIGAND_RESN = "NDP+NAP+LIG+NAI+NAD+SAH+SAM"

# Mutation token: ``D222N`` (one-letter wt, integer position, one-letter
# mut).  Position may be followed by an insertion code (``222A``); the
# trailing letter is the mutated residue.  We only need the integer
# position so the regex captures that.
_MUT_RE = re.compile(r"^[A-Z](?P<pos>\d+)[A-Z*]$")


def _parse_positions(mut_str: str) -> List[int]:
    """Parse ``"D222N,Y196F"`` -> ``[222, 196]``."""
    if not mut_str:
        return []
    out: List[int] = []
    for token in re.split(r"[,;+\s]+", mut_str.strip()):
        if not token:
            continue
        m = _MUT_RE.match(token.strip())
        if not m:
            continue
        try:
            out.append(int(m.group("pos")))
        except ValueError:
            continue
    return out


def _read_top_rows(csv_path: Path, top_k: int) -> List[Dict[str, str]]:
    try:
        with csv_path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except OSError as exc:
        _LOGGER.warning("mutation_overlay: failed to read %s: %s", csv_path, exc)
        return []
    return rows[: max(1, int(top_k))]


def _mutation_field(row: Dict[str, str]) -> str:
    # Stage writers use ``mutations`` (plural) but tolerate older
    # ``mutation`` to keep the figure builder forward-compatible.
    for key in ("mutations", "mutation"):
        v = row.get(key)
        if v:
            return str(v)
    return ""


def _hex_to_pymol(hex_color: str) -> str:
    """Turn ``#009E73`` into ``0x009E73`` (PyMOL hex literal)."""
    s = hex_color.lstrip("#")
    if len(s) == 6:
        return f"0x{s.upper()}"
    return "0x999999"


_HEADER = Template(
    """\
load $pdb_path, wt
hide everything
show cartoon, wt and polymer
color grey80, wt and polymer
show sticks, wt and resn $ligand_resn
color yellow, wt and resn $ligand_resn
bg_color white
set ray_shadows, 1
set ray_opaque_background, off
"""
)

_HIGHLIGHT = Template(
    """\
select mut_$position, wt and polymer and resi $position
show sticks, mut_$position
color $color, mut_$position
"""
)

_FOOTER = Template(
    """\
zoom wt and (resn $ligand_resn or ($highlight_sel)), 6
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


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    top_k: int = 10,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render the top-K mutation positions onto the WT cartoon."""
    wt_pdb = getattr(artifacts, "wt_complex_pdb", None)
    if wt_pdb is None or not Path(wt_pdb).exists():
        _LOGGER.info("mutation_overlay: wt_complex_pdb missing, skipping")
        return None
    csv_path = getattr(artifacts, "final_candidates_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        _LOGGER.info("mutation_overlay: final_candidates_csv missing, skipping")
        return None
    if not pymol_available():
        _LOGGER.info("mutation_overlay: PyMOL not on PATH, deferring to 3Dmol")
        return None

    rows = _read_top_rows(Path(csv_path), int(top_k))
    if not rows:
        _LOGGER.info("mutation_overlay: empty final_candidates.csv, skipping")
        return None

    # Collect (position, evidence_class) - if two candidates touch the
    # same residue the higher-ranked one wins (we walk in CSV order, which
    # the pipeline writes best-first).
    position_color: Dict[int, str] = {}
    for row in rows:
        ec = row.get("evidence_class") or row.get("evidence") or ""
        col = _hex_to_pymol(evidence_color(ec))
        for pos in _parse_positions(_mutation_field(row)):
            position_color.setdefault(pos, col)

    if not position_color:
        _LOGGER.info("mutation_overlay: no parsable mutation positions, skipping")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = style_dpi(style)
    width, height = _resolution_for(dpi)
    parts: List[str] = [
        _HEADER.substitute(
            pdb_path=str(Path(wt_pdb).resolve()),
            ligand_resn=_LIGAND_RESN,
        )
    ]
    for pos, col in sorted(position_color.items()):
        parts.append(_HIGHLIGHT.substitute(position=pos, color=col))

    # ``resi 12+45+78`` is the PyMOL syntax for an arbitrary residue list.
    highlight_sel = "resi " + "+".join(str(p) for p in sorted(position_color))
    parts.append(
        _FOOTER.substitute(
            ligand_resn=_LIGAND_RESN,
            highlight_sel=highlight_sel,
            width=width,
            height=height,
            out_path=str(out_path.resolve()),
            dpi=dpi,
        )
    )

    ok, msg = run_pymol_script("".join(parts))
    if not ok or not out_path.exists():
        _LOGGER.warning("mutation_overlay: PyMOL render failed: %s", msg)
        return None

    return FigureSpec(
        figure_id="09_final_library_structure",
        section="final_library",
        title="Top-K mutation positions on WT",
        description=(
            f"Top-{top_k} candidate mutation sites mapped onto the WT "
            "structure, colored by evidence class."
        ),
        path=out_path,
        source_files=[Path(wt_pdb), Path(csv_path)],
        renderer="pymol",
        params={
            "style": style,
            "top_k": int(top_k),
            "n_positions": len(position_color),
            "dpi": dpi,
            "resolution": [width, height],
        },
    )
