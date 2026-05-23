"""Representative MD frame viewers (start / mid / end) for section 8.

Builds up to three 3Dmol.js viewer contexts so the report can show the
candidate pose at the beginning, middle, and end of the MD trajectory.
This complements the static RMSD time series and the optional movie:
the viewers are interactive (rotate / zoom) and always file://-safe.

Frame extraction strategy (no MDAnalysis, no MDTraj - GPL closure):

1. **Start frame**: always emitted when a minimized PDB exists -- it's
   the topology file the OpenMM engine writes alongside the trajectory.
2. **Mid / End frames**: if OpenMM is importable AND a DCD trajectory
   path exists in ``analysis.json``, we use ``openmm.app.DCDFile`` to
   pull two frames (midpoint and last) and emit them as PDB strings via
   ``openmm.app.PDBFile.writeFile`` to a temp path.  If OpenMM is
   unavailable or anything fails, we fall back to the start frame only
   -- the movie path (``movies/md_trajectory.py``) is the primary route
   for paper-grade animations anyway.

Returns a list (length 1, 2, or 3) of viewer-context dicts ready to be
appended to ``sections["08_md"]["viewers"]``.  Returns ``[]`` when no
candidate has even a minimized PDB on disk.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evoliez.figures.three_d.pdb_inline import make_viewer_context
from evoliez.figures.types import ReportArtifacts

_LOGGER = logging.getLogger(__name__)


def _load_analysis(md_dir: Path) -> Dict[str, Any]:
    candidate = md_dir / "analysis.json"
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text())
    except (OSError, ValueError):
        return {}


def _highest_scoring_candidate(
    artifacts: ReportArtifacts, md_dirs: Dict[str, Path]
) -> Optional[str]:
    """Pick the top candidate by ``final_score`` from final_candidates.csv.

    Falls back to the first md_dirs key when the CSV is absent or doesn't
    line up with any md candidate id.
    """
    csv_path = getattr(artifacts, "final_candidates_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        return next(iter(md_dirs), None)
    best: Optional[Tuple[float, str]] = None
    try:
        with Path(csv_path).open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                cid = row.get("candidate_id")
                if not cid or cid not in md_dirs:
                    continue
                try:
                    score = float(row.get("final_score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                if best is None or score > best[0]:
                    best = (score, cid)
    except OSError:
        return next(iter(md_dirs), None)
    if best is None:
        return next(iter(md_dirs), None)
    return best[1]


def _resolve_path(value: Any, *, base: Path) -> Optional[Path]:
    """Resolve ``value`` to an existing file path or ``None``.

    Tries the raw value first, then ``base / value.name`` and ``base /
    value`` so analysis.json entries that were written with an absolute
    path on a different host still resolve when the run directory was
    copied locally.
    """
    if not value:
        return None
    raw = Path(str(value))
    if raw.exists():
        return raw
    try:
        base = Path(base)
    except TypeError:
        return None
    candidate = base / raw.name
    if candidate.exists():
        return candidate
    candidate = base / raw
    if candidate.exists():
        return candidate
    return None


def _find_minimized_pdb(
    md_dir: Path, candidate_id: str, analysis: Dict[str, Any]
) -> Optional[Path]:
    """Locate the minimized starting PDB.

    Order of attempts:
    1. ``analysis.topology_path`` (the engine writes this for real MD).
    2. ``<md_dir>/<candidate_id>_minimized.pdb`` (the engine's convention).
    3. The first ``*_minimized.pdb`` under ``md_dir``.
    """
    top = _resolve_path(analysis.get("topology_path"), base=md_dir)
    if top is not None:
        return top
    direct = md_dir / f"{candidate_id}_minimized.pdb"
    if direct.exists():
        return direct
    hits = sorted(md_dir.glob("*_minimized.pdb"))
    return hits[0] if hits else None


def _try_extract_mid_end_pdbs(
    topology_pdb: Path,
    trajectory_path: Path,
    *,
    tmp_dir: Path,
) -> List[Tuple[str, Path]]:
    """Extract midpoint + last frame as PDB files via OpenMM.

    Returns a list of ``(label, path)`` tuples (length 0, 1, or 2).
    Never raises -- any failure short-circuits to an empty list and the
    caller falls back to start-only.
    """
    out: List[Tuple[str, Path]] = []
    try:
        # OpenMM is the only stack we're allowed to import for trajectory
        # IO (license note in the module docstring).  Defer the import so
        # this module is harmless to import in a pure-Python env.
        from openmm import app  # type: ignore[import-not-found]  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        _LOGGER.info(
            "md_frames: OpenMM not importable (%s); start-frame only", exc
        )
        return out

    try:
        pdb = app.PDBFile(str(topology_pdb))
        topology = pdb.topology
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "md_frames: could not parse topology PDB %s: %s", topology_pdb, exc
        )
        return out

    try:
        with open(trajectory_path, "rb") as fh:
            dcd = app.DCDFile(fh, topology)
            n_frames = getattr(dcd, "_numFrames", None)
            if n_frames is None:
                # Older OpenMM exposes ``numFrames`` differently; defer
                # to readAllFrames as the safe fallback.
                positions_list = []
                while True:
                    try:
                        positions_list.append(dcd.getPositions())
                    except Exception:  # noqa: BLE001
                        break
            else:
                positions_list = []
                for _ in range(int(n_frames)):
                    try:
                        positions_list.append(dcd.getPositions())
                    except Exception:  # noqa: BLE001
                        break
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "md_frames: could not read DCD %s: %s", trajectory_path, exc
        )
        return out

    if not positions_list:
        return out

    mid_idx = len(positions_list) // 2
    end_idx = len(positions_list) - 1
    wanted: List[Tuple[str, int]] = []
    if mid_idx > 0 and mid_idx != end_idx:
        wanted.append(("mid", mid_idx))
    if end_idx > 0:
        wanted.append(("end", end_idx))

    tmp_dir.mkdir(parents=True, exist_ok=True)
    for label, idx in wanted:
        try:
            positions = positions_list[idx]
            dest = tmp_dir / f"md_frame_{label}.pdb"
            with open(dest, "w") as fh:
                app.PDBFile.writeFile(topology, positions, fh)
            out.append((label, dest))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "md_frames: could not write %s frame: %s", label, exc
            )
            continue
    return out


def render(
    artifacts: ReportArtifacts,
    *,
    candidate_id: Optional[str] = None,
    viewer_id_prefix: str = "md_frame",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Build viewer contexts for the start (+ optional mid / end) MD frames.

    Parameters
    ----------
    artifacts:
        Discovered run artifacts.  Reads ``md_dirs`` and
        ``final_candidates_csv``.
    candidate_id:
        Pick a specific candidate; when ``None``, falls back to the
        highest-scoring candidate per the final CSV.
    viewer_id_prefix:
        Prefix for the viewer element ids (sanitized by
        ``make_viewer_context``).

    Returns up to 3 viewer-context dicts (start, mid, end).  Empty list
    when no candidate has a minimized PDB on disk.
    """
    md_dirs = getattr(artifacts, "md_dirs", None) or {}
    if not md_dirs:
        _LOGGER.info("md_frames: no MD directories, skipping")
        return []

    cid = candidate_id or _highest_scoring_candidate(artifacts, md_dirs)
    if cid is None or cid not in md_dirs:
        _LOGGER.info("md_frames: could not resolve candidate id (%r)", cid)
        return []

    md_dir = Path(md_dirs[cid])
    analysis = _load_analysis(md_dir)

    start_pdb = _find_minimized_pdb(md_dir, cid, analysis)
    if start_pdb is None:
        _LOGGER.info(
            "md_frames: no minimized PDB for candidate %s under %s", cid, md_dir
        )
        return []

    contexts: List[Dict[str, Any]] = []
    start_ctx = make_viewer_context(
        start_pdb,
        viewer_id=f"{viewer_id_prefix}_{cid}_start",
        height=360,
        title=f"MD start frame ({cid})",
    )
    if start_ctx and not start_ctx.get("missing"):
        contexts.append(start_ctx)

    # Optional mid / end frames (require OpenMM + trajectory).
    traj_path = _resolve_path(analysis.get("trajectory_path"), base=md_dir)
    if traj_path is not None and traj_path.exists():
        tmp_dir = Path(tempfile.mkdtemp(prefix="md_frames_"))
        extras = _try_extract_mid_end_pdbs(
            start_pdb, traj_path, tmp_dir=tmp_dir
        )
        for label, pdb in extras:
            ctx = make_viewer_context(
                pdb,
                viewer_id=f"{viewer_id_prefix}_{cid}_{label}",
                height=360,
                title=f"MD {label} frame ({cid})",
            )
            if ctx and not ctx.get("missing"):
                contexts.append(ctx)

    return contexts
