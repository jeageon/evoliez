"""MD trajectory movie via headless PyMOL + ffmpeg.

Workflow (best-effort -- any missing tool returns ``None``):

1. Locate the highest-scoring MD candidate (or honor ``candidate_id``)
   and read its ``analysis.json`` for ``trajectory_path`` +
   ``topology_path``.
2. Write a PyMOL script that ``load``s the topology, ``load_traj``s the
   DCD, runs ``intra_fit`` so the camera stays put, then ``mpng``s up to
   ``max_frames`` PNGs into a scratch directory.
3. Invoke ``pymol -cq`` via :func:`subprocess.run`.
4. Stitch the PNGs with ``ffmpeg -framerate 24 -i frame_%04d.png ...``
   into a Chrome / Firefox -compatible H.264 mp4.
5. Build a :class:`FigureSpec` whose ``path`` is the mp4 and whose
   ``params["poster"]`` is the first-frame PNG (the figure_card template
   uses ``poster`` to give browsers something to display before play).

License note: we never ``import pymol`` (its bindings can be GPL on
some builds).  The subprocess invocation is the same isolation pattern
:mod:`evoliez.figures.three_d.pymol_runner` uses.  ffmpeg is similarly
exec-only (no Python bindings).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

# Cap the frame count so PyMOL ray-tracing stays under ~minutes even on
# large trajectories.  The DCDs the OpenMM engine writes default to 50
# frames per production run; 100 is a safe upper bound that still gives
# a usable animation on longer custom runs.
_DEFAULT_MAX_FRAMES = 100

# Ligand resn set kept in sync with the other 3D renderers; PyMOL's
# ``resn`` selection accepts ``+``-joined names.
_LIGAND_RESN = "NDP+NAP+LIG+NAI+NAD+SAH+SAM"


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
    """Pick the top candidate by ``final_score`` from the candidates CSV."""
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
    """Resolve a path string from analysis.json against ``base`` if needed."""
    if not value:
        return None
    raw = Path(str(value))
    if raw.exists():
        return raw
    base = Path(base)
    cand = base / raw.name
    if cand.exists():
        return cand
    cand = base / raw
    if cand.exists():
        return cand
    return None


def _which(name: str) -> Optional[str]:
    found = shutil.which(name)
    return found if found else None


_PYMOL_SCRIPT_TEMPLATE = """\
bg_color white
set ray_shadows, 0
set ray_opaque_background, off
load {topology}, mol
load_traj {trajectory}, mol, 1, , 1, , , {max_frames}
intra_fit mol and polymer
hide everything
show cartoon, mol and polymer
color cyan, mol and polymer
show sticks, mol and resn {ligand_resn}
color magenta, mol and resn {ligand_resn}
zoom mol and resn {ligand_resn}, 8
set ray_trace_mode, 0
mpng {frames_prefix}, 1, 0, 0, 0, {width}, {height}
"""


def _write_pymol_script(
    *,
    topology: Path,
    trajectory: Path,
    frames_prefix: Path,
    max_frames: int,
    width: int,
    height: int,
) -> str:
    return _PYMOL_SCRIPT_TEMPLATE.format(
        topology=str(topology.resolve()),
        trajectory=str(trajectory.resolve()),
        # PyMOL writes <prefix>0001.png, <prefix>0002.png, ...
        frames_prefix=str(frames_prefix.resolve()),
        max_frames=int(max_frames),
        ligand_resn=_LIGAND_RESN,
        width=int(width),
        height=int(height),
    )


def _resolution_for(style: str) -> Tuple[int, int]:
    if style == "paper":
        return (1600, 1200)
    if style == "poster":
        return (1920, 1440)
    return (1280, 960)


def _run_pymol(
    pymol_bin: str, script: str, *, timeout: float = 300
) -> Tuple[bool, str]:
    cmd = [pymol_bin, "-cq", "/dev/stdin"]
    try:
        proc = subprocess.run(
            cmd,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (False, f"pymol timed out after {timeout}s")
    except (OSError, ValueError) as exc:
        return (False, f"pymol launch failed: {exc}")
    if proc.returncode != 0:
        return (False, (proc.stderr or proc.stdout or "").strip() or "pymol failed")
    return (True, proc.stdout or "")


def _run_ffmpeg(
    ffmpeg_bin: str,
    frames_glob: Path,
    out_path: Path,
    *,
    framerate: int = 24,
    timeout: float = 120,
) -> Tuple[bool, str]:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(int(framerate)),
        "-i",
        str(frames_glob),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        # Chrome/Firefox-friendly baseline profile (works on file://).
        "-profile:v",
        "high",
        "-level",
        "4.0",
        str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (False, f"ffmpeg timed out after {timeout}s")
    except (OSError, ValueError) as exc:
        return (False, f"ffmpeg launch failed: {exc}")
    if proc.returncode != 0:
        return (False, (proc.stderr or proc.stdout or "").strip() or "ffmpeg failed")
    return (True, proc.stdout or "")


def _collect_frames(scratch_dir: Path, prefix: str) -> List[Path]:
    """List the PNGs PyMOL emitted, sorted by frame index."""
    return sorted(scratch_dir.glob(f"{prefix}*.png"))


def _renumber_frames(frames: List[Path], scratch_dir: Path) -> Optional[Path]:
    """Copy PyMOL's frames into a strict ``frame_%04d.png`` sequence.

    PyMOL emits names like ``<prefix>0001.png`` -- ffmpeg's ``-i`` does
    accept that, but renaming into a known pattern keeps the ffmpeg
    command independent of PyMOL's frame-number padding.  Returns the
    pattern path to feed ffmpeg, or ``None`` if no frames are present.
    """
    if not frames:
        return None
    renumbered_dir = scratch_dir / "renumbered"
    renumbered_dir.mkdir(exist_ok=True)
    for idx, src in enumerate(frames, start=1):
        dst = renumbered_dir / f"frame_{idx:04d}.png"
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            _LOGGER.warning("md_trajectory: could not copy frame %s: %s", src, exc)
            return None
    return renumbered_dir / "frame_%04d.png"


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    candidate_id: Optional[str] = None,
    style: str = "presentation",
    max_frames: int = _DEFAULT_MAX_FRAMES,
    framerate: int = 24,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render an MD trajectory mp4 for ``candidate_id`` (or top candidate).

    Returns ``None`` for any of the following (no exception ever leaks):

    * No MD directories on the artifacts.
    * Resolved candidate has no usable ``trajectory_path`` /
      ``topology_path`` in ``analysis.json``.
    * PyMOL or ffmpeg not on PATH.
    * PyMOL / ffmpeg subprocess errors out.
    """
    md_dirs = getattr(artifacts, "md_dirs", None) or {}
    if not md_dirs:
        _LOGGER.info("md_trajectory: no MD directories, skipping")
        return None

    pymol_bin = _which("pymol") or _which("pymolcli") or _which("PyMOL")
    if not pymol_bin:
        _LOGGER.info("md_trajectory: pymol not on PATH, skipping movie")
        return None
    ffmpeg_bin = _which("ffmpeg")
    if not ffmpeg_bin:
        _LOGGER.info("md_trajectory: ffmpeg not on PATH, skipping movie")
        return None

    cid = candidate_id or _highest_scoring_candidate(artifacts, md_dirs)
    if cid is None or cid not in md_dirs:
        _LOGGER.info("md_trajectory: cannot resolve candidate id (%r)", cid)
        return None

    md_dir = Path(md_dirs[cid])
    analysis = _load_analysis(md_dir)
    traj = _resolve_path(analysis.get("trajectory_path"), base=md_dir)
    top = _resolve_path(analysis.get("topology_path"), base=md_dir)
    if top is None:
        # Fall back to the engine's minimised PDB convention.
        guess = md_dir / f"{cid}_minimized.pdb"
        if guess.exists():
            top = guess
    if traj is None or top is None:
        _LOGGER.info(
            "md_trajectory: trajectory/topology missing for %s (traj=%s top=%s)",
            cid,
            traj,
            top,
        )
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = _resolution_for(style)
    scratch_dir = Path(tempfile.mkdtemp(prefix="md_movie_"))
    try:
        frames_prefix = scratch_dir / "frame_"
        script = _write_pymol_script(
            topology=top,
            trajectory=traj,
            frames_prefix=frames_prefix,
            max_frames=max(1, int(max_frames)),
            width=width,
            height=height,
        )
        ok, msg = _run_pymol(pymol_bin, script)
        if not ok:
            _LOGGER.warning("md_trajectory: pymol failed: %s", msg)
            return None

        frames = _collect_frames(scratch_dir, "frame_")
        if not frames:
            _LOGGER.warning("md_trajectory: pymol produced no frames")
            return None

        pattern = _renumber_frames(frames, scratch_dir)
        if pattern is None:
            return None

        ok, msg = _run_ffmpeg(
            ffmpeg_bin, pattern, out_path, framerate=int(framerate)
        )
        if not ok or not out_path.exists():
            _LOGGER.warning("md_trajectory: ffmpeg failed: %s", msg)
            return None

        # Save the first frame as a poster image next to the mp4 so
        # browsers that can't autoplay video still have something to show.
        poster_path: Optional[Path] = None
        try:
            poster_path = out_path.with_suffix(".poster.png")
            shutil.copy2(frames[0], poster_path)
        except OSError as exc:
            _LOGGER.info("md_trajectory: could not copy poster image: %s", exc)
            poster_path = None
    finally:
        # Best-effort cleanup; tempdir may be locked on Windows etc.
        try:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        except OSError:
            pass

    rel_poster = (
        str(poster_path) if poster_path is not None and poster_path.exists() else None
    )

    return FigureSpec(
        figure_id="08_md_trajectory",
        section="md",
        title=f"MD trajectory animation ({cid})",
        description=(
            f"Per-frame PyMOL render of the {cid} MD trajectory, "
            "stitched with ffmpeg (H.264 / yuv420p)."
        ),
        path=out_path,
        source_files=[traj, top, md_dir / "analysis.json"],
        renderer="pymol+ffmpeg",
        params={
            "candidate_id": cid,
            "style": style,
            "max_frames": int(max_frames),
            "framerate": int(framerate),
            "resolution": [width, height],
            "poster": rel_poster,
        },
    )


__all__ = ["render"]


# Convenience guard for callers that only want to know whether the
# movie path is even possible without instantiating a full ReportArtifacts.
def movie_tools_available() -> bool:
    """True iff both ``pymol`` and ``ffmpeg`` are on PATH."""
    return bool(_which("pymol") or _which("pymolcli") or _which("PyMOL")) and bool(
        _which("ffmpeg")
    )
