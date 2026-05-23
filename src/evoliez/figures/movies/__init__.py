"""Movie / animation renderers for the EvoLiEZ HTML report.

Currently a single deliverable: :mod:`md_trajectory`, which stitches an
MD trajectory into an mp4 via headless PyMOL + ffmpeg.  All renderers
expose a single :func:`render` entry point matching the same contract
as :mod:`evoliez.figures.plots`::

    def render(artifacts, out_path: Path, *, style: str = "presentation",
               **kwargs) -> Optional[FigureSpec]

Movies are best-effort: when PyMOL or ffmpeg is unavailable (or the
input trajectory is missing), the renderer logs and returns ``None`` so
the HTML builder can degrade to the static frames from
:mod:`evoliez.figures.three_d.md_frames`.
"""

from __future__ import annotations

from evoliez.figures.movies import md_trajectory

__all__ = ["md_trajectory"]
