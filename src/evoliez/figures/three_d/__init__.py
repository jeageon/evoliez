"""3D rendering layer for the EvoLiEZ HTML report.

Two backends are supported, in priority order:

1. **PyMOL** (out-of-process via ``pymol -cq``) - paper-grade PNG renders.
   We call PyMOL as a subprocess, never ``import pymol``, so the (sometimes
   GPL-licensed) PyMOL bindings never enter the EvoLiEZ Python process.
2. **3Dmol.js inline** - a browser-side fallback that embeds the raw PDB
   text in a ``<script type="text/plain">`` tag.  Works on ``file://`` URLs
   so a downloaded report bundle renders without a web server.

Each ``render_*`` helper returns an ``Optional[FigureSpec]``: ``None`` when
either PyMOL is unavailable or the required WT artifact is missing.  The
HTML builder then substitutes the 3Dmol.js viewer (built by
:func:`make_viewer_context`) so the report always has a 3D view.
"""

from __future__ import annotations

from evoliez.figures.three_d.mutation_overlay import render as render_mutation_overlay
from evoliez.figures.three_d.pdb_inline import make_viewer_context
from evoliez.figures.three_d.pocket_view import render as render_pocket_view
from evoliez.figures.three_d.pose_ensemble import render as render_pose_ensemble
from evoliez.figures.three_d.pymol_runner import pymol_available, run_pymol_script

__all__ = [
    "make_viewer_context",
    "pymol_available",
    "run_pymol_script",
    "render_pocket_view",
    "render_pose_ensemble",
    "render_mutation_overlay",
]
