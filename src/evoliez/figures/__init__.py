"""HTML report-package generation for EvoLiEZ runs.

This package builds a portable, self-describing HTML bundle from a finished
(or in-progress) pipeline ``run_dir``.  The foundation layer here contains
only data structures, run-artifact discovery, the manifest writer, and
matplotlib styling helpers.  Concrete renderers live in sibling sub-packages
(``html``, ``plots``) that are built by separate agents in later waves.

Heavy optional dependencies (matplotlib, jinja2, pillow) are imported lazily
inside the functions that need them, so simply importing
``evoliez.figures`` is safe on a barebones environment.
"""

from __future__ import annotations

from evoliez.figures.discovery import discover
from evoliez.figures.manifest import ManifestBuilder
from evoliez.figures.style import (
    EVIDENCE_COLORS,
    JOURNAL_STYLES,
    WONG_PALETTE,
    apply_mpl_style,
    evidence_color,
    is_paper_style,
)
from evoliez.figures.types import FigureSpec, ReportArtifacts

__all__ = [
    "discover",
    "ManifestBuilder",
    "FigureSpec",
    "ReportArtifacts",
    "apply_mpl_style",
    "evidence_color",
    "is_paper_style",
    "WONG_PALETTE",
    "EVIDENCE_COLORS",
    "JOURNAL_STYLES",
]
