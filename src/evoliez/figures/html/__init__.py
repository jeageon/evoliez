"""HTML report dashboard for EvoLiEZ pipeline runs.

Build a portable, single-file or zipped HTML report from a finished run::

    from evoliez.figures.html import build_html  # stubbed in wave 2

Templates live in ``templates/``.  Static assets in ``static/``.  Vendor JS
(3Dmol.js) is bundled for offline use so the resulting report opens cleanly
from ``file://`` URLs without network access.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
VENDOR_DIR = STATIC_DIR / "vendor"

__all__ = ["TEMPLATES_DIR", "STATIC_DIR", "VENDOR_DIR"]
