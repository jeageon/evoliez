"""Packaging layer for EvoLiEZ visual reports.

Two distribution modes are supported:

* **Linked ZIP** (:func:`build_zip`) -- preserves the on-disk layout
  ``visual_report.html`` + ``figures/`` + ``movies/`` + ``static/`` + ``pdb/``
  inside a single portable archive.  Best when the recipient is going to
  unpack and browse locally; keeps file sizes reasonable.
* **Self-contained HTML** (:func:`build_self_contained`) -- inlines every
  external asset (images as base64, CSS / JS as inline blocks, iframes as
  ``data:text/html;base64`` URLs) so the report is a single ``.html`` file
  that opens on any machine without any sibling folder.  Can be large
  (50-200 MB) when the run includes movies.

A safety helper, :func:`apply_mock_watermark`, injects a red "MOCK DATA"
banner into reports built from a mock backend run so they cannot be
confused with publication-grade output.  The companion
:func:`stamp_figures_with_mock` overlays the same warning on PNGs when
Pillow is available.
"""

from .bundler import build_zip
from .selfcontained import build_self_contained
from .watermark import apply_mock_watermark, stamp_figures_with_mock

__all__ = [
    "build_zip",
    "build_self_contained",
    "apply_mock_watermark",
    "stamp_figures_with_mock",
]
