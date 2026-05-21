"""Browser-side 3Dmol.js viewer context builders.

These helpers produce the ``viewer`` dict that
``html/templates/components/threed_viewer.html`` consumes.  The PDB text is
embedded inline (``<script type="text/plain">``) so the rendered HTML works
when opened via ``file://`` - no web server, no network fetch.

This module imports nothing heavy and never raises: missing files become a
context with ``pdb_text=""`` and ``missing=True`` so the template can show
a graceful placeholder.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOGGER = logging.getLogger(__name__)

# 1.5 MB - above this we strip ANISOU / extra REMARK lines to keep the
# inlined ``<script>`` block from bloating the HTML report bundle.
_TRIM_THRESHOLD_BYTES = 1_500_000

# Allow at most this many REMARK lines through when trimming - REMARK 1/2/3
# carry the title, resolution, and refinement program, which are useful
# context; the rest are noise for a 3D viewer.
_MAX_REMARK_KEEP = 3

_SAFE_VIEWER_ID = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_viewer_id(viewer_id: str) -> str:
    """Strip any character that isn't safe in an HTML element id."""
    cleaned = _SAFE_VIEWER_ID.sub("_", viewer_id or "")
    return cleaned or "viewer"


def _trim_pdb_text(text: str) -> str:
    """Drop ANISOU and surplus REMARK lines for an oversized PDB.

    Keeps coordinate lines (ATOM/HETATM/TER/END) untouched.  REMARK 1-3 are
    kept so the viewer still has minimal title/resolution metadata.
    """
    kept_remarks = 0
    out_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith("ANISOU"):
            continue
        if line.startswith("REMARK"):
            if kept_remarks >= _MAX_REMARK_KEEP:
                continue
            kept_remarks += 1
        out_lines.append(line)
    # Preserve a trailing newline so PDB readers don't choke on the final
    # END / TER record.
    return "\n".join(out_lines) + "\n"


def make_viewer_context(
    pdb_path: Path,
    *,
    viewer_id: str,
    height: int = 420,
    highlight_residues: Optional[List[int]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the context dict consumed by ``components/threed_viewer.html``.

    Parameters
    ----------
    pdb_path:
        Path to the PDB file to embed.  Missing / unreadable paths produce
        a "missing" context rather than raising.
    viewer_id:
        Element-id suffix - sanitized to ``[a-zA-Z0-9_-]+`` for HTML
        safety.
    height:
        Pixel height of the viewer ``<div>``.
    highlight_residues:
        Optional list of residue numbers to highlight (passed to the
        template as JSON).  ``None`` -> no highlight.
    title:
        Optional caption rendered above the viewer.
    """
    safe_id = _sanitize_viewer_id(viewer_id)
    highlight_json = json.dumps(list(highlight_residues or []))

    p = Path(pdb_path) if pdb_path is not None else None
    if p is None or not p.exists() or not p.is_file():
        _LOGGER.info("threed.pdb_inline: missing PDB at %s", pdb_path)
        return {
            "id": safe_id,
            "height": int(height),
            "pdb_text": "",
            "highlight_residues": highlight_json,
            "title": title,
            "missing": True,
            "source_path": str(pdb_path) if pdb_path is not None else "",
        }

    try:
        raw = p.read_text()
    except OSError as exc:
        _LOGGER.warning("threed.pdb_inline: failed to read %s: %s", p, exc)
        return {
            "id": safe_id,
            "height": int(height),
            "pdb_text": "",
            "highlight_residues": highlight_json,
            "title": title,
            "missing": True,
            "source_path": str(p),
        }

    try:
        size_bytes = p.stat().st_size
    except OSError:
        size_bytes = len(raw.encode("utf-8"))
    trimmed = size_bytes > _TRIM_THRESHOLD_BYTES
    pdb_text = _trim_pdb_text(raw) if trimmed else raw

    return {
        "id": safe_id,
        "height": int(height),
        "pdb_text": pdb_text,
        "highlight_residues": highlight_json,
        "title": title,
        "missing": False,
        "source_path": str(p),
        "trimmed": trimmed,
    }
