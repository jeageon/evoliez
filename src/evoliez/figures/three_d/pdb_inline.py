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

# Boltz writes per-residue pLDDT into the B-factor column of the predicted
# complex PDB.  This colorfunc block reproduces the AlphaFold/Boltz
# convention (red=low confidence -> blue=high confidence) so the section-3
# WT-complex viewer can convey predicted-structure confidence at a glance.
# The JS is a complete 3Dmol cartoon style override (replaces the
# default ``cartoon: {color: "spectrum"}``).
_PLDDT_COLOR_JS = (
    "// Boltz puts pLDDT in B-factor column. Color cartoon by it "
    "(low=red, high=blue).\n"
    "v.setStyle({}, {cartoon:{colorfunc:function(atom){\n"
    "    var b = atom.b || 50;\n"
    "    if (b < 50) return 'red';\n"
    "    if (b < 70) return 'orange';\n"
    "    if (b < 90) return 'yellow';\n"
    "    return 'blue';\n"
    "}}});"
)

# Cofactor / ligand resnames that the section-3 close-up should center on.
# Matches the default `resn` list baked into
# ``html/templates/components/threed_viewer.html`` so the close-up zooms
# to the same ligand the cartoon highlights with sticks.
_LIGAND_RESNAMES = ("NDP", "NAP", "LIG", "NAI", "NAD", "SAH", "SAM")


def _build_pocket_zoom_js(zoom_radius: float) -> str:
    """JS snippet that zooms the viewer to the ligand pocket.

    ``zoom_radius`` is a logical Angstrom radius around the ligand; we
    translate it into a 3Dmol ``v.zoom`` factor (the higher the requested
    radius, the smaller the zoom factor; ~8 A -> 1.4x is the close-up the
    expert plan calls for).
    """
    # 3Dmol's zoom() factor is relative to whatever zoomTo() framed.  An
    # 8 A radius around the ligand corresponds to ~1.4x; scale linearly
    # so callers can dial in tighter / looser without thinking about the
    # 3Dmol API.
    factor = max(1.0, 8.0 * 1.4 / float(zoom_radius)) if zoom_radius > 0 else 1.4
    resn_js = json.dumps(list(_LIGAND_RESNAMES))
    return (
        f"v.zoomTo({{resn:{resn_js}}});\n"
        f"v.zoom({factor:.3f});  // ~{zoom_radius:.1f} A close-up around ligand"
    )


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
    color_by: str = "spectrum",
    pocket_zoom: Optional[float] = None,
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
    color_by:
        Cartoon coloring mode.  ``"spectrum"`` (default) keeps the
        legacy rainbow cartoon shared by every viewer; ``"plddt"`` swaps
        in a B-factor / pLDDT colorfunc (red < 50, orange < 70, yellow <
        90, blue >= 90) so the WT Boltz complex conveys per-residue
        confidence; ``"chain"`` colors by chain id.  The generated JS
        snippet is stored under ``color_js`` in the returned dict and
        the choice itself under ``color_by`` so downstream code can
        inspect it.
    pocket_zoom:
        ``None`` (default) keeps the template's full-structure
        ``zoomTo(ligand)`` behavior.  A positive number requests a
        close-up around the ligand at roughly that many Angstroms; the
        generated JS is stored under ``zoom_js`` and the radius under
        ``pocket_zoom``.
    """
    safe_id = _sanitize_viewer_id(viewer_id)
    highlight_json = json.dumps(list(highlight_residues or []))

    color_by_norm = (color_by or "spectrum").strip().lower()
    if color_by_norm not in ("spectrum", "plddt", "chain"):
        _LOGGER.warning(
            "threed.pdb_inline: unknown color_by=%r, falling back to 'spectrum'",
            color_by,
        )
        color_by_norm = "spectrum"
    if color_by_norm == "plddt":
        color_js = _PLDDT_COLOR_JS
    elif color_by_norm == "chain":
        color_js = "v.setStyle({}, {cartoon:{color:'chain'}});"
    else:
        # ``""`` means "use the template default" - the threed_viewer
        # component already calls ``v.setStyle({}, {cartoon: {color:
        # 'spectrum'}})``, so emitting more JS would only add cost.
        color_js = ""

    if pocket_zoom is not None:
        try:
            zoom_val: Optional[float] = float(pocket_zoom)
        except (TypeError, ValueError):
            zoom_val = None
    else:
        zoom_val = None
    zoom_js = _build_pocket_zoom_js(zoom_val) if zoom_val is not None else ""

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
            "color_by": color_by_norm,
            "color_js": color_js,
            "pocket_zoom": zoom_val,
            "zoom_js": zoom_js,
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
            "color_by": color_by_norm,
            "color_js": color_js,
            "pocket_zoom": zoom_val,
            "zoom_js": zoom_js,
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
        "color_by": color_by_norm,
        "color_js": color_js,
        "pocket_zoom": zoom_val,
        "zoom_js": zoom_js,
    }
