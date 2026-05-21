"""Paper-grade close-up of the WT binding pocket with bound ligand.

Renders a single ray-traced PNG via headless PyMOL.  When PyMOL is not
available, or the WT complex PDB is missing, ``render`` returns ``None``
and the HTML builder substitutes the 3Dmol.js inline viewer instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import Template
from typing import Any, Optional

from evoliez.figures.style import style_dpi
from evoliez.figures.three_d.pymol_runner import pymol_available, run_pymol_script
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

# Cofactor / ligand residue names recognized as "the ligand" in the WT
# complex.  Kept in sync with the resn set in the 3Dmol viewer template.
_LIGAND_RESN = "NDP+NAP+LIG+NAI+NAD+SAH+SAM"

# String.Template (``$name``) keeps the PyMOL script readable without the
# brace-style ``.format`` collisions PyMOL selections would otherwise
# trigger ({} is a python-format placeholder *and* a PyMOL selection
# operator).
_SCRIPT = Template(
    """\
load $pdb_path, wt
hide everything
show cartoon, wt and polymer
color cyan, wt and polymer
select lig, wt and resn $ligand_resn
show sticks, lig
color magenta, lig
zoom lig, $pocket_radius
bg_color white
set ray_shadows, 1
set ray_opaque_background, off
set cartoon_transparency, 0.15
ray $width, $height
png $out_path, dpi=$dpi
"""
)


def _resolution_for(dpi: int) -> tuple:
    """Return ``(width, height)`` for the ray-trace at the requested DPI."""
    if dpi >= 500:    # paper
        return (2400, 1800)
    if dpi >= 250:    # poster
        return (2000, 1500)
    return (1600, 1200)  # presentation / fallback


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    pocket_radius: float = 8.0,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render the WT pocket + ligand close-up to ``out_path``.

    Returns ``None`` when PyMOL is missing or the WT complex PDB hasn't
    been generated; the HTML builder falls back to the 3Dmol viewer.
    """
    wt_pdb = getattr(artifacts, "wt_complex_pdb", None)
    if wt_pdb is None or not Path(wt_pdb).exists():
        _LOGGER.info("pocket_view: wt_complex_pdb missing, skipping")
        return None

    if not pymol_available():
        _LOGGER.info("pocket_view: PyMOL not on PATH, deferring to 3Dmol fallback")
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = style_dpi(style)
    width, height = _resolution_for(dpi)
    script = _SCRIPT.substitute(
        pdb_path=str(Path(wt_pdb).resolve()),
        ligand_resn=_LIGAND_RESN,
        pocket_radius=f"{float(pocket_radius):.2f}",
        width=width,
        height=height,
        out_path=str(out_path.resolve()),
        dpi=dpi,
    )

    ok, msg = run_pymol_script(script)
    if not ok or not out_path.exists():
        _LOGGER.warning("pocket_view: PyMOL render failed: %s", msg)
        return None

    return FigureSpec(
        figure_id="03_boltz_binding_pocket",
        section="boltz",
        title="WT binding pocket (PyMOL)",
        description=(
            "Close-up of the wild-type binding pocket with bound ligand "
            "(ray-traced PyMOL render)."
        ),
        path=out_path,
        source_files=[Path(wt_pdb)],
        renderer="pymol",
        params={
            "style": style,
            "pocket_radius": float(pocket_radius),
            "dpi": dpi,
            "resolution": [width, height],
            "ligand_resn": _LIGAND_RESN,
        },
    )
