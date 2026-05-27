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
#
# Render style (user-requested 2026-05-27, publication-grade pocket
# view): semi-transparent grey ribbon as topology backdrop, ligand
# drawn as element-coloured sticks (`util.cbag` = green carbons + std
# element colours), pocket residues within $contact_radius A as cyan
# sticks with one-letter+number labels (e.g. "K288") on their CA
# atoms, and yellow dashed polar-contact lines up to
# $distance_cutoff A.  Replaces the earlier "magenta-on-cyan-cartoon"
# style — the new one identifies individual contact residues in a
# single panel without losing the protein context.
_SCRIPT = Template(
    """\
bg_color white
load $pdb_path, wt

# Whole structure: grey ribbon, dialed-down opacity as backdrop.
hide everything
show cartoon, wt and polymer
color gray90, wt and polymer
set cartoon_transparency, $cartoon_transparency

# Ligand: element-coloured sticks (util.cbag = green C + std heteroatoms).
select my_ligand, wt and resn $ligand_resn
show sticks, my_ligand
util.cbag my_ligand

# Pocket residues: every residue with any atom within $contact_radius A
# of the ligand heavy atoms.  byres + br. picks WHOLE residues, not
# just contact atoms; `polymer` excludes the ligand itself reflexively.
select near_ligands, byres (br. (my_ligand around $contact_radius) and not my_ligand) and polymer
show sticks, near_ligands
color cyan, near_ligands

# One-letter+residue-number labels on CA of pocket residues.
label near_ligands and name CA, oneletter+resi
set label_color, black
set label_size, $label_size
set label_font_id, 7
set label_position, [2.5, 1.5, 0.5]

# Polar-contact dashes (mode=2) up to $distance_cutoff A.  Yellow,
# no per-line labels (the residue labels are enough).
dist ligand_contacts, my_ligand, near_ligands, mode=2, cutoff=$distance_cutoff
set dash_color, yellow, ligand_contacts
set dash_width, 2.0, ligand_contacts
set dash_gap, 0.3, ligand_contacts
hide labels, ligand_contacts

# Camera: orient on ligand+pocket envelope, then zoom out by
# $pocket_radius A so the pocket geometry stays in view.
orient my_ligand or near_ligands
zoom my_ligand, $pocket_radius

# Ray-trace.
set ray_shadows, 1
set ray_opaque_background, off
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


def _label_size_for(style: str) -> int:
    """Bigger labels for poster/presentation, smaller for tight paper figs."""
    if style == "paper":
        return 14
    if style == "poster":
        return 28
    return 24  # presentation default


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    pocket_radius: float = 8.0,
    contact_radius: float = 4.0,
    distance_cutoff: float = 5.0,
    cartoon_transparency: float = 0.7,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render the WT pocket + ligand close-up to ``out_path``.

    Style highlights (user-requested publication template):
      - Whole protein as grey ribbon at ``cartoon_transparency`` opacity
        (0.0 = solid, 1.0 = fully transparent).  Default 0.7 keeps the
        topology visible but lets the pocket atoms dominate.
      - Ligand sticks coloured by element (PyMOL ``util.cbag``).
      - Pocket residues within ``contact_radius`` A of the ligand as
        cyan sticks, labelled with one-letter + residue number.
      - Yellow dashed polar contacts up to ``distance_cutoff`` A.

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
    label_size = _label_size_for(style)
    # PyMOL clamps cartoon_transparency to [0, 1]; clip defensively so
    # an out-of-range kwarg doesn't break the script.
    clipped_transp = max(0.0, min(1.0, float(cartoon_transparency)))
    script = _SCRIPT.substitute(
        pdb_path=str(Path(wt_pdb).resolve()),
        ligand_resn=_LIGAND_RESN,
        pocket_radius=f"{float(pocket_radius):.2f}",
        contact_radius=f"{float(contact_radius):.2f}",
        distance_cutoff=f"{float(distance_cutoff):.2f}",
        cartoon_transparency=f"{clipped_transp:.2f}",
        label_size=label_size,
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
            "Close-up of the wild-type binding pocket: ligand as "
            "element-coloured sticks, pocket residues within "
            f"{contact_radius:g} A as cyan sticks (labelled), and "
            f"polar contacts up to {distance_cutoff:g} A as yellow "
            "dashes (ray-traced PyMOL render)."
        ),
        path=out_path,
        source_files=[Path(wt_pdb)],
        renderer="pymol",
        params={
            "style": style,
            "pocket_radius": float(pocket_radius),
            "contact_radius": float(contact_radius),
            "distance_cutoff": float(distance_cutoff),
            "cartoon_transparency": clipped_transp,
            "label_size": label_size,
            "dpi": dpi,
            "resolution": [width, height],
            "ligand_resn": _LIGAND_RESN,
        },
    )
