"""RDKit 2D depiction of the input ligand.

P1d: atoms are colored by functional group via an enzyme-agnostic SMARTS
catalogue (phosphate / pyridinium / carboxylate / amide / amine /
hydroxyl / aromatic-ring / sugar / adenine-purine). The catalogue is
chosen so common cofactors (NADP+/NADPH/NAD+/FAD/CoA/SAM) and common
substrates (carboxylic acids, alcohols, aromatics) all get a
useful colored breakdown without per-target configuration. ``catalytic_atoms``
still highlights its atoms on top of the functional-group palette in a
distinct fifth color, so the catalytic centre is visually layered, not
overwritten.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

# Ordered functional-group catalogue. Order matters: SMARTS that
# overlap (e.g. amide vs amine) are resolved by first-match-wins, so
# put more-specific patterns earlier. Colors use the Wong colorblind-safe
# palette so they compose with the rest of the figures.
_FUNCTIONAL_GROUPS: List[Dict[str, str]] = [
    # Phosphates first (they live inside cofactors and beat the bare
    # hydroxyl pattern).
    {"name": "phosphate",      "smarts": "[P](=O)([O,OH])[O,OH]",
     "color": "#E69F00"},  # orange
    # Pyridinium - the NADP+/NAD+ redox handle. The bare aromatic-N+
    # pattern is enough; we don't need full nicotinamide.
    {"name": "pyridinium",     "smarts": "[n+]",
     "color": "#D55E00"},  # vermilion
    # Amide bonds (peptide / nicotinamide backbone).
    {"name": "amide",          "smarts": "[NX3][CX3](=O)",
     "color": "#CC79A7"},  # reddish-purple
    # Carboxylate / carboxylic acid (substrate side, e.g. formate).
    {"name": "carboxylate",    "smarts": "[CX3](=O)[OX1H0-,OX2H1]",
     "color": "#F0E442"},  # yellow
    # Adenine / purine ring system - the cofactor "tag" half.
    {"name": "purine",         "smarts": "c1ncnc2[nH0]cnc12",
     "color": "#0072B2"},  # blue
    # Furanose ring (ribose / deoxyribose).
    {"name": "sugar (furanose)", "smarts": "[CX4]1[OX2][CX4][CX4][CX4]1",
     "color": "#009E73"},  # bluish-green
    # Primary / secondary alcohol (catch-all polyol on sugars).
    {"name": "hydroxyl",       "smarts": "[OX2H]",
     "color": "#56B4E9"},  # sky blue
    # Primary amine (substrate side, e.g. amino acid).
    {"name": "amine",          "smarts": "[NX3;H2,H1;!$(NC=O)]",
     "color": "#999999"},  # neutral grey
]
_CATALYTIC_COLOR = "#C00000"  # deep red, reserved for catalytic_atoms overlay

# A small lookup so the rendered title says "NADP+" rather than the raw
# SMILES.  The pipeline's cofactor library would be authoritative but we
# don't want a hard import dependency on it from the plotting layer.
_SMILES_TO_NAME = {
    # NADP+ canonical SMILES (the pipeline's cofactor library uses this form)
    "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)c1": "NADP+",
    # NADPH (reduced)
    "NC(=O)C1=CN(C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(OP(=O)(O)O)C3O)C(O)C2O)CC=C1": "NADPH",
    # NAD+/NADH appear in some assays too
    "NC(=O)c1ccc[n+](C2OC(COP(=O)(O)OP(=O)(O)OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O)C(O)C2O)c1": "NAD+",
    "CCO": "ethanol",
}


def _ligand_name(smiles: str) -> str:
    if smiles in _SMILES_TO_NAME:
        return _SMILES_TO_NAME[smiles]
    # Fall back to the raw SMILES, truncated so the title stays one line.
    return smiles if len(smiles) <= 40 else smiles[:37] + "..."


def _ligand_smiles_from_artifacts(artifacts: ReportArtifacts) -> Optional[str]:
    """Resolve the ligand SMILES from the artifact catalog or provenance."""
    smiles = getattr(artifacts, "ligand_smiles", None)
    if smiles:
        return str(smiles)
    prov = getattr(artifacts, "provenance_json", None)
    if prov is None:
        return None
    try:
        data = json.loads(Path(prov).read_text())
    except (OSError, ValueError):
        return None
    # Provenance shape is loose - probe a couple of common keys.
    for key in ("ligand_smiles", "smiles"):
        if isinstance(data.get(key), str):
            return data[key]
    ligand = data.get("ligand")
    if isinstance(ligand, dict) and isinstance(ligand.get("smiles"), str):
        return ligand["smiles"]
    return None


def _match_functional_groups(
    mol: Any,
) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
    """First-match-wins SMARTS scan over ``_FUNCTIONAL_GROUPS``.

    Returns ``(atom_to_color, legend_entries)`` where:
      * ``atom_to_color`` maps RDKit atom index -> hex color string,
      * ``legend_entries`` is the list of groups that actually matched
        (deduped, in catalogue order) so the figure legend only shows
        present groups.
    """
    from rdkit import Chem  # noqa: WPS433
    atom_to_color: Dict[int, str] = {}
    present: List[Dict[str, Any]] = []
    for entry in _FUNCTIONAL_GROUPS:
        patt = Chem.MolFromSmarts(entry["smarts"])
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        hit_any = False
        for tup in matches:
            for atom_idx in tup:
                if atom_idx in atom_to_color:
                    continue  # first-match wins
                atom_to_color[atom_idx] = entry["color"]
                hit_any = True
        if hit_any:
            present.append(entry)
    return atom_to_color, present


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    catalytic_atoms: Optional[Iterable[int]] = None,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render the ligand as a 2D structural diagram (PNG) with
    functional-group color highlights and a legend strip.

    Returns ``None`` when no SMILES is available or RDKit isn't installed.
    """
    smiles = _ligand_smiles_from_artifacts(artifacts)
    if not smiles:
        _LOGGER.warning("ligand_2d: no SMILES available, skipping")
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        _LOGGER.warning("ligand_2d: rdkit not installed, skipping")
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        _LOGGER.warning("ligand_2d: failed to parse SMILES %r", smiles)
        return None
    # 2D coords for a clean depiction.
    AllChem.Compute2DCoords(mol)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433
    from matplotlib.patches import Patch  # noqa: WPS433

    # Functional-group palette + catalytic-atom overlay.
    fg_color, present_groups = _match_functional_groups(mol)
    catalytic_list: List[int] = list(catalytic_atoms) if catalytic_atoms else []
    # Catalytic colour overrides functional-group colour so the catalytic
    # centre always reads loud and clear (e.g. the C4 hydride donor on
    # nicotinamide).
    final_color: Dict[int, Tuple[float, float, float]] = {}
    for idx, hex_c in fg_color.items():
        final_color[idx] = _hex_to_rgb(hex_c)
    for idx in catalytic_list:
        final_color[idx] = _hex_to_rgb(_CATALYTIC_COLOR)

    # MolToImage ignores per-atom colors (it only honours a single
    # highlightColor). rdMolDraw2D's Cairo backend respects the per-atom
    # dict so the functional-group palette actually reaches the PNG.
    highlight_atoms = list(final_color.keys())
    img_w, img_h = 520, 520
    drawer = rdMolDraw2D.MolDraw2DCairo(img_w, img_h)
    opts = drawer.drawOptions()
    opts.fillHighlights = True
    opts.highlightRadius = 0.32
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms or [],
        highlightAtomColors=final_color or {},
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    from io import BytesIO  # noqa: WPS433
    from PIL import Image  # noqa: WPS433
    pil_img = Image.open(BytesIO(png_bytes))

    name = _ligand_name(smiles)
    # Figure height includes a thin strip below the molecule for the
    # legend so it never overlaps the structure.
    fig, (ax_mol, ax_leg) = plt.subplots(
        2, 1, figsize=(5.4, 6.0),
        gridspec_kw={"height_ratios": [10, 1.4], "hspace": 0.05},
    )
    ax_mol.imshow(pil_img)
    ax_mol.set_axis_off()
    ax_mol.set_title(f"Ligand: {name}")

    # Legend strip: one swatch per present functional group + the
    # catalytic-atom marker if any catalytic atoms were highlighted.
    handles: List[Patch] = []
    for entry in present_groups:
        handles.append(
            Patch(facecolor=entry["color"], edgecolor="black",
                  linewidth=0.4, label=entry["name"])
        )
    if catalytic_list:
        handles.append(
            Patch(facecolor=_CATALYTIC_COLOR, edgecolor="black",
                  linewidth=0.4, label="catalytic atom")
        )
    ax_leg.set_axis_off()
    if handles:
        ax_leg.legend(
            handles=handles,
            loc="center", ncol=min(4, len(handles)),
            fontsize=8, frameon=False,
        )
    else:
        # No SMARTS matched (e.g. very simple ligand like 'CCO' with
        # only hydroxyl, which IS matched; but defensive in case
        # someone disables the catalogue).
        ax_leg.text(
            0.5, 0.5, "(no functional groups matched)",
            ha="center", va="center", fontsize=8, color="#555",
            transform=ax_leg.transAxes,
        )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    sources: List[Path] = []
    if getattr(artifacts, "provenance_json", None):
        sources.append(Path(artifacts.provenance_json))

    return FigureSpec(
        figure_id="01_ligand_2d",
        section="input",
        title=f"Ligand: {name}",
        description=(
            "2D structural depiction of the input ligand (RDKit) with "
            "functional-group color highlights. Catalytic atoms (when "
            "supplied by the caller) are layered on top in deep red."
        ),
        path=out_path,
        source_files=sources,
        renderer="rdkit",
        params={
            "smiles": smiles,
            "name": name,
            "catalytic_atoms": catalytic_list,
            "functional_groups": [g["name"] for g in present_groups],
            "n_atoms_highlighted": len(final_color),
            "style": style,
        },
    )


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    """RDKit Draw wants RGB floats in [0,1]."""
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)
