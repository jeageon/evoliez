"""RDKit 2D depiction of the input ligand."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

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


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    catalytic_atoms: Optional[Iterable[int]] = None,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render the ligand as a 2D structural diagram (PNG).

    Returns ``None`` when no SMILES is available or RDKit isn't installed.
    """
    smiles = _ligand_smiles_from_artifacts(artifacts)
    if not smiles:
        _LOGGER.warning("ligand_2d: no SMILES available, skipping")
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
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

    highlight: List[int] = list(catalytic_atoms) if catalytic_atoms else []
    img_size = (500, 500)
    pil_img = Draw.MolToImage(
        mol,
        size=img_size,
        highlightAtoms=highlight or None,
    )

    name = _ligand_name(smiles)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pil_img)
    ax.set_axis_off()
    ax.set_title(f"Ligand: {name}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    sources: List[Path] = []
    if getattr(artifacts, "provenance_json", None):
        sources.append(Path(artifacts.provenance_json))

    return FigureSpec(
        figure_id="01_ligand_2d",
        section="input",
        title=f"Ligand: {name}",
        description="2D structural depiction of the input ligand (RDKit).",
        path=out_path,
        source_files=sources,
        renderer="rdkit",
        params={
            "smiles": smiles,
            "name": name,
            "catalytic_atoms": highlight,
            "style": style,
        },
    )
