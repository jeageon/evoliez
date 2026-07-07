"""extra_ligands -> additional Boltz ligand entities (cofactor + substrate)."""

from __future__ import annotations

import yaml

from evoliez.adapters.boltz import predict_complex
from evoliez.config import Backend, ComplexPredictionConfig
from evoliez.types import Ligand


def _spec(tmp_path, extra):
    """Drive the dry-run real path (writes the Boltz input YAML, no binary) and
    return the parsed spec."""
    predict_complex(
        "wt", "ACDEFGHIKLMNPQRSTVWY",
        Ligand(id="NADP", smiles="C1=CC(=O)N", source="rdkit"),
        ComplexPredictionConfig(), tmp_path,
        backend=Backend.real, dry_run=True, extra_ligands=extra,
    )
    return yaml.safe_load((tmp_path / "wt_boltz_input.yaml").read_text())


def test_extra_ligands_become_own_entities(tmp_path):
    formate = Ligand(id="formate", smiles="[O-]C=O", source="rdkit")
    spec = _spec(tmp_path, [formate])
    ligs = [s["ligand"] for s in spec["sequences"] if "ligand" in s]
    assert [g["id"] for g in ligs] == ["B", "C"]            # primary + formate
    assert ligs[0]["smiles"] == "C1=CC(=O)N"               # NADP stays B
    assert ligs[1]["smiles"] == "[O-]C=O"                  # formate is C
    # affinity (when predicted) targets the primary binder, never the extra
    if spec.get("properties"):
        assert spec["properties"][0]["affinity"]["binder"] == "B"


def test_no_extra_ligands_is_unchanged(tmp_path):
    spec = _spec(tmp_path, [])
    ligs = [s["ligand"] for s in spec["sequences"] if "ligand" in s]
    assert [g["id"] for g in ligs] == ["B"]                # single ligand only
