"""Regression tests for the fixed-charge cofactor path (the -3 NADP that on-the-fly
AM1-BCC cannot parameterise). These guard the anti-s06 invariant: a mis-charged or
mis-built cofactor must FAIL LOUDLY, never silently mis-assign charges or skip.

RDKit-only (the OpenFF/OpenMM injection is exercised by the server smoke run, not here).
"""
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem

from evoliez.config import LigandInput
from evoliez.features.ligand import parse_ligand
from evoliez.md import charges as C
from evoliez.types import Ligand

PARAMS = Path(__file__).resolve().parent.parent / "params" / "nadp_3minus"
NAP_MOL2 = PARAMS / "NAP.fixed.mol2"
NAP_SDF = PARAMS / "NAP.fixed.ref.sdf"
_have_params = NAP_MOL2.exists() and NAP_SDF.exists()


# --- plumbing: config charge fields reach the Ligand ----------------------------
def test_charge_fields_flow_from_config():
    spec = LigandInput(id="cof", type="smiles", value="[O-]C=O",
                       net_charge=-1, charges_mol2="x.mol2", allow_am1bcc=False)
    lig = parse_ligand(spec)
    assert lig.charges_mol2 == "x.mol2"
    assert lig.allow_am1bcc is False
    assert lig.formal_charge == -1


def test_default_ligand_is_backward_compatible():
    lig = parse_ligand(LigandInput(id="d", type="smiles", value="CCO"))
    assert lig.charges_mol2 is None
    assert lig.allow_am1bcc is True


# --- guard: which ligands must use a fixed-charge template -----------------------
def test_guard_flags_high_risk_cofactor():
    assert C.requires_fixed_charge_template(Ligand(id="x", smiles="", charges_mol2="y"))
    assert C.requires_fixed_charge_template(Ligand(id="x", smiles="", allow_am1bcc=False))
    assert C.requires_fixed_charge_template(Ligand(id="NADP_cofactor", smiles=""))


def test_guard_allows_small_neutral_ligand():
    # formate-like: small, only -1 -> on-the-fly AM1-BCC is fine
    assert not C.requires_fixed_charge_template(
        Ligand(id="formate", smiles="", formal_charge=-1))
    assert not C.requires_fixed_charge_template(
        Ligand(id="drug", smiles="", formal_charge=0))


# --- reference + transfer (need the vendored NADP params) ------------------------
@pytest.mark.skipif(not _have_params, reason="NADP params not vendored")
def test_reference_loads_and_sums_to_minus3():
    ref, q = C.load_reference(NAP_SDF, NAP_MOL2)
    assert ref.GetNumAtoms() == 73
    assert abs(sum(q) + 3.0) < 0.05            # AM1-BCC rounding


@pytest.mark.skipif(not _have_params, reason="NADP params not vendored")
def test_transfer_normalises_to_exact_net_charge():
    ref, q = C.load_reference(NAP_SDF, NAP_MOL2)
    out = C.transfer_charges(ref, ref, q, -3)  # self-transfer -> identity map
    assert len(out) == ref.GetNumAtoms()
    assert abs(sum(out) - (-3.0)) < 1e-6       # exact, not just rounded


@pytest.mark.skipif(not _have_params, reason="NADP params not vendored")
def test_transfer_raises_on_a_different_molecule():
    # The anti-s06 guarantee: charges are NEVER guessed onto the wrong molecule.
    ref, q = C.load_reference(NAP_SDF, NAP_MOL2)
    formate = Chem.AddHs(Chem.MolFromSmiles("[O-]C=O"))
    with pytest.raises(ValueError, match="graph match incomplete|element mismatch"):
        C.transfer_charges(formate, ref, q, -3)
