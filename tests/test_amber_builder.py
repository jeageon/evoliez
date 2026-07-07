"""V6-1 unit tests for the Amber system builder pure helpers.

The subprocess build (tleap/antechamber/parmchk2/parmed) is server-only and is
exercised by scripts/build_amber_system.py against the real CAR complex. Here we
lock the PURE logic that governs correctness: reproducible fingerprint, the tleap
deck, curated-charge validation (fail-loud), and — the crux — resolving the
reactive atoms (ATP Pα / leaving O) onto the right atom names.
"""
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem

from evoliez.adapters.amber_builder import (
    AmberBuildError, AmberBuildSpec, LigandBuildSpec, MetalBuildSpec,
    ReactiveBuildSpec, _one_letter, parse_mol2_atom_names, reactive_atom_names,
    system_fingerprint, tleap_amber_script, validate_mol2_net_charge,
)

PARAMS = Path(__file__).resolve().parent.parent / "params" / "atp_4minus"
ATP_MOL2 = PARAMS / "ATP.fixed.mol2"
ATP_SDF = PARAMS / "ATP.fixed.ref.sdf"


def _car_spec(protein_ff="ff14SB"):
    return AmberBuildSpec(
        ligands=[
            LigandBuildSpec(id="LIG", role="design_ligand", smiles="OCCC(=O)[O-]",
                            net_charge=-1, allow_am1bcc=True),
            LigandBuildSpec(id="ATP", role="cofactor", net_charge=-4,
                            allow_am1bcc=False,
                            charges_mol2="params/atp_4minus/ATP.fixed.mol2"),
        ],
        metal=MetalBuildSpec(enabled=True, ion="MG", element="MG",
                             ion_frcmod="frcmod.ions234lm_1264_tip3p"),
        reactive=ReactiveBuildSpec(donor_smarts="[OX1-]",
                                   acceptor_smarts="[PX4]([OX2][CX4])",
                                   transfer_is_h=False),
        catalytic_residues=["S268", "T269", "K273"],
        protein_ff=protein_ff, water="tip3p", solvent="explicit",
    )


def test_fingerprint_deterministic_and_sensitive():
    spec = _car_spec()
    src = {"LIG": "am1bcc_gaff2", "ATP": "curated:ATP.fixed.mol2"}
    fp = system_fingerprint(spec, "MSEQ", src)
    assert fp == system_fingerprint(spec, "MSEQ", src)          # deterministic
    assert fp != system_fingerprint(_car_spec("ff19SB"), "MSEQ", src)   # FF-sensitive
    assert fp != system_fingerprint(spec, "OTHERSEQ", src)      # sequence-sensitive


def test_tleap_deck_car_explicit_metal():
    spec = _car_spec()
    deck = tleap_amber_script(spec, [("LIG", "LIG.mol2"), ("ATP", "ATP.mol2")])
    assert "source leaprc.protein.ff14SB" in deck
    assert "source leaprc.gaff2" in deck
    assert "source leaprc.water.tip3p" in deck
    assert "loadamberparams frcmod.ions234lm_1264_tip3p" in deck   # 12-6-4 divalent
    assert "combine {prot LIG ATP}" in deck
    assert "solvateOct comp TIP3PBOX 12.0" in deck
    assert "addIons comp Na+ 0" in deck and "addIons comp Cl- 0" in deck
    assert "saveamberparm comp complex.prmtop complex.inpcrd" in deck


@pytest.mark.skipif(not ATP_MOL2.exists(), reason="ATP curated mol2 not present")
def test_curated_mol2_charge_validation():
    text = ATP_MOL2.read_text()
    names = parse_mol2_atom_names(text)
    assert len(names) == 43 and "P1" in names
    assert abs(validate_mol2_net_charge(text, -4, "ATP") + 4) < 0.01
    with pytest.raises(AmberBuildError):          # mis-declared charge => fail loud
        validate_mol2_net_charge(text, -3, "ATP")


@pytest.mark.skipif(not (ATP_MOL2.exists() and ATP_SDF.exists()),
                    reason="ATP curated template not present")
def test_reactive_atom_resolution_atp_alpha_p_and_leaving(tmp_path):
    spec = _car_spec().reactive
    atp_names = parse_mol2_atom_names(ATP_MOL2.read_text())
    # 3-HP design ref
    m = Chem.AddHs(Chem.MolFromSmiles("OCCC(=O)[O-]"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    hp_sdf = tmp_path / "hp.sdf"
    with Chem.SDWriter(str(hp_sdf)) as w:
        w.write(m)
    hp_heavy = Chem.RemoveHs(m)
    design_names = [f"{a.GetSymbol()}{i}" for i, a in enumerate(hp_heavy.GetAtoms())]
    res = reactive_atom_names(spec, hp_sdf, design_names, ATP_SDF, atp_names)
    # ATP alpha-phosphate is P1 (bonded to the ribose ester O + bridging to Pβ);
    # the leaving O is the bridging O5 toward Pβ.
    assert res["P_alpha"] == ("acceptor", "P1")
    assert res["O_leaving"][0] == "acceptor"
    assert res["O_nuc"][0] == "design"        # a 3-HP carboxylate O


def test_one_letter():
    assert _one_letter("SER") == "S"
    assert _one_letter("LYS") == "K"
    assert _one_letter("HIE") == "H"          # Amber protonation variant
    assert _one_letter("UNK") == "X"
