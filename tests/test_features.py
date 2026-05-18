from evoliez.adapters.base import synthetic_structure
from evoliez.config import LigandInput
from evoliez.features.evolutionary import (
    assign_residue_classes,
    compute_position_features,
)
from evoliez.features.geometry import (
    ligand_proximal_residues,
    residue_ligand_contacts,
)
from evoliez.features.ligand import parse_ligand


def test_position_features_conservation_monotonic():
    # column 0 fully conserved, column 1 fully variable
    msa = [
        ("target", "AC"),
        ("h1", "AD"),
        ("h2", "AE"),
        ("h3", "AF"),
    ]
    feats = compute_position_features(msa)
    assert len(feats) == 2
    assert feats[0].conservation_score > feats[1].conservation_score
    assert feats[0].target_position == 1
    assert feats[1].entropy > feats[0].entropy


def test_residue_class_protects_catalytic():
    msa = [("target", "ACDEFGHIK"), ("h1", "ACDEFGHIK"), ("h2", "ACDQFGHIK")]
    feats = compute_position_features(msa)
    assign_residue_classes(feats, catalytic_positions={3})
    by_pos = {f.target_position: f for f in feats}
    assert by_pos[3].residue_class == 1  # catalytic fixed


def test_ligand_parse_and_contacts():
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="CC(=O)O"))
    assert lig.n_heavy >= 1
    struct = synthetic_structure("ACDEFGHIKLMNPQRSTVWY", seed=7)
    # place ligand atoms near residue 1 so a contact is guaranteed
    for a in lig.atoms:
        a.coord = struct.residues[0].sidechain_centroid
    contacts = residue_ligand_contacts(struct, lig.atoms, cutoff=5.0)
    assert any(c.residue_index == 1 for c in contacts)
    prox = ligand_proximal_residues(struct, lig.atoms, radius=8.0)
    assert 1 in prox
