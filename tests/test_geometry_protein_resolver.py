"""V3-2: protein-atom resolver in the geometry layer (ROADMAP_V3 D1).

A catalytic-residue contact term resolves to a PROTEIN atom (not just a ligand SMARTS
match) and scores occupancy. Protein-only terms need no rdkit, so this runs in the
light env.
"""
import numpy as np

from evoliez.md.geometry_spec import (
    GeometryTerm, evaluate_geometry_term, make_protein_resolver,
)
from evoliez.mechanism import MechanismSpec
from evoliez.mechanism.spec import ReactionInfo, ReactionState


def test_protein_resolver_tokens():
    residues = [
        {"resname": "ARG", "seqid": 290, "atoms": {"NH1": 10, "NH2": 11, "CZ": 12}},
        {"resname": "SER", "seqid": 70, "atoms": {"OG": 4, "CB": 3}},
    ]
    r = make_protein_resolver(residues)
    assert r("ARG290", "NH1") == 10
    assert r("290", "CZ") == 12            # bare seqid
    assert r("SER70", "OG") == 4
    assert r("ARG290", "ZZ") is None       # missing atom
    assert r("HIS50", "ND1") is None       # missing residue


def test_protein_contact_term_scores_occupancy():
    # two atoms: index 0 = formate carboxylate O (ligand, but we resolve it as a
    # protein-style fixed index here for a rdkit-free test); index 10 = ARG290 NH1.
    # 3 frames: distances 2.8, 3.0, 6.0 Å -> 2/3 within 3.5 Å.
    frames = [
        np.array([[0.0, 0.0, 0.0]] + [[0, 0, 0]] * 9 + [[2.8, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]] + [[0, 0, 0]] * 9 + [[3.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]] + [[0, 0, 0]] * 9 + [[6.0, 0.0, 0.0]]),
    ]
    resolver = make_protein_resolver(
        [{"resname": "GLY", "seqid": 1, "atoms": {"CA": 0}},
         {"resname": "ARG", "seqid": 290, "atoms": {"NH1": 10}}])
    term = GeometryTerm(
        kind="distance", label="formate_carboxylate_stabilization",
        a_residue="GLY1", a_atom="CA",          # stand-in ligand anchor
        b_residue="ARG290", b_atom="NH1",
        distance_min=0.0, distance_max=3.5)
    res = evaluate_geometry_term(frames, mol_blocks=[], term=term, protein_resolver=resolver)
    assert res.status == "ok"
    assert res.atoms == {"a": 0, "b": 10}
    assert abs(res.occupancy - 2 / 3) < 1e-9


def test_missing_protein_atom_is_honest_skip():
    term = GeometryTerm(kind="distance", a_residue="ARG290", a_atom="NH1",
                        b_residue="HIS999", b_atom="ND1")
    resolver = make_protein_resolver(
        [{"resname": "ARG", "seqid": 290, "atoms": {"NH1": 10}}])
    res = evaluate_geometry_term([np.zeros((11, 3))], [], term, resolver)
    assert res.status == "skipped_missing_atoms"   # missing partner -> not satisfaction


def test_acyl_substitution_template_emits_protein_term():
    # the nucleophilic_acyl_substitution template uses a catalytic Ser Oγ protein atom
    spec = MechanismSpec(
        reaction=ReactionInfo(**{"class": "nucleophilic_acyl_substitution"}),
        reaction_state=ReactionState(protonation_model="propka",
                                     conformational_state="acyl_enzyme"),
    )
    terms = spec.to_geometry_terms()
    nuc = next(t for t in terms if t.label == "nucleophile_carbonyl")
    assert nuc.a_residue == "SER" and nuc.a_atom == "OG"
