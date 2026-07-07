import pytest

from evoliez.provenance import ReactiveAtomMap


def test_reactive_atom_map_roundtrip():
    atom_map = ReactiveAtomMap(
        cofactor_atoms={"C4": 10},
        substrate_atoms={"hydride": 20},
        catalytic_residue_atoms={"D221": "OD1"},
        validated=True,
    )
    restored = ReactiveAtomMap(**atom_map.model_dump())
    assert restored == atom_map


def test_validated_atom_map_requires_all_terms():
    with pytest.raises(ValueError):
        ReactiveAtomMap(cofactor_atoms={"C4": 10}, validated=True)
