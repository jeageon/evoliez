from evoliez.provenance import ArtifactProvenance, ReactiveAtomMap, classify_artifact


def _atom_map():
    return ReactiveAtomMap(
        cofactor_atoms={"C4": 10},
        substrate_atoms={"H": 20},
        catalytic_residue_atoms={"D221": "OD1"},
        validated=True,
    )


def test_full_atom_required():
    artifact = ArtifactProvenance(
        structure_path="x.pdb",
        source_stage="s04",
        backend="real",
        atom_count=100,
        is_full_atom=False,
        chain_map={"A": "A"},
        residue_numbering_map={"A:1": "A:1"},
    )
    result = classify_artifact(artifact, _atom_map())
    assert result.status == "invalid"
    assert any("full-atom" in r for r in result.reasons)
