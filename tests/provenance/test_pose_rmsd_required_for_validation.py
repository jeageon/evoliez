from evoliez.provenance import ArtifactProvenance, ReactiveAtomMap, classify_artifact


def test_pose_rmsd_required_for_validation():
    artifact = ArtifactProvenance(
        structure_path="pose.pdb",
        ligand_path="lig.sdf",
        source_stage="s09",
        backend="real",
        atom_count=2000,
        is_full_atom=True,
        chain_map={"A": "A"},
        residue_numbering_map={"A:1": "A:1"},
    )
    atom_map = ReactiveAtomMap(
        cofactor_atoms={"C4": 10},
        substrate_atoms={"hydride": 20},
        catalytic_residue_atoms={"D221": "OD1"},
        validated=True,
    )
    result = classify_artifact(artifact, atom_map, requires_pose_rmsd=True)
    assert result.status == "usable_with_warning"
    assert "pose RMSD source missing" in result.reasons
