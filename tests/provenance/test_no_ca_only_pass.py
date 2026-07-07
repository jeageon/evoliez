from evoliez.provenance import ArtifactProvenance, classify_artifact


def test_no_ca_only_pass():
    artifact = ArtifactProvenance(
        structure_path="ca_only.pdb",
        source_stage="s09",
        backend="real",
        atom_count=300,
        is_full_atom=False,
    )
    assert classify_artifact(artifact).status != "valid"
