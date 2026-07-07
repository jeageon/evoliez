"""GNN checkpoint must be pinned by CONTENT, not just by path (ultra-review
#17). An in-place `train-gnn` overwrites the same path with different weights;
recording only the path string yields identical provenance/fingerprint and an
uninvalidated resume despite a different model driving ml_score."""

from evoliez.io.provenance import build_provenance


def test_provenance_hashes_checkpoint_bytes(tmp_path):
    ck = tmp_path / "evoligand_gnn.pt"
    ck.write_bytes(b"WEIGHTS-A")
    a = build_provenance(sequence="ACDE", ligand_smiles="CCO",
                         config_dict={"x": 1}, seed=1, backend="mock",
                         gnn_ckpt=str(ck))
    ck.write_bytes(b"WEIGHTS-B")          # in-place retrain, SAME path
    b = build_provenance(sequence="ACDE", ligand_smiles="CCO",
                         config_dict={"x": 1}, seed=1, backend="mock",
                         gnn_ckpt=str(ck))
    assert a["gnn_checkpoint"] == b["gnn_checkpoint"]            # same path
    assert a["gnn_checkpoint_sha256"] != b["gnn_checkpoint_sha256"]  # diff content


def test_provenance_checkpoint_hash_none_when_absent():
    p = build_provenance(sequence="A", ligand_smiles="C",
                         config_dict={}, seed=1, backend="mock", gnn_ckpt=None)
    assert p["gnn_checkpoint_sha256"] is None
