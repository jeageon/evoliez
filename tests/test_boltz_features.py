import json
from pathlib import Path

import pytest

from evoliez.adapters.boltz import predict_complex
from evoliez.config import Backend, ComplexPredictionConfig, LigandInput
from evoliez.features.boltz_features import (
    confidence_weighted_contact,
    ensemble_contacts,
    pose_consensus,
)
from evoliez.features.delta import boltz_delta_features
from evoliez.features.ligand import parse_ligand
from evoliez.ml.datasets import write_datasets
from evoliez.ml.labels import LabelPolicyError, assert_supervised_label_allowed


def _cx(tmp_path, n=5):
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="CC(=O)OP(O)=O"))
    cfg = ComplexPredictionConfig(diffusion_samples=n)
    return predict_complex("t", "ACDEFGHIKLMNPQRSTVWY" * 3, lig, cfg,
                           tmp_path, backend=Backend.mock)


def test_boltz_mock_returns_sorted_ensemble_with_metrics(tmp_path):
    cx = _cx(tmp_path, n=6)
    assert len(cx.samples) == 6
    confs = [s.metrics["confidence_score"] for s in cx.samples]
    assert confs == sorted(confs, reverse=True)  # best sample first
    for k in ("ligand_iptm", "complex_iplddt", "complex_ipde",
              "affinity_pred_value", "ensemble_disagreement"):
        assert k in cx.metrics


def test_confidence_weighted_contact_monotonic():
    hi = confidence_weighted_contact(1.0, 0.9, 90.0, 0.5)
    lo = confidence_weighted_contact(1.0, 0.9, 90.0, 5.0)  # worse PDE
    assert 0.0 <= lo < hi <= 1.0


def test_ensemble_contacts_have_frequency(tmp_path):
    cx = _cx(tmp_path, n=8)
    econ = ensemble_contacts(cx.structure, cx.samples, cutoff=6.0,
                             ligand_iptm=cx.metrics["ligand_iptm"])
    assert econ
    for e in econ:
        assert 0.0 <= e.contact_frequency <= 1.0
        assert 0.0 <= e.confidence_weighted_score <= 1.0
    pc = pose_consensus(cx.samples)
    assert pc["n_samples"] == 8


def test_delta_zero_for_identical_complex(tmp_path):
    cx = _cx(tmp_path)
    d = boltz_delta_features(cx, cx, catalytic_positions=[5, 10])
    assert d["d_ligand_iptm"] == 0.0
    assert d["d_complex_ipde"] == 0.0
    assert d["d_key_distance"] == 0.0


def test_label_policy_blocks_boltz_columns():
    assert_supervised_label_allowed("activity")  # ok
    assert_supervised_label_allowed("kcat")  # ok
    for bad in ("confidence_score", "ligand_iptm", "affinity_pred_value",
                "d_complex_ipde", "contact_frequency"):
        with pytest.raises(LabelPolicyError):
            assert_supervised_label_allowed(bad)


def test_write_datasets_roles(tmp_path):
    tables = {
        "edge_level": [{"residue_index": 1, "ligand_atom_id": "O1",
                        "contact_frequency": 0.8, "weak_contact": 1}],
        "mutation_level": [{"candidate_id": "m0", "ligand_iptm": 0.7,
                            "experimental_label": ""}],
    }
    written = write_datasets(tmp_path, tables)
    assert (tmp_path / "edge_level.csv").exists()
    roles = json.loads((tmp_path / "roles.json").read_text())
    er = roles["column_roles"]["mutation_level"]
    assert "supervised_label" in er["experimental_label"]
    assert "feature" in er["ligand_iptm"]
    assert len(written) == 2
