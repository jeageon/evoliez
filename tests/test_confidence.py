from pathlib import Path

import pytest

from evoliez.adapters.base import synthetic_structure
from evoliez.adapters.disorder import predict_disorder
from evoliez.config import Backend
from evoliez.features.boltz_features import edge_confidence
from evoliez.features.confidence import (
    norm_plddt,
    plddt_bin,
    residue_confidence,
)
from evoliez.ml.labels import LabelPolicyError, assert_supervised_label_allowed


def test_plddt_helpers():
    assert norm_plddt(95.0) == 0.95
    assert norm_plddt(0.95) == 0.95          # already 0-1
    assert plddt_bin(95) == 3 and plddt_bin(75) == 2
    assert plddt_bin(60) == 1 and plddt_bin(40) == 0


def test_residue_confidence_windows_and_low_segments():
    s = synthetic_structure("ACDEFGHIKLMNPQRSTVWY", seed=3)
    # force a contiguous low-pLDDT stretch
    for r in s.residues[5:9]:
        r.plddt = 30.0
    confs = residue_confidence(s)
    assert len(confs) == len(s.residues)
    low = [c for c in confs if c.index in (6, 7, 8, 9)]
    assert all(c.plddt_bin == 0 for c in low)
    assert max(c.low_seg_len for c in low) == 4
    for c in confs:
        assert 0.0 <= c.plddt_norm <= 1.0


def test_edge_confidence_monotonic():
    hi = edge_confidence(1.0, 0.95, 0.9, 1.0)
    lo_plddt = edge_confidence(1.0, 0.30, 0.9, 1.0)
    lo_pde = edge_confidence(1.0, 0.95, 0.9, 20.0)
    assert hi > lo_plddt and hi > lo_pde
    assert edge_confidence(0.0, 0.95, 0.9, 1.0) == 0.0


def test_disorder_proxy_runs_offline(tmp_path):
    seq = "MKKKKKKKKSSSSSSGGGGNNNNACDEFGHIKLMNPQRSTVWY"
    t = predict_disorder(seq, tmp_path, backend=Backend.mock)
    assert len(t.iupred) == len(seq)
    assert len(t.low_complexity) == len(seq)
    assert all(0.0 <= v <= 1.0 for v in t.iupred)
    # the poly-K / poly-S stretch should look more disordered than the tail
    assert sum(t.iupred[:20]) / 20 >= sum(t.iupred[-20:]) / 20


def test_confidence_features_are_never_labels():
    assert_supervised_label_allowed("kcat")  # experimental still ok
    for bad in ("plddt_norm", "iupred_score", "mobidb_score",
                "disorder_score", "edge_conf", "risk_label",
                "relia_label", "low_complexity_flag"):
        with pytest.raises(LabelPolicyError):
            assert_supervised_label_allowed(bad)
