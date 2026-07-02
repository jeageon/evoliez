"""Tests for the MD-free functional-state learnability harness (evoliez.ml.learnability)."""
import numpy as np
import pytest

from evoliez.ml import learnability as L


def test_leakage_guard_blocks_expensive_features():
    # an expensive (label-derived) quantity must never be accepted as a feature
    with pytest.raises(ValueError, match="LEAKAGE"):
        L.assert_no_leakage(["conservation", "nac_delta_vs_wt"])
    with pytest.raises(ValueError, match="LEAKAGE"):
        L.assert_no_leakage(["redocking_consistency"])
    # unknown fields are rejected too (must be explicitly declared cheap)
    with pytest.raises(ValueError, match="CHEAP ALLOWLIST"):
        L.assert_no_leakage(["mystery_feature"])
    # cheap ones (incl. distance_to_* prefix) pass
    L.assert_no_leakage(["conservation", "ddg_fold", "distance_to_design_ligand"])


def test_labels():
    assert L.label_reference_like({"pose_gate": {"design_ligand": {"status": "reference_like"}}}) == 1
    assert L.label_reference_like({"pose_gate": {"design_ligand": {"status": "displaced"}}}) == 0
    assert L.label_reference_like({}) is None
    assert L.label_nac_nonneg({"nac_delta_vs_wt": 0.1}) == 1
    assert L.label_nac_nonneg({"nac_delta_vs_wt": -0.2}) == 0
    assert L.label_nac_nonneg({"nac_delta_vs_wt": None}) is None
    assert L.label_md_pass({"passed": True}) == 1
    assert L.label_md_pass({"passed": False}) == 0


def test_auc_basic():
    assert L.auc([3, 2, 1], [1, 1, 0]) == 1.0
    assert L.auc([1, 2, 3], [1, 1, 0]) == 0.0
    assert L.auc([1, 1], [1, 0]) == 0.5  # tie
    assert L.auc([1, 2], [1, 1]) is None  # one class


def test_loo_detects_planted_signal():
    # cheap feature f0 perfectly separates the label; LOO logistic should recover high AUC
    rng = np.random.RandomState(0)
    n = 40
    y = np.array([0] * 20 + [1] * 20, dtype=float)
    f0 = y + rng.normal(0, 0.25, n)          # informative
    f1 = rng.normal(0, 1, n)                  # noise
    X = np.column_stack([f0, f1])
    a = L.loo_logistic_auc(X, y)
    assert a is not None and a > 0.85

    # pure noise -> AUC near chance
    Xn = rng.normal(0, 1, (n, 2))
    an = L.loo_logistic_auc(Xn, y)
    assert an is not None and 0.3 < an < 0.7


def test_degenerate_label_handled():
    rows = [
        L.Row(candidate_id=f"c{i}", mutation_string="A1B",
              x={"conservation": float(i)}, labels={"reference_like": 1},  # all same class
              ml_score=0.5)
        for i in range(10)
    ]
    rep = L.assess(rows)["reference_like"]
    assert rep.loo_cheap_auc is None
    assert "NO variance" in rep.note


def test_assess_smoke_on_synthetic_join():
    rng = np.random.RandomState(1)
    rows = []
    for i in range(30):
        ref = 1 if rng.rand() > 0.5 else 0
        rows.append(L.Row(
            candidate_id=f"c{i}", mutation_string="A1B",
            x={"conservation": ref + rng.normal(0, 0.3),  # signal
               "gap_freq": rng.normal(0, 1)},             # noise
            labels={"reference_like": ref, "nac_nonneg": None, "md_pass": ref},
            ml_score=rng.rand(),
        ))
    reports = L.assess(rows)
    assert reports["reference_like"].n == 30
    assert reports["reference_like"].loo_cheap_auc is not None
    # the informative feature should top the univariate ranking
    assert reports["reference_like"].univariate[0][0] == "conservation"
