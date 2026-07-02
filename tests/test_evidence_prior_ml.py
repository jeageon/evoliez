"""V3-7: ML Evidence Prior framework — heads, provenance, split policy (ROADMAP_V3 D5)."""
import pytest

from evoliez.ml import evidence_prior as ep


def test_heads_are_priors_not_activity_predictors():
    ep.assert_heads_are_priors()                 # default heads OK
    bad = [ep.HeadSpec("P(kcat_improved)", "md_pass")]
    with pytest.raises(ValueError):
        ep.assert_heads_are_priors(bad)


def test_every_default_head_has_computational_provenance():
    for h in ep.HEAD_SPECS:
        assert h.label_source == ep.COMPUTATIONAL_SURROGATE
        assert h.prohibited_claim == "experimental activity predictor"
        assert h.allowed_claim == "surrogate evidence prior"
        assert h.label_key in __import__("evoliez.ml.learnability",
                                         fromlist=["LABELS"]).LABELS


def test_generalization_claim_blocks_broad_with_few_classes():
    # Stage 1 has 3 enzyme classes -> only a stress test, no broad claim
    g3 = ep.generalization_claim(ep.ENZYME_CLASS, n_enzyme_classes=3)
    assert g3.broad_generalization_allowed is False
    assert "stress test" in g3.note
    g5 = ep.generalization_claim(ep.ENZYME_CLASS, n_enzyme_classes=5)
    assert g5.broad_generalization_allowed is True
    # random split is interpolation only
    assert ep.generalization_claim(ep.RANDOM, 5).allowed_claim == "internal interpolation only"


def test_group_split_no_leak():
    groups = {"v1": "FDH", "v2": "FDH", "v3": "TEM1", "v4": "GLY"}
    train, test = ep.group_split(groups, holdout_groups=["TEM1"])
    assert set(test) == {"v3"}
    assert "v1" in train and "v2" in train
    ep.assert_no_group_leak(train, test, groups)        # no raise


def test_group_leak_detected():
    groups = {"v1": "FDH", "v2": "FDH"}
    # deliberately leak FDH into both
    with pytest.raises(ValueError):
        ep.assert_no_group_leak(["v1"], ["v2"], groups)


def test_feature_contract_still_blocks_md_leakage():
    # an MD/pose label quantity as a feature must still raise (leakage guard intact)
    with pytest.raises(Exception):
        ep.feature_contract_ok(["conservation", "nac_occupancy"])  # nac_occupancy is expensive
    assert ep.feature_contract_ok(["conservation", "gap_freq"]) is True
