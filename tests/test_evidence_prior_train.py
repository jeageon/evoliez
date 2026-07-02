"""Trainable multi-head EvidencePrior (ML strategy §3.1)."""

import pytest

from evoliez.ml import evidence_prior as ep
from evoliez.ml.learnability import Row


def _rows():
    """30 synthetic candidates. md_pass is learnable from `conservation` (both classes);
    reference_like is ALL-POSITIVE (the survivorship failure mode -> head must be skipped);
    nac_nonneg is unlabeled. Groups A/B interleaved for a group split."""
    rows = []
    for i in range(30):
        cons = 0.20 + 0.02 * i           # 0.20 .. 0.78
        x = {"conservation": cons, "gap_freq": round(1.0 - cons, 4)}
        labels = {
            "md_pass": 1 if cons < 0.5 else 0,
            "reference_like": 1,          # no negatives -> not learnable
            "nac_nonneg": None,
        }
        rows.append(Row(candidate_id=f"c{i}", mutation_string=f"m{i}", x=x,
                        labels=labels, ml_score=0.5))
    return rows


def _groups():
    return {f"c{i}": ("A" if i % 2 == 0 else "B") for i in range(30)}


def test_fit_trains_learnable_head_and_skips_survivorship_head():
    prior = ep.fit_evidence_prior(_rows(), group_by_id=_groups(), holdout_groups=["B"])
    # md_pass head learns; all-positive reference_like + unlabeled nac are skipped
    assert "P(structural_viable)" in prior.heads
    assert "P(reference_pose_accommodation)" not in prior.heads
    assert "P(reaction_geometry_nonneg)" not in prior.heads
    head = prior.heads["P(structural_viable)"]
    assert head.train_auc is not None and head.train_auc >= 0.9
    assert head.holdout_auc is not None and head.holdout_auc >= 0.6


def test_predictions_follow_the_learned_direction():
    prior = ep.fit_evidence_prior(_rows(), group_by_id=_groups(), holdout_groups=["B"])
    low = prior.predict_row({"conservation": 0.22, "gap_freq": 0.78})
    high = prior.predict_row({"conservation": 0.76, "gap_freq": 0.24})
    # low conservation -> md_pass positive class -> higher score
    assert low["P(structural_viable)"]["score"] > high["P(structural_viable)"]["score"]
    assert low["P(structural_viable)"]["confidence"] in {"high", "medium", "low"}


def test_activity_head_cannot_be_fit():
    bad = [ep.HeadSpec("P(activity_improved)", "md_pass")]
    with pytest.raises(ValueError):
        ep.fit_evidence_prior(_rows(), heads=bad)


def test_expensive_feature_leakage_blocked_at_fit():
    rows = _rows()
    for r in rows:
        r.x["nac_occupancy"] = 0.5   # expensive quantity as a feature -> must raise
    with pytest.raises(Exception):
        ep.fit_evidence_prior(rows)


def test_mixed_label_sources_refused():
    heads = [
        ep.HeadSpec("P(structural_viable)", "md_pass", label_source=ep.COMPUTATIONAL_SURROGATE),
        ep.HeadSpec("P(reference_pose_accommodation)", "reference_like",
                    label_source=ep.EXPERIMENTAL),
    ]
    with pytest.raises(ValueError):
        ep.fit_evidence_prior(_rows(), heads=heads)
