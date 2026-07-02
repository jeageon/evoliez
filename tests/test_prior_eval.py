"""Prior/selection evaluation — enrichment, FN rate, lane rescue (ML strategy §8)."""

from evoliez.ml.prior_eval import (
    evaluate_prior,
    false_negative_rate,
    false_negative_rescue_by_lane,
    topk_enrichment,
)


def test_topk_enrichment_rewards_hits_at_the_top():
    scores = [8, 7, 6, 5, 4, 3, 2, 1]
    hits = [1, 1, 0, 0, 0, 0, 0, 0]  # both hits at the top
    enr = topk_enrichment(scores, hits, ks=(2, 4))
    assert enr[2] > enr[4] >= 1.0     # tighter top-k is more enriched


def test_false_negative_rate_high_when_hits_are_low_scored():
    scores = [9, 8, 7, 6, 5, 4, 3, 2]
    hits_bottom = [0, 0, 0, 0, 0, 0, 1, 1]
    hits_top = [1, 1, 0, 0, 0, 0, 0, 0]
    assert false_negative_rate(scores, hits_bottom, cutoff_rank=2) == 1.0
    assert false_negative_rate(scores, hits_top, cutoff_rank=2) == 0.0


def test_low_ml_lane_rescue_is_surfaced():
    ml = [9, 8, 7, 1]
    hits = [0, 0, 0, 1]                # the only hit has the lowest ml score
    lanes = ["ml_high", "ml_high", "ml_high", "low_ml_control"]
    rep = false_negative_rescue_by_lane(ml, hits, lanes, cutoff_rank=2)
    low = next(r for r in rep if r.lane == "low_ml_control")
    assert low.hits_below_ml_cut == 1   # a hard ml cut would have dropped this hit


def test_evaluate_prior_bundles_metrics():
    scores = [9, 8, 7, 6, 5, 4, 3, 1]
    hits = [1, 0, 0, 0, 0, 0, 0, 1]
    lanes = ["ml_high"] * 7 + ["low_ml_control"]
    rep = evaluate_prior(scores, hits, cutoff_rank=3, lanes=lanes, ks=(2, 4))
    assert rep.n == 8 and rep.n_hits == 2
    assert rep.fn_rate_at_cut == 0.5    # 1 of 2 hits below the top-3 cut
    assert any(r.lane == "low_ml_control" and r.hits_below_ml_cut == 1 for r in rep.lane_rescue)
