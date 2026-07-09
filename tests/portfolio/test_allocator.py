"""Tests for the V7 multi-fidelity allocator (portfolio.allocator).

The bundle is built directly from the ledger contract (bands / q-values / control + probe
flags set by hand) — no sibling module is imported — so the allocator is exercised against
the frozen keystone only.
"""
from __future__ import annotations

from evoliez.portfolio.allocator import (
    TierAssignment, TierBudget, TierPlan, allocate_tiers, non_converged_pmf_gate,
)
from evoliez.portfolio.ledger import (
    AXIS_EVOLUTIONARY, AXIS_LIGAND, AXIS_MECHANISM, AXIS_STRUCTURAL, AXIS_UNCERTAINTY,
    BAND_EXPLORATORY, BAND_SIGNIFICANT, BAND_STRONG, BAND_UNRESOLVED, TIER_CHEAP,
    TIER_EXPENSIVE, TIER_FOCUSED, TIER_GPU_BROAD, AxisEvidenceV7, EvidenceLedgerV7,
    LedgerBundle,
)


def _axis(axis, *, band=BAND_UNRESOLVED, q=None, score=0.5):
    return AxisEvidenceV7(axis=axis, score=score, band=band, q_value=q,
                          confidence="medium")


def _led(vid, axes=None, **kw):
    led = EvidenceLedgerV7(variant_id=vid, mutation=f"{vid}_mut",
                           axes={a.axis: a for a in (axes or [])}, **kw)
    led.recompute_overall_band()
    return led


def _bundle():
    """20 candidates: 2 strong-band, 3 controls, 3 significant, 2 consensus, 3 low-q,
    2 deconvolution probes, 2 learning-value, 3 cheap."""
    ledgers = []
    # 2 strong-band mechanism hypotheses -> reaction core
    for i in range(2):
        ledgers.append(_led(f"strong_{i}",
                            [_axis(AXIS_MECHANISM, band=BAND_STRONG, q=0.001, score=0.95)]))
    # 3 controls -> broad GPU only
    for i in range(3):
        ledgers.append(_led(f"control_{i}", [_axis(AXIS_STRUCTURAL)],
                            is_control=True, control_role="wt_parental"))
    # 3 significant (not strong) -> focused MD
    for i in range(3):
        ledgers.append(_led(f"sig_{i}",
                            [_axis(AXIS_MECHANISM, band=BAND_SIGNIFICANT, q=0.02, score=0.8)]))
    # 2 cross-axis consensus -> focused MD
    for i in range(2):
        ledgers.append(_led(
            f"cons_{i}",
            [_axis(AXIS_STRUCTURAL, q=0.05), _axis(AXIS_EVOLUTIONARY, q=0.06)],
            consensus_axes=[AXIS_STRUCTURAL, AXIS_EVOLUTIONARY]))
    # 3 low-q on a single axis but no significant band -> broad GPU only
    for i in range(3):
        ledgers.append(_led(f"lowq_{i}", [_axis(AXIS_LIGAND, band=BAND_UNRESOLVED, q=0.04)]))
    # 2 deconvolution probes (not significant) -> broad GPU only
    for i in range(2):
        ledgers.append(_led(f"deconv_{i}", [_axis(AXIS_STRUCTURAL)],
                            deconvolution_of="strong_0"))
    # 2 high learning-value (exploratory uncertainty axis) -> broad GPU only
    for i in range(2):
        ledgers.append(_led(f"learn_{i}",
                            [_axis(AXIS_UNCERTAINTY, band=BAND_EXPLORATORY, score=0.9)]))
    # 3 cheap: nothing triggers escalation
    for i in range(3):
        ledgers.append(_led(f"cheap_{i}", [_axis(AXIS_STRUCTURAL, band=BAND_UNRESOLVED,
                                                 q=0.5)]))
    return LedgerBundle(run_id="r", target_id="t", ledgers=ledgers, n_candidates=len(ledgers))


def test_expensive_is_a_small_bounded_subset():
    bundle = _bundle()
    budget = TierBudget()
    plan = allocate_tiers(bundle, budget=budget)

    n = len(bundle.ledgers)
    assert plan.count(TIER_EXPENSIVE) <= budget.tier3_reaction_core_max
    # exactly the two strong-band hypotheses reach the reaction core
    assert plan.count(TIER_EXPENSIVE) == 2
    exp_ids = set(plan.by_tier[TIER_EXPENSIVE])
    assert exp_ids == {"strong_0", "strong_1"}
    # the whole universe is NOT escalated to an expensive tier
    assert plan.count(TIER_EXPENSIVE) < n
    assert plan.count(TIER_CHEAP) >= 3
    # every candidate is assigned exactly once and covered
    assert len(plan.assignments) == n
    assert sum(plan.count(t) for t in
               (TIER_CHEAP, TIER_GPU_BROAD, TIER_FOCUSED, TIER_EXPENSIVE)) == n


def test_tier_membership_matches_policy():
    plan = allocate_tiers(_bundle())
    tier_of = {a.variant_id: a.tier for a in plan.assignments}

    for i in range(2):
        assert tier_of[f"strong_{i}"] == TIER_EXPENSIVE
    for i in range(3):
        assert tier_of[f"sig_{i}"] == TIER_FOCUSED
    for i in range(2):
        assert tier_of[f"cons_{i}"] == TIER_FOCUSED
    for i in range(3):
        assert tier_of[f"control_{i}"] == TIER_GPU_BROAD
    for i in range(3):
        assert tier_of[f"lowq_{i}"] == TIER_GPU_BROAD
    for i in range(2):
        assert tier_of[f"deconv_{i}"] == TIER_GPU_BROAD
    for i in range(2):
        assert tier_of[f"learn_{i}"] == TIER_GPU_BROAD
    for i in range(3):
        assert tier_of[f"cheap_{i}"] == TIER_CHEAP


def test_every_non_cheap_assignment_has_a_reason():
    plan = allocate_tiers(_bundle())
    for a in plan.assignments:
        if a.tier != TIER_CHEAP:
            assert a.reason.strip(), f"{a.variant_id} missing reason"
        # cheap ones still carry a (baseline) reason but must never claim a GPU
        if a.tier == TIER_CHEAP:
            assert a.gpu_hint is None


def test_gpu_hint_round_robins_over_pool():
    pool = (0, 1, 2)
    budget = TierBudget(gpu_pool=pool)
    plan = allocate_tiers(_bundle(), budget=budget)

    hints = [a.gpu_hint for a in plan.assignments if a.tier != TIER_CHEAP]
    assert len(hints) == 17  # 2 + 3 + 3 + 2 + 3 + 2 + 2
    assert all(h is not None for h in hints)
    expected = [pool[i % len(pool)] for i in range(len(hints))]
    assert hints == expected
    # each GPU is used a near-equal number of times (no single-GPU pileup)
    assert set(hints) == {0, 1, 2}


def test_empty_pool_gives_no_gpu_hints():
    plan = allocate_tiers(_bundle(), budget=TierBudget(gpu_pool=()))
    assert all(a.gpu_hint is None for a in plan.assignments)


def test_deterministic():
    bundle = _bundle()
    budget = TierBudget(gpu_pool=(0, 1, 2))
    plan_a = allocate_tiers(bundle, budget=budget)
    plan_b = allocate_tiers(bundle, budget=budget)
    assert plan_a.model_dump() == plan_b.model_dump()


def test_caps_bound_the_expensive_tier():
    # 30 strong-band candidates, tiny caps -> reaction core is capped, not all-escalated
    ledgers = [
        _led(f"s_{i}", [_axis(AXIS_MECHANISM, band=BAND_STRONG, q=0.001 + i * 1e-4,
                              score=0.9)])
        for i in range(30)
    ]
    bundle = LedgerBundle(ledgers=ledgers, n_candidates=len(ledgers))
    budget = TierBudget(tier3_reaction_core_max=5, tier2_focused_md_max=8)
    plan = allocate_tiers(bundle, budget=budget)
    assert plan.count(TIER_EXPENSIVE) == 5
    assert plan.count(TIER_FOCUSED) == 3   # 8 focused - 5 promoted to expensive
    assert plan.count(TIER_EXPENSIVE) < len(ledgers)
    # cap-hit is recorded for provenance
    assert any("tier3_reaction_core capped" in note for note in plan.budget_notes)


def test_expensive_ranked_by_lowest_q():
    # lowest-q strong candidates win the scarce reaction-core slots
    ledgers = [
        _led(f"s_{i}", [_axis(AXIS_MECHANISM, band=BAND_STRONG, q=0.10 - i * 0.01,
                              score=0.9)])
        for i in range(5)
    ]
    bundle = LedgerBundle(ledgers=ledgers, n_candidates=len(ledgers))
    plan = allocate_tiers(bundle, budget=TierBudget(tier3_reaction_core_max=2))
    exp = set(plan.by_tier[TIER_EXPENSIVE])
    assert exp == {"s_3", "s_4"}  # smallest q (0.07, 0.06)


def test_non_converged_pmf_gate_blocks_unconverged():
    ids = ["a", "b", "c", "d"]
    converged = {"a": True, "b": False, "c": True}   # "d" absent
    blocked = non_converged_pmf_gate(ids, converged)
    assert blocked == ["b", "d"]
    # a fully-converged set blocks nothing
    assert non_converged_pmf_gate(ids, {k: True for k in ids}) == []


def test_assignment_rejects_bad_tier():
    import pytest
    with pytest.raises(Exception):
        TierAssignment(variant_id="x", tier="not_a_tier")


def test_tierplan_count_of_absent_tier_is_zero():
    plan = TierPlan()
    assert plan.count(TIER_EXPENSIVE) == 0
