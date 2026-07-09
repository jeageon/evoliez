"""Regression tests for the ClaimGuard hardening (Fable safety review, 2026-07-09).

Each test pins a fail-open path the review demonstrated with a probe: an over-claim that USED
to slip past the linter must now raise a violation, and every legitimate negation must still be
honored (no new false positives). See the review comments for the failure scenarios.
"""
from __future__ import annotations

import pytest

from evoliez.ranking.claim_guard import (
    ACTIVITY_IMPROVEMENT, KINETIC_PARAMETER_PREDICTION, ClaimProvenance, evaluate, lint_text,
)


# --- BLOCKING: clause-scoped English negation -----------------------------------------
@pytest.mark.parametrize("text", [
    "The fold is not compromised, activity improved over WT.",
    "The variant is not unstable but activity improved.",
    "There is no direct assay, yet the mutant is more active than WT.",
])
def test_earlier_clause_negation_does_not_launder_a_later_overclaim(text):
    assert lint_text(text), f"over-claim slipped past the linter: {text!r}"


@pytest.mark.parametrize("text", [
    "This workflow does not predict kcat or kcat/KM.",
    "The mutant is not more active than WT.",
])
def test_same_clause_negation_still_honored(text):
    assert lint_text(text) == []


# --- MAJOR: markdown emphasis cannot hide a claim -------------------------------------
@pytest.mark.parametrize("text", [
    "The **activity** **improved** dramatically.",
    "activity **improved** over WT",
    "the mutant is `catalytically superior` to WT",
    "~~unrelated~~ activity ~~improved~~",
])
def test_markdown_emphasis_inside_a_claim_is_still_caught(text):
    assert lint_text(text)


# --- MAJOR: Korean negation scoping (없/아니 are not claim-negations) -------------------
@pytest.mark.parametrize("text", [
    "부작용이 없이 활성이 증가했다.",                 # "activity increased without side effects"
    "이것은 문제가 아니라 촉매 활성이 향상되었다.",     # "not a problem — catalytic activity improved"
])
def test_incidental_korean_negation_does_not_suppress(text):
    assert lint_text(text), f"KO over-claim laundered by an incidental 없/아니: {text!r}"


@pytest.mark.parametrize("text", [
    "이 워크플로는 활성 향상을 직접 예측하지 않는다",   # "does not predict activity improvement"
    "촉매 활성이 향상되지 않았다.",                   # "catalytic activity did not improve"
])
def test_real_korean_predicate_negation_still_honored(text):
    assert lint_text(text) == []


def test_korean_negation_scoped_to_clause():
    # first clause is a real over-claim; the negation is only on the SECOND (toxicity) clause.
    assert lint_text("활성이 증가했고, 독성은 늘지 않았다.")


# --- MINOR: LaTeX kcat wrappers -------------------------------------------------------
@pytest.mark.parametrize("text", [
    r"predicts $k_{\text{cat}}$ improvement",
    r"higher $k_\mathrm{cat}$ for the mutant",
    r"improved $k_{cat}$",
])
def test_latex_kcat_wrappers_are_caught(text):
    viols = lint_text(text)
    assert any(v.category == KINETIC_PARAMETER_PREDICTION for v in viols)


# --- MAJOR: fuzzy-truthy provenance must not unlock the strongest categories ----------
@pytest.mark.parametrize("val", ["yes", "1", "true", "on", "y", 1, 1.0])
def test_fuzzy_truthy_wetlab_flag_fails_safe_to_floor(val):
    verdict = evaluate({"wetlab_replicated": val})
    assert verdict.fail_safe is True
    assert ACTIVITY_IMPROVEMENT not in verdict.allowed_categories
    assert KINETIC_PARAMETER_PREDICTION not in verdict.allowed_categories


def test_real_bool_wetlab_flag_still_unlocks():
    verdict = evaluate(ClaimProvenance(wetlab_replicated=True))
    assert ACTIVITY_IMPROVEMENT in verdict.allowed_categories


def test_strict_bool_construction_rejects_string():
    with pytest.raises(Exception):
        ClaimProvenance(wetlab_replicated="yes")


# --- Fable verification residuals: KO predicate vs connective negation ----------------
@pytest.mark.parametrize("text", [
    "활성 향상 효과는 없다.",                 # "there is NO enhancement effect" (predicate 없다)
    "이것은 활성 증가가 아니라 안정성 향상이다.",   # "not an activity increase, but ..." (아니라 negates claim)
])
def test_korean_predicate_negation_is_honored(text):
    assert lint_text(text) == []


@pytest.mark.parametrize("text", [
    "부작용이 없이 활성이 증가했다.",             # 없이 "without" does not negate the claim
    "이것은 문제가 아니라 촉매 활성이 향상되었다.",   # 아니라 negates 문제, the later claim is asserted
])
def test_korean_connective_negation_does_not_launder(text):
    assert lint_text(text)
