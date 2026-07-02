"""V3-4: full ClaimGuard engine — provenance-driven, fail-safe (ROADMAP_V3 D3)."""
import pytest

from evoliez.ranking import claim_guard as cg
from evoliez.mechanism import vocab


def test_no_wetlab_tier_c_short_md_blocks_all():
    prov = cg.ClaimProvenance(
        reference_claim_strength=vocab.MODERATE_SCREENING,
        geometry_claim_ceiling=vocab.MODERATE_SCREENING,
        only_short_md=True, wetlab_replicated=False,
        known_active_controls=False, known_inactive_controls=False,
        wt_reaction_geometry_sparse=True)
    v = cg.evaluate(prov)
    assert v.allowed_categories == set()                 # nothing unlocked
    assert v.claim_ceiling == vocab.UNCALIBRATED          # no controls -> uncalibrated
    assert "uncalibrated" in v.flags
    assert "diagnostic_only_geometry" in v.flags
    # a report that over-claims under this provenance fails
    with pytest.raises(AssertionError):
        cg.assert_report_clean("Activity improved and kcat improved.", prov)


def test_wetlab_replicated_unlocks_activity():
    prov = cg.ClaimProvenance(
        reference_claim_strength=vocab.STRONG_SCREENING,
        geometry_claim_ceiling=vocab.STRONG_SCREENING,
        wetlab_replicated=True, known_active_controls=True)
    v = cg.evaluate(prov)
    assert cg.ACTIVITY_IMPROVEMENT in v.allowed_categories
    # now an enrichment claim passes the report lint
    cg.assert_report_clean(
        "With replicated assays, activity improved over the stability-only baseline.", prov)


def test_short_md_overclaim_locked_until_enhanced_sampling():
    short = cg.ClaimProvenance(only_short_md=True, known_active_controls=True)
    assert cg.SHORT_MD_OVERCLAIM not in cg.evaluate(short).allowed_categories
    enhanced = cg.ClaimProvenance(only_short_md=False, enhanced_sampling_or_qmmm=True,
                                  known_active_controls=True)
    assert cg.SHORT_MD_OVERCLAIM in cg.evaluate(enhanced).allowed_categories


def test_high_pose_disagreement_forbids_inactive_even_with_wetlab():
    prov = cg.ClaimProvenance(wetlab_replicated=True, known_active_controls=True,
                              de_novo_pose_disagreement_high=True)
    v = cg.evaluate(prov)
    assert cg.INACTIVE_CLASSIFICATION not in v.allowed_categories
    assert "pose_uncertainty_high" in v.flags


def test_fail_safe_on_malformed_provenance():
    # a typo'd key would raise in the strict model -> evaluate must fail SAFE, not crash
    v = cg.evaluate({"reference_claim_stength": "strong_screening"})  # typo
    assert v.fail_safe is True
    assert v.allowed_categories == set()
    assert v.claim_ceiling == vocab.UNCALIBRATED


def test_fail_safe_on_none():
    v = cg.evaluate(None)
    assert v.fail_safe is True and v.claim_ceiling == vocab.UNCALIBRATED


def test_ceiling_is_weakest_contributor():
    prov = cg.ClaimProvenance(
        reference_claim_strength=vocab.STRONG_SCREENING,
        geometry_claim_ceiling=vocab.HYPOTHESIS_GRADE,   # weakest contributor wins
        known_active_controls=True)
    assert cg.evaluate(prov).claim_ceiling == vocab.HYPOTHESIS_GRADE


def test_allowed_templates_track_unlock():
    locked = cg.evaluate(cg.ClaimProvenance(known_active_controls=True))
    unlocked = cg.evaluate(cg.ClaimProvenance(wetlab_replicated=True,
                                              known_active_controls=True))
    assert "does not directly predict kcat" in \
        " ".join(cg.allowed_templates(cg.KINETIC_PARAMETER_PREDICTION, locked))
    assert cg.allowed_templates(cg.ACTIVITY_IMPROVEMENT, unlocked)  # non-empty when unlocked
