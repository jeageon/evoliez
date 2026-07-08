"""ClaimGuard linter tests (ROADMAP_V3 D3 / V3-1b acceptance table)."""
import pytest

from evoliez.ranking import claim_guard as cg
from evoliez.ranking.terminology import display_name, normalize_key


# --- the acceptance table from the roadmap -------------------------------------------
def test_positive_prohibited():
    v = cg.lint_text("The variant activity improved over wild type.")
    assert any(x.category == cg.ACTIVITY_IMPROVEMENT for x in v)


def test_negated_safe_english():
    # "does not predict kcat" must be ALLOWED
    assert cg.lint_text("This workflow does not predict kcat or kcat/KM.") == []


def test_roadmap_v6_section13_all_forbidden_caught():
    """Gate 4 (no overclaim possible): EVERY phrase in ROADMAP_V6 §13's forbidden
    list must be rejected with no wet-lab evidence. Regression guard for a real hole
    found in live V6 verification (only 2 of 6 were caught)."""
    from evoliez.ranking.claim_guard import ClaimProvenance
    prov = ClaimProvenance()  # no wet-lab -> nothing unlocked
    forbidden = [
        "activity improved", "kcat improved", "catalytically validated",
        "validated lead", "reduced activation barrier", "experimentally active",
    ]
    for phrase in forbidden:
        with pytest.raises(AssertionError):
            cg.assert_report_clean(f"The lead shows {phrase} in the assay.", prov)


def test_roadmap_v6_section13_allowed_language_passes():
    """The roadmap's *allowed* language must NOT be flagged (no false positives)."""
    from evoliez.ranking.claim_guard import ClaimProvenance
    prov = ClaimProvenance()
    for phrase in ["screening-level reaction-geometry evidence", "near-attack access cost",
                   "mechanism-probe candidate panel", "hypothesis-grade candidate",
                   "higher-cost validation recommended"]:
        cg.assert_report_clean(f"This is {phrase}.", prov)  # must not raise


def test_v6_wetlab_unlocks_validation_and_barrier():
    """The new categories unlock ONLY with replicated wet-lab evidence."""
    from evoliez.ranking.claim_guard import ClaimProvenance, evaluate, lint_text
    allow = evaluate(ClaimProvenance(wetlab_replicated=True)).allow()
    assert cg.CATALYTIC_VALIDATION in allow and cg.ACTIVATION_BARRIER in allow
    assert lint_text("A validated lead with reduced activation barrier.", allow=allow) == []
    # without wet-lab, both are forbidden again
    assert lint_text("A validated lead with reduced activation barrier.") != []


def test_synonym_without_banned_token():
    # over-claim with no 'kcat' token
    v = cg.lint_text("The variant is catalytically superior to the wild type.")
    assert any(x.category == cg.ACTIVITY_IMPROVEMENT for x in v)


def test_korean_overclaim_caught():
    v = cg.lint_text("이 변이는 활성 증가를 보인다.")
    assert any(x.category == cg.ACTIVITY_IMPROVEMENT for x in v)
    v2 = cg.lint_text("촉매 효율 향상이 예상된다.")
    assert v2


def test_korean_negation_safe():
    # "활성을 직접 예측하지 않는다" — Korean post-verbal negation
    assert cg.lint_text("이 워크플로는 활성 향상을 직접 예측하지 않는다.") == []


def test_latex_kcat_caught():
    v = cg.lint_text(r"The mutant shows $k_{cat}/K_M$ improved binding.")
    assert any(x.category == cg.KINETIC_PARAMETER_PREDICTION for x in v)


def test_short_md_and_long_term_overclaim():
    assert cg.lint_text("Short MD confirms a functional catalytic state.")
    assert cg.lint_text("The complex shows long-term ligand stability.")
    assert cg.lint_text("This demonstrates stable functional complex formation.")


def test_inactive_classification_caught():
    assert cg.lint_text("This is an inactive mutant.")


def test_allowed_phrases_pass():
    safe = (
        "This variant is prioritized for experimental testing. "
        "It carries screening-level evidence for structural and ligand/cofactor "
        "competence. Short MD supports local reference-pose accommodation. "
        "The workflow does not directly predict kcat or kcat/KM."
    )
    assert cg.lint_text(safe) == []


def test_allow_unlocks_category():
    txt = "With replicated assays, activity improved over the stability-only baseline."
    assert cg.lint_text(txt)  # prohibited by default
    assert cg.lint_text(txt, allow=[cg.ACTIVITY_IMPROVEMENT]) == []  # unlocked


def test_assert_clean_raises():
    with pytest.raises(AssertionError):
        cg.assert_clean("kcat improved markedly.")
    cg.assert_clean("Prioritized for wet-lab validation.")  # no raise


# --- terminology map -----------------------------------------------------------------
def test_terminology_aliases():
    assert normalize_key("reference_like") == "reference_pose_accommodation"
    assert normalize_key("nac_score") == "reaction_geometry_accommodation"
    assert normalize_key("ml_score") == "ml_evidence_prior"
    # a new key normalizes to itself; unknown passes through
    assert normalize_key("reference_pose_accommodation") == "reference_pose_accommodation"
    assert normalize_key("some_unrelated_key") == "some_unrelated_key"
    assert display_name("nac") == "reaction-geometry accommodation"
