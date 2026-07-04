"""V5-5 claim-integrity guards (ROADMAP_V5 §5):

1. recommendation() gates "strong candidate" on real reaction geometry — a top, confident but
   catalytically-dead (geometry 0) or unmeasured (geometry None) candidate is downgraded to a
   structural/binding recommendation. This is the CAR-class overclaim ("strong candidate" printed
   right above "catalytic geometry score 0.00") killed at the source.
2. is_paper_grade() requires a non-REJECTED verdict, so a structurally-rejected but
   wt_anchored + reference_like record is no longer double-counted as BOTH paper_grade and
   classes['rejected'] (the confusing "paper-grade:N ∧ rejected:N" dual state).
"""
import pytest

from evoliez.md.gate_stack import CONFIRMED_COMPUTATIONAL, REJECTED
from evoliez.ml.calibration import recommendation
from evoliez.ranking.claim_guard import lint_text
from evoliez.ranking.evidence import build_evidence_library, is_paper_grade


# --- 1. geometry-gated recommendation -------------------------------------------------

def test_strong_requires_positive_geometry():
    # top rank + confident, but geometry is 0 (computed, dead) -> NOT strong
    assert recommendation(5.0, 0.1, 1, 100, geometry=0.0) == "structural/binding candidate"
    # geometry not computed (None) -> also NOT strong (fail-safe: unmeasured != positive)
    assert recommendation(5.0, 0.1, 1, 100, geometry=None) == "structural/binding candidate"
    # default (no geometry passed) is the fail-safe unmeasured path
    assert recommendation(5.0, 0.1, 1, 100) == "structural/binding candidate"
    # real positive reaction geometry -> a claim-safe priority label (NOT bare "strong candidate")
    assert recommendation(5.0, 0.1, 1, 100, geometry=0.25) == "priority screening candidate"


def test_geometry_gate_does_not_touch_reject_or_uncertain():
    # high uncertainty still rejects regardless of geometry
    assert recommendation(0.0, 0.9, 50, 100, geometry=1.0) == "reject"
    # mid candidate stays uncertain regardless of geometry
    assert recommendation(1.0, 0.5, 40, 100, geometry=1.0) == "uncertain candidate"


def test_car_case_geometry_zero_not_strong():
    # CAR reality: a top mutant with ΔNAC == 0 (dynamic NAC invalid w/o Mg2+) is structural/binding
    rec = recommendation(4.2, 0.09, 1, 40, geometry=0.0)
    assert rec == "structural/binding candidate"
    assert rec != "strong candidate"


# --- 3. ClaimGuard forbids catalysis-implying label words (V5-5 patterns) --------------

def test_bare_strong_candidate_forbidden_qualified_allowed():
    assert lint_text("This is a strong candidate for testing.")        # bare -> violation
    # the qualifier sits between the words, so the bare-form anchor does not match
    assert not lint_text("This is a strong structural candidate.")
    assert not lint_text("This is a strong binding candidate.")


def test_catalytic_lead_and_paper_grade_forbidden():
    assert lint_text("G430R is the catalytic lead.")                   # -> violation
    assert lint_text("This candidate is paper-grade.")                 # -> violation
    assert lint_text("The mutant is confirmed productive.")            # -> violation
    # the claim-safe replacements the report copy now uses must pass
    assert not lint_text("G430R is the top-ranked lead.")
    assert not lint_text("This candidate is anchored-evaluated.")


def test_new_patterns_are_negation_safe():
    # a negated statement about the forbidden phrase is allowed
    assert not lint_text("This is not a strong candidate.")


# --- 4. CAR overclaim on the provenance path (reviewer requirement C) ------------------

def test_car_overclaim_phrases_blocked_at_floor_provenance():
    from evoliez.ranking.claim_guard import assert_report_clean, lint_report
    # floor (no wet-lab) provenance: none of the CAR-run problem phrases may appear in a report
    for phrase in ("G430R;S433F;G407K is the catalytic lead",
                   "the mutant is confirmed productive",
                   "this candidate is paper-grade",
                   "a strong candidate for testing",
                   "activity improved over the wild type",
                   "a stable functional complex"):
        assert lint_report(phrase, None), f"should be blocked at floor provenance: {phrase!r}"
        with pytest.raises(AssertionError):
            assert_report_clean(phrase, None)


def test_car_allowed_replacements_pass():
    from evoliez.ranking.claim_guard import assert_report_clean, lint_report
    for phrase in ("G430R;S433F;G407K is a hypothesis-grade lead",
                   "a screening-level catalytic-geometry hypothesis",
                   "a structurally viable candidate",
                   "the candidate is anchored-evaluated",
                   "prioritized for experimental testing"):
        assert not lint_report(phrase, None), f"should pass under floor provenance: {phrase!r}"
        assert_report_clean(phrase, None)      # must not raise


# --- 2. paper-grade requires a non-rejected verdict -----------------------------------

def _rec(verdict, *, anchored=True, reference_like=True, has_gate_stack=True):
    r = {}
    if anchored:
        r["validation_structure"] = "wt_anchored"
    if reference_like:
        r["pose_gate"] = {"design_ligand": {"status": "reference_like"}}
    if has_gate_stack:
        r["gate_stack"] = {"verdict": verdict}
    return r


def test_rejected_anchored_record_is_not_paper_grade():
    # wt_anchored + reference_like + gate_stack present, but verdict == REJECTED
    rec = _rec(REJECTED)
    assert is_paper_grade(rec) is False


def test_confirmed_anchored_record_is_paper_grade():
    rec = _rec(CONFIRMED_COMPUTATIONAL)
    assert is_paper_grade(rec) is True


def test_non_anchored_never_paper_grade():
    rec = _rec(CONFIRMED_COMPUTATIONAL, anchored=False)
    assert is_paper_grade(rec) is False


def test_paper_grade_and_rejected_no_longer_double_count():
    # A rejected+anchored+reference_like record must not appear in BOTH partitions.
    recs = [_rec(REJECTED), _rec(CONFIRMED_COMPUTATIONAL)]
    lib = build_evidence_library(recs)
    # exactly the confirmed record is paper-grade; the rejected one is not
    assert len(lib.paper_grade) == 1
    # and the rejected record is still counted in the rejected class (not lost) — so the two
    # numbers are now a clean partition, not an overlapping "paper-grade:N ∧ rejected:N" pair
    assert lib.counts.get(REJECTED, 0) == 1
