"""ROADMAP_V3 B3 regression: ClaimGuard is now enforced structurally in the shared
report wrapper (_report_kit.page). Every HTML report is linted against its claim
provenance; an injected over-claim with no wet-lab evidence must fail under strict
mode, warn+banner by default, and be permitted only when provenance unlocks it.
"""
from __future__ import annotations

import pytest

from evoliez.io._report_kit import page
from evoliez.ranking.claim_guard import ClaimProvenance

OVERCLAIM = "<h1>Result</h1><p>This variant shows activity improved over the wild type.</p>"
KCAT = "<p>The model predicts kcat for every mutant.</p>"
CLEAN = "<h1>Result</h1><p>This variant is prioritized for experimental testing.</p>"


def test_strict_mode_raises_on_overclaim():
    with pytest.raises(AssertionError) as ei:
        page("t", OVERCLAIM, strict=True)
    assert "activity_improvement" in str(ei.value)


def test_strict_mode_raises_on_kcat():
    with pytest.raises(AssertionError):
        page("t", KCAT, strict=True)


def test_default_mode_warns_with_banner_not_raise():
    out = page("t", OVERCLAIM)  # default (non-strict) — must NOT raise
    assert "ClaimGuard" in out and "unguarded claim" in out
    # the offending phrase is still in the doc, now flagged by a banner
    assert "activity improved" in out.lower()


def test_wetlab_provenance_unlocks_activity_claim():
    prov = ClaimProvenance(wetlab_replicated=True)
    out = page("t", OVERCLAIM, strict=True, claim_provenance=prov)  # unlocked -> no raise
    assert "unguarded claim" not in out


def test_clean_report_has_no_banner():
    out = page("t", CLEAN, strict=True)
    assert "unguarded claim" not in out


def test_env_strict_flag_raises(monkeypatch):
    monkeypatch.setenv("EVOLIEZ_STRICT_CLAIMS", "1")
    with pytest.raises(AssertionError):
        page("t", OVERCLAIM)  # strict resolved from env


def test_claim_allow_escape_hatch_for_doc_reports():
    # a report that quotes forbidden phrases as EXAMPLES can opt out per-category
    out = page("t", OVERCLAIM, strict=True, claim_allow=["activity_improvement"])
    assert "unguarded claim" not in out
