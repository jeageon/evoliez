"""Regression tests for the report-level ClaimGuard surface (Fable safety review, 2026-07-09).

The linter only ever saw stripped-tag body TEXT, so two visible surfaces were unguarded: the
page() <title> and HTML attribute values (title=/alt=/aria-label=/data-, CSS content:). Both now
route through the guard.
"""
from __future__ import annotations

import pytest

from evoliez.io._report_kit import page


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except AssertionError:
        return True


def test_title_overclaim_is_linted_under_strict():
    assert _raises(lambda: page("validated lead that improves activity",
                                "<p>clean body.</p>", strict=True))


@pytest.mark.parametrize("body", [
    '<p title="this variant is catalytically superior to WT">tooltip</p>',
    '<img alt="mutant improves activity over WT">',
    '<div aria-label="kcat improved for the mutant">x</div>',
    '<span data-note="activity improved over WT">y</span>',
    '<style>.x::after{content:"the variant is more active than WT"}</style><p>ok</p>',
])
def test_attribute_and_css_overclaims_are_linted(body):
    assert _raises(lambda: page("clean title", body, strict=True))


def test_clean_report_with_attributes_passes_strict():
    body = ('<p title="prioritized for experimental testing">'
            'This variant is prioritized for experimental testing.</p>')
    # must NOT raise
    page("EvoLiEZ portfolio", body, strict=True)


def test_claim_allow_still_works_for_doc_examples_but_is_audited(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="evoliez.claim_guard"):
        out = page("t", "<p>activity improved</p>", strict=True,
                   claim_allow=["activity_improvement"])
    assert "unguarded claim" not in out                     # opt-out honored
    assert any("claim_allow" in r.message for r in caplog.records)  # but logged loudly


@pytest.mark.parametrize("body", [
    "<span title=paper-grade>x</span>",                       # unquoted single-token value
    '<meta content="activity improved over WT">',            # meta content
    '<input value="validated lead">',                        # form field value
    '<button aria-description="the mutant is more active than WT">x</button>',
])
def test_unquoted_and_extra_attribute_surfaces_are_linted(body):
    assert _raises(lambda: page("clean", body, strict=True))
