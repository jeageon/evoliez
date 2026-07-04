"""ROADMAP_V5 (reviewer 1.4) — no report bypasses ClaimGuard. The 8 structural reports that
render their own <!doctype> now route their HTML through kit.assert_html_clean, so an injected
over-claim fails under EVOLIEZ_STRICT_CLAIMS everywhere, not just in page()-based reports."""
import importlib
import inspect

import pytest

from evoliez.io._report_kit import assert_html_clean

_HTML = "<!doctype html><html><body><p>{}</p></body></html>"

# the 8 own-template writers that previously bypassed the guard
STRUCTURAL_REPORTS = [
    "complex_report", "docking_report", "homolog_report", "input_report",
    "interaction_model_report", "msa_report", "provenance_report", "s07_report",
]


def test_assert_html_clean_blocks_overclaim_in_strict():
    with pytest.raises(AssertionError):
        assert_html_clean(_HTML.format("G430R is the catalytic lead"), title="x", strict=True)


def test_assert_html_clean_passes_clean_html_unchanged():
    clean = _HTML.format("a structurally viable candidate prioritized for testing")
    assert assert_html_clean(clean, title="x", strict=True) == clean


def test_assert_html_clean_nonstrict_prepends_banner_inside_body():
    html = _HTML.format("catalytic lead here")
    out = assert_html_clean(html, title="x", strict=False)
    assert out != html and "ClaimGuard" in out
    assert out.index("ClaimGuard") > out.index("<body")   # banner injected inside <body>


@pytest.mark.parametrize("mod_name", STRUCTURAL_REPORTS)
def test_structural_report_routes_through_guard(mod_name):
    mod = importlib.import_module(f"evoliez.io.{mod_name}")
    assert "assert_html_clean" in inspect.getsource(mod), (
        f"{mod_name} must route its HTML through assert_html_clean (no ClaimGuard bypass)")
