"""Tests for the V7 portfolio report (portfolio.report).

The claim-safety gate is the point of this suite: a clean panel renders under
``strict=True`` without raising, and an injected over-claim makes the same strict render
RAISE (ClaimGuard bites). We also check Gate 5 (subset-level vs full-population separation)
and Gate 6 (HTML + JSON + CSV artifacts).

Per the module seam, ``builder.Portfolio`` is a sibling that may not be importable, so we
build a stand-in that matches the SHARED PortfolioVariant/Portfolio contract exactly
(duck-typed by the report). The ``LedgerBundle`` is constructed directly from the frozen
ledger contract — bands/subset_level are set by hand, no sibling module is imported.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel

from evoliez.portfolio.ledger import (
    AXIS_EVOLUTIONARY, AXIS_LIGAND, AXIS_MECHANISM, AXIS_STRUCTURAL, BAND_CONSENSUS,
    BAND_CONTROL, BAND_SIGNIFICANT, BAND_STRONG, CLAIM_L0_HYPOTHESIS, LANE_CONSENSUS,
    LANE_CONTROL, LANE_STRONG, LANE_WT, AxisEvidenceV7, EvidenceLedgerV7, LedgerBundle,
)
from evoliez.portfolio.report import render_portfolio_html, write_portfolio_report


# --- stand-ins for the sibling builder.Portfolio (shared contract, duck-typed) --------
class _Variant(BaseModel):
    model_config = {"extra": "forbid"}
    variant_id: str
    mutation: str = ""
    lane: str
    overall_band: str
    reason_to_test: str = ""
    is_control: bool = False
    control_role: Optional[str] = None
    deconvolution_of: Optional[str] = None
    significant_axes: List[str] = []
    tier_reached: str = "tier0_cheap"


class _Portfolio(BaseModel):
    model_config = {"extra": "forbid"}
    panel_size: int
    target_id: str = ""
    mechanism_class: str = ""
    variants: List[_Variant] = []
    lane_counts: Dict[str, int] = {}
    notes: List[str] = []
    claim_ceiling: str = CLAIM_L0_HYPOTHESIS

    def as_rows(self) -> List[dict]:
        return [v.model_dump() for v in self.variants]


def _axis(name: str, band: str, *, subset: bool = False, q: float = 0.01) -> AxisEvidenceV7:
    return AxisEvidenceV7(axis=name, score=0.8, q_value=q, band=band, subset_level=subset)


def _make_bundle() -> LedgerBundle:
    led_strong = EvidenceLedgerV7(
        variant_id="v_strong", mutation="A10G", lane=LANE_STRONG, overall_band=BAND_STRONG,
        axes={AXIS_STRUCTURAL: _axis(AXIS_STRUCTURAL, BAND_STRONG),
              AXIS_EVOLUTIONARY: _axis(AXIS_EVOLUTIONARY, BAND_SIGNIFICANT)})
    led_consensus = EvidenceLedgerV7(
        variant_id="v_cons", mutation="B20K;C30F", lane=LANE_CONSENSUS,
        overall_band=BAND_CONSENSUS, consensus_axes=[AXIS_STRUCTURAL, AXIS_MECHANISM],
        axes={AXIS_STRUCTURAL: _axis(AXIS_STRUCTURAL, BAND_SIGNIFICANT),
              AXIS_MECHANISM: _axis(AXIS_MECHANISM, BAND_SIGNIFICANT),
              # subset-level q on the ligand axis -> must trigger the Gate-5 note
              AXIS_LIGAND: _axis(AXIS_LIGAND, BAND_SIGNIFICANT, subset=True)})
    led_control = EvidenceLedgerV7(
        variant_id="v_ctrl", mutation="D40A", lane=LANE_CONTROL, overall_band=BAND_CONTROL,
        is_control=True, control_role="scalar_top")
    led_wt = EvidenceLedgerV7(
        variant_id="v_wt", mutation="", lane=LANE_WT, overall_band=BAND_CONTROL,
        is_control=True, control_role="wt_parental")
    return LedgerBundle(
        run_id="run_test", target_id="enzymeX", mechanism_class="oxidoreductase",
        ledgers=[led_strong, led_consensus, led_control, led_wt], n_candidates=4,
        axis_null_sizes={AXIS_STRUCTURAL: 200, AXIS_MECHANISM: 200},
        notes=["null models built from 200 background variants"])


def _make_portfolio() -> _Portfolio:
    return _Portfolio(
        panel_size=4, target_id="enzymeX", mechanism_class="oxidoreductase",
        lane_counts={LANE_STRONG: 1, LANE_CONSENSUS: 1, LANE_CONTROL: 1, LANE_WT: 1},
        notes=["Panel prioritized for experimental testing."],
        variants=[
            _Variant(variant_id="v_strong", mutation="A10G", lane=LANE_STRONG,
                     overall_band=BAND_STRONG, reason_to_test="prioritized for experimental testing",
                     significant_axes=[AXIS_STRUCTURAL, AXIS_EVOLUTIONARY], tier_reached="tier1_gpu_broad"),
            _Variant(variant_id="v_cons", mutation="B20K;C30F", lane=LANE_CONSENSUS,
                     overall_band=BAND_CONSENSUS, reason_to_test="cross-axis consensus evidence band",
                     significant_axes=[AXIS_STRUCTURAL, AXIS_MECHANISM], deconvolution_of=None),
            _Variant(variant_id="v_ctrl", mutation="D40A", lane=LANE_CONTROL,
                     overall_band=BAND_CONTROL, reason_to_test="scalar-top control probe",
                     is_control=True, control_role="scalar_top"),
            _Variant(variant_id="v_wt", mutation="", lane=LANE_WT, overall_band=BAND_CONTROL,
                     reason_to_test="wild-type parental control", is_control=True,
                     control_role="wt_parental"),
        ])


# --- Gate 4: the claim-safety gate ---------------------------------------------------
def test_clean_report_passes_strict():
    """A clean panel renders under strict=True without raising, and reads hypothesis-grade."""
    html = render_portfolio_html(_make_portfolio(), _make_bundle(), strict=True, run_id="run_test")
    assert "<title>" in html and "portfolio" in html.lower()
    assert "hypothesis-grade" in html.lower()
    assert "prioritized for experimental testing" in html.lower()
    # panel content is present
    assert "v_strong" in html and "v_cons" in html
    assert "oxidoreductase" in html


def test_injected_overclaim_raises_under_strict():
    """Inject an over-claim into a variant's reason-to-test; strict rendering must RAISE."""
    portfolio = _make_portfolio()
    bundle = _make_bundle()
    # sanity: clean before injection
    render_portfolio_html(portfolio, bundle, strict=True)
    portfolio.variants[0].reason_to_test = (
        "This variant is catalytically superior and improves activity")
    with pytest.raises(AssertionError):
        render_portfolio_html(portfolio, bundle, strict=True)


def test_default_mode_does_not_raise_on_overclaim():
    """Without strict, an over-claim degrades to a visible banner, never breaks rendering."""
    portfolio = _make_portfolio()
    portfolio.variants[0].reason_to_test = "improves activity dramatically"
    html = render_portfolio_html(portfolio, _make_bundle())  # strict defaults to None/off
    assert "ClaimGuard" in html  # warn banner prepended


# --- Gate 5: subset-level vs full-population separation -------------------------------
def test_subset_level_note_rendered():
    html = render_portfolio_html(_make_portfolio(), _make_bundle(), strict=True)
    assert "subset-level" in html.lower()
    # the flagged axis (ligand) is named in the note
    assert "ligand" in html.lower()


def test_full_population_note_when_no_subset():
    bundle = _make_bundle()
    for led in bundle.ledgers:
        for ax in led.axes.values():
            ax.subset_level = False
    html = render_portfolio_html(_make_portfolio(), bundle, strict=True)
    assert "full-population" in html.lower()


# --- Gate 6: reproducible artifacts (HTML + JSON + CSV) ------------------------------
def test_write_portfolio_report_artifacts(tmp_path: Path):
    portfolio = _make_portfolio()
    bundle = _make_bundle()
    paths = write_portfolio_report(portfolio, bundle, tmp_path, run_id="run_test")

    assert set(paths) == {"html", "json", "csv"}
    for key in ("html", "json", "csv"):
        p = Path(paths[key])
        assert p.exists() and p.stat().st_size > 0
        assert str(p).startswith(str(tmp_path))

    # HTML is a real page
    assert Path(paths["html"]).read_text(encoding="utf-8").startswith("<!doctype html>")

    # JSON parses and carries both the panel dump and a bundle summary
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert "portfolio" in data and "bundle_summary" in data
    assert data["bundle_summary"]["run_id"] == "run_test"
    assert data["bundle_summary"]["per_axis_band_distribution"][AXIS_STRUCTURAL][BAND_STRONG] == 1
    assert data["bundle_summary"]["subset_level_counts"][AXIS_LIGAND] == 1

    # CSV parses with a stable header + one row per variant
    with open(paths["csv"], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(portfolio.variants)
    assert rows[0]["variant_id"] == "v_strong"
    # list-valued field is serialized (not dropped)
    assert AXIS_STRUCTURAL in rows[0]["significant_axes"]
    assert rows[2]["is_control"] == "true"


def test_write_creates_missing_out_dir(tmp_path: Path):
    nested = tmp_path / "reports" / "v7"
    paths = write_portfolio_report(_make_portfolio(), _make_bundle(), nested, run_id="r")
    assert (nested / "v7_portfolio.html").exists()
    assert Path(paths["csv"]).parent == nested
