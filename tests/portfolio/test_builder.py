"""Tests for the V7 mechanism-ranked portfolio builder (ROADMAP_V7 §8).

The builder consumes a *banded* LedgerBundle. Per the module contract we construct that
bundle directly from the ledger primitives (setting axes bands / q-values / overall_band
ourselves) rather than importing the sibling bands module. ControlSpec is duck-typed by the
builder, so we stand it in with a tiny local shape that mirrors the shared contract.
"""
from __future__ import annotations

from collections import namedtuple

import pytest

from evoliez.portfolio.builder import (
    Portfolio, PortfolioVariant, build_portfolio, default_lane_quota,
)
from evoliez.portfolio.ledger import (
    ALL_LANES, AXIS_LIGAND, AXIS_MECHANISM, AXIS_STRUCTURAL, AXIS_UNCERTAINTY,
    BAND_CONTROL, BAND_EXPLORATORY, BAND_SIGNIFICANT, BAND_STRONG, BAND_CONSENSUS,
    CLAIM_L1_SCREENING, LANE_CONTROL, LANE_DECONVOLUTION, LANE_WT,
    AxisEvidenceV7, EvidenceLedgerV7, LedgerBundle,
)
from evoliez.ranking.claim_guard import assert_clean

# A minimal stand-in for portfolio.controls.ControlSpec (builder only duck-types it).
Ctl = namedtuple("Ctl", "variant_id mutation control_type reason provenance")


def _ax(axis: str, *, band: str, score: float = 0.8, q=None, missing: bool = False):
    return AxisEvidenceV7(axis=axis, score=score, band=band, q_value=q, missing=missing)


def _led(vid, mut, band, *, sig_axis=AXIS_STRUCTURAL, q=None, unc=None, consensus=()):
    axes = {}
    if band in (BAND_STRONG, BAND_SIGNIFICANT):
        axes[sig_axis] = _ax(sig_axis, band=band, q=q)
    if unc is not None:
        axes[AXIS_UNCERTAINTY] = _ax(AXIS_UNCERTAINTY, band=BAND_EXPLORATORY, score=unc,
                                     missing=False)
    return EvidenceLedgerV7(variant_id=vid, mutation=mut, axes=axes, overall_band=band,
                            consensus_axes=list(consensus))


def _bundle():
    ledgers = [
        # strong / significant singles
        _led("v_strong1", "A10G", BAND_STRONG, sig_axis=AXIS_STRUCTURAL, q=0.001),
        _led("v_sig1", "B20C", BAND_SIGNIFICANT, sig_axis=AXIS_LIGAND, q=0.02),
        # a consensus candidate
        _led("v_cons1", "C30E", BAND_CONSENSUS, consensus=[AXIS_STRUCTURAL, AXIS_MECHANISM]),
        # the strongest MULTIPOINT hypothesis (parent) + its component candidates
        _led("v_multi", "P100A;Q200B;R300C", BAND_STRONG, sig_axis=AXIS_MECHANISM, q=0.0005),
        _led("v_p", "P100A", BAND_EXPLORATORY),
        _led("v_q", "Q200B", BAND_EXPLORATORY),
        _led("v_pq", "P100A;Q200B", BAND_EXPLORATORY),
        # a clean single for the single-site lane
        _led("v_single", "S400T", BAND_EXPLORATORY),
        # an unplaced multipoint carrying the highest uncertainty -> uncertainty lane
        _led("v_unc", "M1N;O2P", BAND_EXPLORATORY, unc=0.95),
    ]
    return LedgerBundle(run_id="r1", target_id="t1", mechanism_class="hydride_transfer",
                        ledgers=ledgers, n_candidates=len(ledgers))


def _controls():
    return [
        Ctl("WT", "WT", "wt_parental", "parental", []),
        Ctl("c_scalar", "Q382R", "scalar_top", "top scalar", []),
        Ctl("c_geom", "G291A", "geometry_weak", "weak geometry", []),
        Ctl("c_low", "N260C", "low_signal", "low signal", []),
    ]


def test_default_lane_quota_sums_to_panel_size():
    for size in (24, 32, 48):
        q = default_lane_quota(size)
        assert set(q) == set(ALL_LANES)
        assert sum(q.values()) == size
        assert q[LANE_WT] == 1
    with pytest.raises(ValueError):
        default_lane_quota(50)


def test_build_panel_shape_and_lanes():
    port = build_portfolio(_bundle(), panel_size=48, controls=_controls(), seed=0)
    assert isinstance(port, Portfolio)
    assert len(port.variants) <= 48
    assert len(port.variants) > 0

    # every variant is well-formed: valid lane + non-empty reason_to_test.
    for v in port.variants:
        assert v.lane in ALL_LANES
        assert isinstance(v, PortfolioVariant)
        assert v.reason_to_test.strip(), f"empty reason for {v.variant_id}"

    # WT parental present and first.
    assert port.variants[0].lane == LANE_WT
    assert any(v.lane == LANE_WT and v.mutation == "WT" for v in port.variants)

    # lane_counts sums to the panel size.
    assert sum(port.lane_counts.values()) == len(port.variants)

    # claim ceiling is L1_screening (a strong band is present) — never higher.
    assert port.claim_ceiling == CLAIM_L1_SCREENING


def test_deconvolution_probes_carry_parent():
    port = build_portfolio(_bundle(), panel_size=48, controls=_controls(), seed=0)
    deconv = [v for v in port.variants if v.lane == LANE_DECONVOLUTION]
    assert deconv, "expected deconvolution probes for the multipoint hypothesis"
    for v in deconv:
        assert v.deconvolution_of == "v_multi"
    # the component singles/pairs were expanded from the parent multipoint.
    assert {v.mutation for v in deconv} <= {"P100A", "Q200B", "P100A;Q200B"}
    assert "P100A" in {v.mutation for v in deconv}


def test_controls_and_uncertainty_lanes_present():
    port = build_portfolio(_bundle(), panel_size=48, controls=_controls(), seed=0)
    control_variants = [v for v in port.variants if v.lane == LANE_CONTROL]
    assert control_variants
    assert all(v.is_control and v.overall_band == BAND_CONTROL for v in control_variants)
    assert {v.control_role for v in control_variants} <= {
        "scalar_top", "geometry_weak", "low_signal"}
    # the high-uncertainty multipoint became a learning probe.
    assert any(v.mutation == "M1N;O2P" for v in port.variants)


def test_claim_safe_reasons():
    port = build_portfolio(_bundle(), panel_size=48, controls=_controls(), seed=0)
    blob = " ".join(v.reason_to_test for v in port.variants)
    assert_clean(blob)   # raises AssertionError on any prohibited over-claim


def test_deterministic():
    b = _bundle()
    p1 = build_portfolio(b, panel_size=48, controls=_controls(), seed=0)
    p2 = build_portfolio(b, panel_size=48, controls=_controls(), seed=0)
    assert p1.as_rows() == p2.as_rows()
    # a fresh identical bundle yields the identical panel too.
    p3 = build_portfolio(_bundle(), panel_size=32, controls=_controls(), seed=0)
    p4 = build_portfolio(_bundle(), panel_size=32, controls=_controls(), seed=0)
    assert p3.as_rows() == p4.as_rows()
    assert len(p3.variants) <= 32


def test_invalid_panel_size_raises():
    with pytest.raises(ValueError):
        build_portfolio(_bundle(), panel_size=50)
    with pytest.raises(ValueError):
        build_portfolio(_bundle(), panel_size=16)


def test_panel_useful_without_strong_band():
    # No strong/significant candidate at all -> still a usable panel from exploratory +
    # controls, and the claim ceiling drops to hypothesis-grade (not L1).
    ledgers = [
        _led("e1", "A1B", BAND_EXPLORATORY),
        _led("e2", "C2D", BAND_EXPLORATORY),
        _led("e3", "E3F", BAND_EXPLORATORY, unc=0.7),
    ]
    bundle = LedgerBundle(run_id="r2", ledgers=ledgers, n_candidates=3)
    port = build_portfolio(bundle, panel_size=24, controls=_controls(), seed=0)
    assert len(port.variants) > 0
    assert any(v.lane == LANE_WT for v in port.variants)
    assert port.claim_ceiling != CLAIM_L1_SCREENING
    assert_clean(" ".join(v.reason_to_test for v in port.variants))


def test_as_rows_keys():
    port = build_portfolio(_bundle(), panel_size=48, controls=_controls(), seed=0)
    rows = port.as_rows()
    assert rows
    expected = {"variant_id", "mutation", "lane", "overall_band", "reason_to_test",
                "is_control", "control_role", "deconvolution_of", "significant_axes",
                "tier_reached"}
    for row in rows:
        assert set(row) == expected


def test_exploratory_fill_uses_budget_when_no_bands():
    """A hard target where NO candidate reaches a band still yields a useful calibration panel:
    the exploratory fill uses the remaining budget with interpretable single-site probes,
    claim-safe, so the first wet-lab round can estimate axis-level enrichment (Gate 7)."""
    from evoliez.portfolio.ledger import AXIS_EVOLUTIONARY, BAND_UNRESOLVED
    # 40 single-site candidates, all UNRESOLVED (no band), with varying cheap scores
    ledgers = []
    for i in range(40):
        axes = {AXIS_STRUCTURAL: _ax(AXIS_STRUCTURAL, band=BAND_UNRESOLVED, score=0.3 + 0.01 * i),
                AXIS_EVOLUTIONARY: _ax(AXIS_EVOLUTIONARY, band=BAND_UNRESOLVED, score=0.5)}
        ledgers.append(EvidenceLedgerV7(variant_id=f"mut_{i:05d}", mutation=f"A{100 + i}K",
                                        axes=axes, overall_band=BAND_UNRESOLVED))
    bundle = LedgerBundle(ledgers=ledgers, n_candidates=len(ledgers))
    pf = build_portfolio(bundle, panel_size=32, seed=0)
    # panel uses (most of) the budget rather than collapsing to just WT despite 0 bands
    assert len(pf.variants) >= 24
    assert len(pf.variants) <= 32
    # every exploratory pick is a claim-safe single-site probe with a reason-to-test
    assert_clean(" ".join(v.reason_to_test for v in pf.variants))
    assert all(v.reason_to_test for v in pf.variants)
    # deterministic
    pf2 = build_portfolio(bundle, panel_size=32, seed=0)
    assert [v.variant_id for v in pf.variants] == [v.variant_id for v in pf2.variants]
