"""Tests for the V7 mechanism-generic controls injector (portfolio.controls).

Ledger inputs are constructed directly from the frozen ledger contract (no sibling
portfolio module is imported) so the test isolates the controls logic.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evoliez.portfolio.controls import (
    CTRL_GEOMETRY_WEAK, CTRL_LOW_SIGNAL, CTRL_RANDOM_MATCHED, CTRL_SCALAR_TOP, CTRL_WT,
    ControlSpec, derive_controls, inject_controls,
)
from evoliez.portfolio.ledger import (
    AXIS_MECHANISM, AXIS_PORTFOLIO, AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL,
    AXIS_UNCERTAINTY, BAND_CONTROL, BAND_DEFERRED, BAND_SIGNIFICANT, BAND_UNRESOLVED,
    AxisEvidenceV7, EvidenceLedgerV7,
)


def _axis(axis, score, band=BAND_UNRESOLVED, missing=False):
    return AxisEvidenceV7(axis=axis, score=score, band=band, missing=missing)


def _led(vid, *, struct=0.5, mech=0.5, unc=0.5, geom=None, geom_band=BAND_UNRESOLVED,
         mutation="", significant=False):
    """Build a ledger with the axes the controls logic reads. ``geom=None`` leaves the
    reaction-geometry axis deferred (the default auto-filled deferred axis)."""
    axes = {
        AXIS_STRUCTURAL: _axis(AXIS_STRUCTURAL, struct,
                               band=BAND_SIGNIFICANT if significant else BAND_UNRESOLVED),
        AXIS_MECHANISM: _axis(AXIS_MECHANISM, mech),
        AXIS_UNCERTAINTY: _axis(AXIS_UNCERTAINTY, unc),
    }
    if geom is not None:
        axes[AXIS_REACTION_GEOMETRY] = _axis(AXIS_REACTION_GEOMETRY, geom, band=geom_band)
    led = EvidenceLedgerV7(variant_id=vid, mutation=mutation, axes=axes)
    led.recompute_overall_band()
    return led


def _make_pool(n=8):
    """A pool of n candidates with spread-out axis scores so each probe has a target."""
    leds = []
    for i in range(n):
        leds.append(_led(
            f"v{i:02d}", mutation=f"A{i}G",
            struct=0.1 * i, mech=0.05 * i, unc=0.1 * (n - i),
            geom=0.1 * i, geom_band=BAND_UNRESOLVED))
    return leds


def test_wt_always_present_and_counts_bounded():
    leds = _make_pool(8)
    ctrls = derive_controls(leds, n_scalar_top=2, n_geometry_weak=2,
                            n_low_signal=2, n_random_matched=2, seed=0)
    by_type = {}
    for c in ctrls:
        by_type.setdefault(c.control_type, []).append(c)
    assert len(by_type[CTRL_WT]) == 1
    assert by_type[CTRL_WT][0].variant_id == "WT"
    assert by_type[CTRL_WT][0].control_type == CTRL_WT
    assert len(by_type[CTRL_SCALAR_TOP]) == 2
    assert len(by_type[CTRL_GEOMETRY_WEAK]) == 2
    assert len(by_type[CTRL_LOW_SIGNAL]) == 2
    assert len(by_type[CTRL_RANDOM_MATCHED]) == 2


def test_no_duplicate_variant_ids():
    leds = _make_pool(10)
    ctrls = derive_controls(leds, seed=3)
    ids = [c.variant_id for c in ctrls]
    assert len(ids) == len(set(ids))


def test_counts_bounded_by_availability():
    # only 2 non-WT candidates but 2+2+2+2 = 8 requested -> bounded by what exists.
    leds = _make_pool(2)
    ctrls = derive_controls(leds, n_scalar_top=2, n_geometry_weak=2,
                            n_low_signal=2, n_random_matched=2, seed=0)
    non_wt = [c for c in ctrls if c.control_type != CTRL_WT]
    ids = [c.variant_id for c in non_wt]
    assert len(ids) == len(set(ids))
    assert len(ids) <= 2  # cannot exceed the candidate pool
    assert sum(1 for c in ctrls if c.control_type == CTRL_WT) == 1


def test_deterministic_same_seed():
    leds = _make_pool(9)
    a = derive_controls(leds, seed=7)
    b = derive_controls(leds, seed=7)
    assert [(c.variant_id, c.control_type) for c in a] == \
           [(c.variant_id, c.control_type) for c in b]


def test_random_matched_seed_sensitive():
    # a large pool so random_matched has real freedom; different seeds should be able to
    # pick a different random-matched set (deterministic per seed, not identical across).
    leds = _make_pool(20)
    a = [c.variant_id for c in derive_controls(leds, seed=1)
         if c.control_type == CTRL_RANDOM_MATCHED]
    b = [c.variant_id for c in derive_controls(leds, seed=999)
         if c.control_type == CTRL_RANDOM_MATCHED]
    assert a != b


def test_scalar_top_excludes_band_supported():
    # v_hi has the highest scalar but is band-supported -> must NOT be a scalar_top probe.
    leds = [
        _led("v_hi", struct=0.99, mech=0.99, unc=0.0, significant=True),
        _led("v_mid", struct=0.6, mech=0.6, unc=0.2),
        _led("v_lo", struct=0.1, mech=0.1, unc=0.9),
    ]
    ctrls = derive_controls(leds, n_scalar_top=1, n_geometry_weak=0,
                            n_low_signal=0, n_random_matched=0, seed=0)
    scalar_ids = [c.variant_id for c in ctrls if c.control_type == CTRL_SCALAR_TOP]
    assert scalar_ids == ["v_mid"]
    assert "v_hi" not in scalar_ids


def test_geometry_weak_fallback_to_low_mechanism():
    # all reaction-geometry deferred -> fallback picks the LOWEST mechanism-consistency.
    leds = [
        _led("g_high_mech", mech=0.9),
        _led("g_low_mech", mech=0.05),
        _led("g_mid_mech", mech=0.5),
    ]
    for led in leds:
        assert led.axes[AXIS_REACTION_GEOMETRY].band == BAND_DEFERRED
    ctrls = derive_controls(leds, n_scalar_top=0, n_geometry_weak=1,
                            n_low_signal=0, n_random_matched=0, seed=0)
    geom = [c.variant_id for c in ctrls if c.control_type == CTRL_GEOMETRY_WEAK]
    assert geom == ["g_low_mech"]


def test_geometry_weak_prefers_present_weak_geometry():
    # geometry present -> pick the weakest present (lowest geom score), not significant.
    leds = [
        _led("g_strong", geom=0.9, geom_band=BAND_SIGNIFICANT),
        _led("g_weak", geom=0.05, geom_band=BAND_UNRESOLVED),
        _led("g_mid", geom=0.5, geom_band=BAND_UNRESOLVED),
    ]
    ctrls = derive_controls(leds, n_scalar_top=0, n_geometry_weak=1,
                            n_low_signal=0, n_random_matched=0, seed=0)
    geom = [c.variant_id for c in ctrls if c.control_type == CTRL_GEOMETRY_WEAK]
    assert geom == ["g_weak"]


def test_low_signal_picks_highest_uncertainty():
    leds = [
        _led("u_low", unc=0.1),
        _led("u_high", unc=0.95),
        _led("u_mid", unc=0.5),
    ]
    ctrls = derive_controls(leds, n_scalar_top=0, n_geometry_weak=0,
                            n_low_signal=1, n_random_matched=0, seed=0)
    low = [c.variant_id for c in ctrls if c.control_type == CTRL_LOW_SIGNAL]
    assert low == ["u_high"]


def test_mechanism_none_and_stub_both_work():
    leds = _make_pool(6)
    # mechanism=None (default) works.
    ctrls_none = derive_controls(leds, seed=0)
    assert any(c.control_type == CTRL_WT for c in ctrls_none)
    # a generic stub with mechanism_spec_id also works and tags provenance.
    mech = SimpleNamespace(mechanism_spec_id="mech_test_v1")
    ctrls_mech = derive_controls(leds, mechanism=mech, seed=0)
    wt = next(c for c in ctrls_mech if c.control_type == CTRL_WT)
    assert any("mech_test_v1" in p for p in wt.provenance)


def test_records_supply_missing_mutation():
    led = _led("v0", mutation="")  # no mutation on the ledger
    records = [{"variant_id": "v0", "mutation": "K12A"}]
    ctrls = derive_controls([led], records=records, n_scalar_top=1, n_geometry_weak=0,
                            n_low_signal=0, n_random_matched=0, seed=0)
    picked = next(c for c in ctrls if c.variant_id == "v0")
    assert picked.mutation == "K12A"


def test_inject_controls_marks_ledgers_and_stamps_portfolio_axis():
    leds = _make_pool(6)
    ctrls = derive_controls(leds, seed=0)
    inject_controls(leds, ctrls)
    ctrl_ids = {c.variant_id for c in ctrls if c.control_type != CTRL_WT}
    for led in leds:
        if led.variant_id in ctrl_ids:
            assert led.is_control is True
            assert led.control_role is not None
            assert led.overall_band == BAND_CONTROL
            pa = led.axes[AXIS_PORTFOLIO]
            assert pa.band == BAND_CONTROL
            assert pa.missing is False
            assert any("control_role" in p for p in pa.provenance)


def test_inject_controls_adds_synthetic_wt_when_absent():
    leds = _make_pool(4)  # no "WT" ledger present
    assert all(led.variant_id != "WT" for led in leds)
    n_before = len(leds)
    ctrls = derive_controls(leds, wt_id="WT", wt_mutation="WT", seed=0)
    inject_controls(leds, ctrls)
    assert len(leds) == n_before + 1
    wt = next(led for led in leds if led.variant_id == "WT")
    assert wt.is_control is True
    assert wt.control_role == CTRL_WT
    assert wt.overall_band == BAND_CONTROL
    assert wt.axes[AXIS_PORTFOLIO].band == BAND_CONTROL


def test_inject_controls_no_duplicate_wt_when_present():
    leds = _make_pool(4)
    # add a WT ledger up front so inject must NOT synthesize a second one.
    leds.append(_led("WT", mutation="WT"))
    n_before = len(leds)
    ctrls = derive_controls(leds, wt_id="WT", wt_mutation="WT", seed=0)
    inject_controls(leds, ctrls)
    assert sum(1 for led in leds if led.variant_id == "WT") == 1
    assert len(leds) == n_before


def test_control_spec_rejects_bad_type():
    with pytest.raises(Exception):
        ControlSpec(variant_id="x", mutation="A1G", control_type="not_a_type",
                    reason="nope")
