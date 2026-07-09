"""Tests for evoliez.portfolio.bands (ROADMAP_V7 Phase V7-2 evidence bands).

These tests build a ``LedgerBundle`` directly from the ledger contract (never importing a
sibling module) and calibrate it with a self-made ``.values`` / ``.source`` null object, so
the only V7 dependencies are ``bands`` + ``ledger`` + ``stats``.
"""
from __future__ import annotations

import numpy as np

from evoliez.portfolio.bands import BandThresholds, assign_bands, band_summary
from evoliez.portfolio.ledger import (
    AXIS_EVOLUTIONARY, AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL, AXIS_UNCERTAINTY,
    BAND_DEFERRED, BAND_SIGNIFICANT, BAND_STRONG, BAND_UNRESOLVED, ALL_BANDS,
    AxisEvidenceV7, EvidenceLedgerV7, LedgerBundle,
)


class _Null:
    """Minimal AxisNull stand-in: exposes ``.values`` (the null draws) and ``.source``."""

    def __init__(self, values, source="test_reference_null"):
        self.values = np.asarray(values, dtype=float)
        self.source = source


# A broad reference null with real spread (median 0.5, non-zero MAD) so effect sizes are
# meaningful and a large null lets a single outlier survive the BH multiplicity penalty.
_REF_NULL = _Null(np.linspace(0.30, 0.70, 500))


def _bundle(struct_scores, *, evo_scores=None):
    """Build a bundle with a structural axis (and optionally an evolutionary axis) populated;
    every other axis auto-fills as deferred via the ledger validator."""
    ledgers = []
    for i, s in enumerate(struct_scores):
        axes = {AXIS_STRUCTURAL: AxisEvidenceV7(axis=AXIS_STRUCTURAL, score=float(s))}
        if evo_scores is not None:
            axes[AXIS_EVOLUTIONARY] = AxisEvidenceV7(
                axis=AXIS_EVOLUTIONARY, score=float(evo_scores[i]))
        ledgers.append(EvidenceLedgerV7(variant_id=f"v{i:03d}", axes=axes))
    return LedgerBundle(run_id="r", target_id="t", ledgers=ledgers,
                        n_candidates=len(ledgers))


def test_outlier_reaches_strong_background_unresolved():
    # one clear high outlier, flat background exactly at the null median
    scores = [0.99] + [0.50] * 9
    bundle = _bundle(scores)
    assign_bands(bundle, nulls={AXIS_STRUCTURAL: _REF_NULL})

    by_id = bundle.by_id()
    out = by_id["v000"].axes[AXIS_STRUCTURAL]
    assert out.is_significant()          # strong or significant
    assert out.band == BAND_STRONG       # q ~= 0.02 <= strong_q, |effect| >> 0.5
    assert out.effect_size is not None and out.effect_size > 0.5
    assert out.null_source.startswith("test_reference_null")   # + the calibration method tag

    for i in range(1, 10):
        bg = by_id[f"v{i:03d}"].axes[AXIS_STRUCTURAL]
        assert bg.band == BAND_UNRESOLVED
    # the null size is recorded for the report
    assert bundle.axis_null_sizes[AXIS_STRUCTURAL] == 500


def test_q_values_are_monotone_in_p():
    scores = [0.99, 0.90, 0.80] + [0.50] * 12
    bundle = _bundle(scores)
    assign_bands(bundle, nulls={AXIS_STRUCTURAL: _REF_NULL})

    pairs = []
    for led in bundle.ledgers:
        ax = led.axes[AXIS_STRUCTURAL]
        pairs.append((ax.empirical_p, ax.q_value))
    pairs.sort(key=lambda pq: pq[0])
    qs = [q for _, q in pairs]
    for a, b in zip(qs, qs[1:]):
        assert a <= b + 1e-12          # BH q is non-decreasing along ascending p
    assert all(0.0 <= q <= 1.0 for q in qs)


def test_deferred_axis_stays_deferred():
    bundle = _bundle([0.99] + [0.50] * 9)
    assign_bands(bundle, nulls={AXIS_STRUCTURAL: _REF_NULL})
    # reaction_geometry was auto-filled deferred and has no null -> untouched
    for led in bundle.ledgers:
        rg = led.axes[AXIS_REACTION_GEOMETRY]
        assert rg.missing is True
        assert rg.band == BAND_DEFERRED
        assert rg.q_value is None
        assert rg.effect_size is None
    assert AXIS_REACTION_GEOMETRY not in bundle.axis_null_sizes


def test_subset_level_flag_propagates_and_notes():
    bundle = _bundle([0.99] + [0.50] * 9)
    assign_bands(bundle, nulls={AXIS_STRUCTURAL: _REF_NULL}, subset_level=True)
    for led in bundle.ledgers:
        assert led.axes[AXIS_STRUCTURAL].subset_level is True
        # deferred axes are never recalibrated, so they never get flagged
        assert led.axes[AXIS_REACTION_GEOMETRY].subset_level is False
    assert any("subset level" in note for note in bundle.notes)


def test_higher_is_worse_axis_orientation():
    # uncertainty axis: LOW score is beneficial -> should band; HIGH score must not.
    ledgers = []
    for i, s in enumerate([0.05] + [0.95] * 9):
        axes = {AXIS_UNCERTAINTY: AxisEvidenceV7(axis=AXIS_UNCERTAINTY, score=float(s))}
        ledgers.append(EvidenceLedgerV7(variant_id=f"u{i:03d}", axes=axes))
    bundle = LedgerBundle(ledgers=ledgers, n_candidates=len(ledgers))
    assign_bands(bundle, nulls={AXIS_UNCERTAINTY: _REF_NULL})

    by_id = bundle.by_id()
    low = by_id["u000"].axes[AXIS_UNCERTAINTY]
    assert low.is_significant()
    assert low.effect_size is not None and low.effect_size > 0.5   # sign-flipped to +ve
    for i in range(1, 10):
        hi = by_id[f"u{i:03d}"].axes[AXIS_UNCERTAINTY]
        assert hi.band == BAND_UNRESOLVED


def test_self_population_null_built_when_none():
    scores = [0.99] + [0.50] * 9
    bundle = _bundle(scores)
    assign_bands(bundle, nulls=None)     # no external null -> self-population null
    out = bundle.by_id()["v000"].axes[AXIS_STRUCTURAL]
    assert out.null_source.startswith("population_self")   # null was the population itself
    assert bundle.axis_null_sizes[AXIS_STRUCTURAL] == 10
    # The robust-Gaussian empirical null fits the null's location/scale from the bulk (median /
    # MAD), so its analytic tail p (unlike the count-based p, floored at 1/(n+1)) lets a clear
    # outlier survive BH even at a self-population null — but the FLAT BACKGROUND must NOT band
    # (no false discovery). null_source records "population_self" so a report can down-weight
    # self-null evidence relative to an independent control-derived null.
    assert out.band in (BAND_STRONG, BAND_SIGNIFICANT)
    assert out.q_value is not None and 0.0 <= out.q_value <= 1.0
    for i in range(1, 10):
        bg = bundle.by_id()[f"v{i:03d}"].axes[AXIS_STRUCTURAL]
        assert bg.band == BAND_UNRESOLVED


def test_consensus_promotes_overall_band():
    # a candidate strong on TWO axes should be promoted to (at least) consensus overall.
    scores = [0.99] + [0.50] * 9
    evo = [0.99] + [0.50] * 9
    bundle = _bundle(scores, evo_scores=evo)
    thr = BandThresholds(consensus_min_axes=2)
    assign_bands(bundle, nulls={AXIS_STRUCTURAL: _REF_NULL, AXIS_EVOLUTIONARY: _REF_NULL},
                 thresholds=thr)
    top = bundle.by_id()["v000"]
    assert set(top.consensus_axes) >= {AXIS_STRUCTURAL, AXIS_EVOLUTIONARY}
    # strongest-axis band is strong; overall band must be a real (non-deferred) band
    assert top.overall_band in (BAND_STRONG, BAND_SIGNIFICANT)


def test_deterministic():
    scores = [0.99, 0.85] + [0.50] * 8
    b1 = _bundle(scores)
    b2 = _bundle(scores)
    assign_bands(b1, nulls={AXIS_STRUCTURAL: _REF_NULL}, seed=0)
    assign_bands(b2, nulls={AXIS_STRUCTURAL: _REF_NULL}, seed=123)
    for l1, l2 in zip(b1.ledgers, b2.ledgers):
        a1 = l1.axes[AXIS_STRUCTURAL]
        a2 = l2.axes[AXIS_STRUCTURAL]
        assert a1.band == a2.band
        assert a1.q_value == a2.q_value
        assert a1.empirical_p == a2.empirical_p
        assert a1.effect_size == a2.effect_size


def test_band_summary_shape_and_counts():
    scores = [0.99] + [0.50] * 9
    bundle = _bundle(scores)
    assign_bands(bundle, nulls={AXIS_STRUCTURAL: _REF_NULL})
    summary = band_summary(bundle)
    assert "overall" in summary
    # every axis present, every band key present (0-filled)
    assert set(ALL_BANDS) <= set(summary[AXIS_STRUCTURAL])
    # structural: 1 strong + 9 unresolved = 10 candidates
    assert summary[AXIS_STRUCTURAL][BAND_STRONG] == 1
    assert summary[AXIS_STRUCTURAL][BAND_UNRESOLVED] == 9
    assert sum(summary[AXIS_STRUCTURAL].values()) == 10
    # reaction_geometry axis is fully deferred
    assert summary[AXIS_REACTION_GEOMETRY][BAND_DEFERRED] == 10
