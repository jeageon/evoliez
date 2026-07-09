"""Regression tests for the V7 evidence-discipline hardening (Fable safety review, 2026-07-09).

(1) Axis-4: reaction-geometry must not band on a short DISTANCE alone — it requires a real
ABSOLUTE near-attack occupancy, not ΔNAC-vs-WT (which maps 0 -> a neutral 0.5). (2) subset_level
is sticky: an axis computed on the MD subset stays subset-level even if the bundle covers it fully.
"""
from __future__ import annotations

from evoliez.portfolio import bands as B
from evoliez.portfolio import cheap_axes as CA
from evoliez.portfolio.ledger import (
    AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL, BAND_UNRESOLVED, AxisEvidenceV7,
    EvidenceLedgerV7, LedgerBundle, TIER_FOCUSED,
)


def test_delta_only_nac_defers_reaction_geometry():
    # ΔNAC present, absolute occupancy ABSENT, one short distance -> must NOT band; DEFERRED.
    recs = [{"candidate_id": f"mut_{i:05d}", "mutation_string": f"A{i}K", "generator": "multipoint",
             "nac_delta_vs_wt": 0.0, "md_lite_score": 0.5,
             "catalytic_distance_mean": 3.0 if i == 0 else 7.5} for i in range(8)]
    leds = CA.build_cheap_ledgers(recs)
    CA.enrich_expensive_axes(recs, leds)
    assert all(led.axes[AXIS_REACTION_GEOMETRY].missing for led in leds)


def test_absolute_nac_fills_reaction_geometry():
    recs = [{"candidate_id": f"mut_{i:05d}", "mutation_string": f"A{i}K", "generator": "multipoint",
             "nac_occupancy": 0.15, "md_lite_score": 0.5, "catalytic_distance_mean": 3.2}
            for i in range(6)]
    leds = CA.build_cheap_ledgers(recs)
    n = CA.enrich_expensive_axes(recs, leds)
    assert n == 6
    rg = leds[0].axes[AXIS_REACTION_GEOMETRY]
    assert rg.missing is False and rg.tier == TIER_FOCUSED and rg.subset_level is True


def test_subset_level_flag_is_sticky_through_assign_bands():
    # a fully-covered axis pre-flagged subset_level (by enrichment) must NOT be reset to
    # full-population by assign_bands.
    ledgers = []
    for i in range(10):
        rg = AxisEvidenceV7(axis=AXIS_REACTION_GEOMETRY, score=0.3 + 0.02 * i,
                            band=BAND_UNRESOLVED, subset_level=True, missing=False,
                            tier=TIER_FOCUSED)
        st = AxisEvidenceV7(axis=AXIS_STRUCTURAL, score=0.5, band=BAND_UNRESOLVED)
        ledgers.append(EvidenceLedgerV7(variant_id=f"v{i}", axes={
            AXIS_REACTION_GEOMETRY: rg, AXIS_STRUCTURAL: st}))
    bundle = LedgerBundle(ledgers=ledgers, n_candidates=len(ledgers))
    B.assign_bands(bundle, nulls=None, subset_level=False)
    # reaction-geometry covered all 10 ledgers, yet stays subset_level (sticky)...
    assert all(l.axes[AXIS_REACTION_GEOMETRY].subset_level for l in bundle.ledgers)
    # ...while the ordinary full-population structural axis is not subset-level.
    assert not any(l.axes[AXIS_STRUCTURAL].subset_level for l in bundle.ledgers)
