"""Tests for portfolio.null_models — per-axis null construction (ROADMAP_V7 §4.1).

Depends only on null_models + the ledger contract + numpy/pydantic + pytest. Sibling module
outputs (banded ledgers) are constructed directly from the ledger contract; no sibling
module is imported.
"""
from __future__ import annotations

import numpy as np
import pytest

from evoliez.portfolio.ledger import (
    AXIS_LIGAND, AXIS_MECHANISM, AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL,
    AXIS_UNCERTAINTY, SEVEN_AXES, AxisEvidenceV7, EvidenceLedgerV7,
)
from evoliez.portfolio.null_models import (
    ALL_NULL_SOURCES, NULL_CONTROL_DERIVED, NULL_POPULATION, AxisNull,
    build_axis_nulls, null_source_for,
)


def _led(vid, scores, *, missing_axes=(), is_control=False, control_role=None):
    """A ledger carrying an explicit (non-missing) score on each axis in ``scores`` and
    leaving the rest to auto-fill as deferred (missing)."""
    axes = {}
    for axis, score in scores.items():
        if axis in missing_axes:
            continue
        axes[axis] = AxisEvidenceV7(axis=axis, score=float(score))
    return EvidenceLedgerV7(
        variant_id=vid, axes=axes, is_control=is_control, control_role=control_role)


def _candidates(n=8):
    rng = np.random.default_rng(123)
    leds = []
    for i in range(n):
        leds.append(_led(
            f"v{i:03d}",
            {
                AXIS_STRUCTURAL: rng.uniform(0.2, 0.9),
                AXIS_MECHANISM: rng.uniform(0.1, 0.8),
                AXIS_LIGAND: rng.uniform(0.3, 0.95),
                AXIS_REACTION_GEOMETRY: rng.uniform(0.2, 0.85),
                AXIS_UNCERTAINTY: rng.uniform(0.0, 1.0),
            },
        ))
    return leds


# --- AxisNull schema ------------------------------------------------------------------
def test_axisnull_autofills_n_and_validates_vocab():
    an = AxisNull(axis=AXIS_STRUCTURAL, source=NULL_POPULATION, values=[0.1, 0.2, 0.3])
    assert an.n == 3
    assert an.as_array().tolist() == [0.1, 0.2, 0.3]

    with pytest.raises(Exception):
        AxisNull(axis="not_an_axis", source=NULL_POPULATION)
    with pytest.raises(Exception):
        AxisNull(axis=AXIS_STRUCTURAL, source="made_up_source")
    with pytest.raises(Exception):
        AxisNull(axis=AXIS_STRUCTURAL, source=NULL_POPULATION, bogus=1)  # extra=forbid


# --- population null -------------------------------------------------------------------
def test_population_null_n_equals_nonmissing_count():
    leds = _candidates(8)
    # one candidate has NO structural score (deferred) -> excluded from that axis's null
    leds.append(_led("v_nostruct", {AXIS_MECHANISM: 0.5}))  # structural auto-fills as deferred

    nulls = build_axis_nulls(leds)
    assert nulls[AXIS_STRUCTURAL].source == NULL_POPULATION
    assert nulls[AXIS_STRUCTURAL].n == 8                    # the 9th has no structural score
    assert nulls[AXIS_STRUCTURAL].n == len(nulls[AXIS_STRUCTURAL].values)
    assert nulls[AXIS_MECHANISM].n == 9

    # source vocab is always valid
    for an in nulls.values():
        assert an.source in ALL_NULL_SOURCES


def test_axis_with_no_scores_is_omitted_deferred():
    # nobody carries a reaction-geometry score -> that axis stays deferred (omitted)
    leds = [_led("a", {AXIS_STRUCTURAL: 0.5}), _led("b", {AXIS_STRUCTURAL: 0.6})]
    nulls = build_axis_nulls(leds)
    assert AXIS_STRUCTURAL in nulls
    assert AXIS_REACTION_GEOMETRY not in nulls
    assert AXIS_LIGAND not in nulls
    # never returns a deferred/other axis with a fabricated null
    assert set(nulls).issubset(set(SEVEN_AXES))


def test_uncertainty_gets_population_null_of_disagreement_scores():
    leds = _candidates(6)
    nulls = build_axis_nulls(leds)
    assert AXIS_UNCERTAINTY in nulls
    assert nulls[AXIS_UNCERTAINTY].source == NULL_POPULATION
    expected = [led.axes[AXIS_UNCERTAINTY].score for led in leds]
    assert sorted(nulls[AXIS_UNCERTAINTY].values) == pytest.approx(sorted(expected))


# --- control-derived null (reaction-geometry ONLY, unbiased, >= _MIN_CONTROL_NULL) -----
def test_control_derived_null_only_for_reaction_geometry_with_enough_unbiased_controls():
    leds = _candidates(8)
    # 32 UNBIASED negative controls (random-matched / known-negative) with real spread — a
    # credible background. They also carry a ligand score, which must be IGNORED for the null.
    rng = np.random.default_rng(7)
    controls = [
        _led(f"neg{i:02d}",
             {AXIS_REACTION_GEOMETRY: float(rng.uniform(0.1, 0.5)),
              AXIS_LIGAND: float(rng.uniform(0.1, 0.5))},
             is_control=True, control_role=("negative" if i % 2 else "random_matched"))
        for i in range(32)
    ]
    nulls = build_axis_nulls(leds, control_ledgers=controls)

    # reaction-geometry adopts the (large, unbiased) control-derived null...
    assert nulls[AXIS_REACTION_GEOMETRY].source == NULL_CONTROL_DERIVED
    assert nulls[AXIS_REACTION_GEOMETRY].n == 32
    # ...but the LIGAND (and every other cheap) axis stays on the robust POPULATION null —
    # a null built from score-selected weak controls would be biased low and over-band.
    assert nulls[AXIS_LIGAND].source == NULL_POPULATION
    assert nulls[AXIS_STRUCTURAL].source == NULL_POPULATION
    assert nulls[AXIS_UNCERTAINTY].source == NULL_POPULATION


def test_small_control_set_falls_back_to_population():
    # fewer than _MIN_CONTROL_NULL unbiased controls -> not a credible null -> population.
    leds = _candidates(8)
    controls = [
        _led(f"neg{i}", {AXIS_REACTION_GEOMETRY: 0.05 + 0.01 * i},
             is_control=True, control_role="negative")
        for i in range(5)
    ]
    nulls = build_axis_nulls(leds, control_ledgers=controls)
    assert nulls[AXIS_REACTION_GEOMETRY].source == NULL_POPULATION
    assert nulls[AXIS_REACTION_GEOMETRY].n == 8


def test_only_unbiased_role_controls_feed_the_null():
    # 30 unbiased negatives feed the null; a positive control AND a score-selected
    # geometry_weak control are BOTH excluded (a biased null would inflate significance).
    leds = _candidates(6)
    controls = [
        _led(f"neg{i:02d}", {AXIS_REACTION_GEOMETRY: 0.05 + 0.001 * i},
             is_control=True, control_role="negative")
        for i in range(30)
    ]
    controls.append(_led("pos1", {AXIS_REACTION_GEOMETRY: 0.99},
                         is_control=True, control_role="positive"))
    controls.append(_led("gw1", {AXIS_REACTION_GEOMETRY: 0.02},
                         is_control=True, control_role="geometry_weak"))
    nulls = build_axis_nulls(leds, control_ledgers=controls)
    assert nulls[AXIS_REACTION_GEOMETRY].source == NULL_CONTROL_DERIVED
    assert nulls[AXIS_REACTION_GEOMETRY].n == 30           # only the 30 unbiased negatives
    assert 0.99 not in nulls[AXIS_REACTION_GEOMETRY].values   # positive excluded
    assert 0.02 not in nulls[AXIS_REACTION_GEOMETRY].values   # geometry_weak (biased) excluded


def test_ligand_axis_always_population_even_with_controls():
    # regression guard: a small biased control set must NEVER become the ligand null (the
    # 6-control ligand null was the false-positive-explosion bug).
    leds = _candidates(8)
    controls = [
        _led("c1", {AXIS_LIGAND: 0.05}, is_control=True, control_role="geometry_weak"),
        _led("c2", {AXIS_LIGAND: 0.08}, is_control=True, control_role="low_signal"),
    ]
    nulls = build_axis_nulls(leds, control_ledgers=controls)
    assert nulls[AXIS_LIGAND].source == NULL_POPULATION
    assert nulls[AXIS_LIGAND].n == 8


# --- null_source_for policy documentation ---------------------------------------------
def test_null_source_for_matches_build_policy():
    # only reaction-geometry can (intend to) use a control-derived null; ligand does NOT.
    assert null_source_for(AXIS_REACTION_GEOMETRY, has_controls=True) == NULL_CONTROL_DERIVED
    assert null_source_for(AXIS_LIGAND, has_controls=True) == NULL_POPULATION
    assert null_source_for(AXIS_REACTION_GEOMETRY, has_controls=False) == NULL_POPULATION
    assert null_source_for(AXIS_STRUCTURAL, has_controls=True) == NULL_POPULATION
    assert null_source_for(AXIS_UNCERTAINTY, has_controls=True) == NULL_POPULATION


# --- determinism -----------------------------------------------------------------------
def test_deterministic_under_fixed_seed():
    leds = _candidates(10)
    controls = [
        _led("c1", {AXIS_REACTION_GEOMETRY: 0.05, AXIS_LIGAND: 0.1},
             is_control=True, control_role="geometry_weak"),
    ]
    a = build_axis_nulls(leds, control_ledgers=controls, seed=0)
    b = build_axis_nulls(leds, control_ledgers=controls, seed=0)
    assert {k: v.model_dump() for k, v in a.items()} == {k: v.model_dump() for k, v in b.items()}


def test_mechanism_arg_accepted_and_ignored_for_now():
    leds = _candidates(5)

    class _FakeMech:  # any object is tolerated (reserved arg)
        pass

    with_mech = build_axis_nulls(leds, mechanism=_FakeMech())
    without = build_axis_nulls(leds)
    assert {k: v.model_dump() for k, v in with_mech.items()} == \
        {k: v.model_dump() for k, v in without.items()}
