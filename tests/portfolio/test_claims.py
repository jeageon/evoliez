"""V7 ledger -> ClaimGuard bridge (ROADMAP_V7 §11).

These tests construct banded ledgers DIRECTLY from the ledger contract (setting
``axes[...].band`` / ``q_value`` by hand) rather than importing any sibling builder, and
assert the claim discipline: pre-wet-lab the bridge never unlocks activity / kinetic
categories, deferred reaction-geometry stays uncalibrated + flagged sparse, replicated
wet-lab lifts the gate, and the portfolio ceiling caps at L1_screening.
"""
import pytest

from evoliez.portfolio import claims
from evoliez.portfolio.claims import (
    ledger_claim_provenance, overall_claim_ceiling, portfolio_claim_allow,
)
from evoliez.portfolio.ledger import (
    AXIS_EVOLUTIONARY, AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL, BAND_SIGNIFICANT,
    BAND_STRONG, BAND_UNRESOLVED, CLAIM_CEILINGS, CLAIM_L0_HYPOTHESIS, CLAIM_L1_SCREENING,
    AxisEvidenceV7, EvidenceLedgerV7, LedgerBundle,
)

# claim-category names as returned by ClaimGuard.evaluate().allow() (kept as literals so the
# test depends only on the ledger + this module, per the module-boundary rule).
ACTIVITY = "activity_improvement"
KINETIC = "kinetic_parameter_prediction"


def _axis(axis, band, *, q=None):
    return AxisEvidenceV7(axis=axis, score=0.9, band=band, q_value=q)


def _bundle(*ledgers) -> LedgerBundle:
    return LedgerBundle(run_id="r", ledgers=list(ledgers), n_candidates=len(ledgers))


def _geom_bundle(band) -> LedgerBundle:
    """A bundle whose single candidate has a REAL (non-deferred) reaction-geometry axis."""
    led = EvidenceLedgerV7(variant_id="v1", mutation="A1B",
                           axes={AXIS_REACTION_GEOMETRY: _axis(AXIS_REACTION_GEOMETRY, band,
                                                               q=0.01)})
    return _bundle(led)


# --- conservative provenance: no wet-lab never unlocks activity / kcat ----------------

def test_no_wetlab_provenance_locks_activity_and_kcat():
    prov = ledger_claim_provenance(_geom_bundle(BAND_SIGNIFICANT))
    # a real banded reaction-geometry axis is capped at hypothesis-grade pre-wet-lab
    assert prov.geometry_claim_ceiling == claims.HYPOTHESIS_GRADE
    assert prov.reference_claim_strength == claims.HYPOTHESIS_GRADE
    assert prov.wt_reaction_geometry_sparse is False
    assert prov.wetlab_replicated is False
    allow = portfolio_claim_allow(prov)
    assert ACTIVITY not in allow
    assert KINETIC not in allow


def test_strong_band_still_capped_without_wetlab():
    prov = ledger_claim_provenance(_geom_bundle(BAND_STRONG))
    # even a STRONG reaction-geometry band cannot exceed hypothesis-grade before wet-lab
    assert prov.geometry_claim_ceiling == claims.HYPOTHESIS_GRADE
    assert ACTIVITY not in portfolio_claim_allow(prov)


# --- deferred reaction geometry -> uncalibrated + sparse ------------------------------

def test_all_deferred_reaction_geometry_is_uncalibrated_and_sparse():
    # ledgers built with no reaction-geometry axis -> auto-filled DEFERRED placeholders
    b = _bundle(EvidenceLedgerV7(variant_id="v1"), EvidenceLedgerV7(variant_id="v2"))
    prov = ledger_claim_provenance(b)
    assert prov.geometry_claim_ceiling == claims.UNCALIBRATED
    assert prov.reference_claim_strength == claims.UNCALIBRATED
    assert prov.wt_reaction_geometry_sparse is True
    assert ACTIVITY not in portfolio_claim_allow(prov)


def test_empty_bundle_is_conservative():
    prov = ledger_claim_provenance(_bundle())
    assert prov.geometry_claim_ceiling == claims.UNCALIBRATED
    assert prov.wt_reaction_geometry_sparse is True


def test_mixed_deferred_and_live_uses_live_axis():
    live = EvidenceLedgerV7(
        variant_id="v1",
        axes={AXIS_REACTION_GEOMETRY: _axis(AXIS_REACTION_GEOMETRY, BAND_SIGNIFICANT, q=0.02)})
    deferred = EvidenceLedgerV7(variant_id="v2")   # reaction-geometry auto-deferred
    prov = ledger_claim_provenance(_bundle(live, deferred))
    assert prov.wt_reaction_geometry_sparse is False       # at least one live axis
    assert prov.geometry_claim_ceiling == claims.HYPOTHESIS_GRADE


# --- replicated wet-lab lifts the gate ------------------------------------------------

def test_wetlab_replicated_unlocks_activity_and_kcat():
    prov = ledger_claim_provenance(
        _geom_bundle(BAND_STRONG), wetlab_replicated=True, known_active_controls=True)
    allow = portfolio_claim_allow(prov)
    assert ACTIVITY in allow
    assert KINETIC in allow
    # with wet-lab the pre-wet-lab hypothesis cap lifts; a strong band supports screening
    assert prov.geometry_claim_ceiling == claims.MODERATE_SCREENING


# --- overall_claim_ceiling caps at L1_screening pre-wet-lab ---------------------------

def test_overall_ceiling_defaults_to_l0():
    # significant support but NO controls -> not promoted
    led = EvidenceLedgerV7(
        variant_id="v1",
        axes={AXIS_STRUCTURAL: _axis(AXIS_STRUCTURAL, BAND_STRONG),
              AXIS_EVOLUTIONARY: _axis(AXIS_EVOLUTIONARY, BAND_SIGNIFICANT)})
    assert overall_claim_ceiling(_bundle(led)) == CLAIM_L0_HYPOTHESIS


def test_overall_ceiling_l1_needs_controls_and_multiaxis():
    strong = EvidenceLedgerV7(
        variant_id="v1",
        axes={AXIS_STRUCTURAL: _axis(AXIS_STRUCTURAL, BAND_STRONG),
              AXIS_EVOLUTIONARY: _axis(AXIS_EVOLUTIONARY, BAND_SIGNIFICANT)})
    control = EvidenceLedgerV7(variant_id="wt", is_control=True, control_role="wt_parental")
    ceiling = overall_claim_ceiling(_bundle(strong, control))
    assert ceiling == CLAIM_L1_SCREENING
    # never above L1 pre-wet-lab, and always a valid ceiling
    assert ceiling in CLAIM_CEILINGS
    assert ceiling != "higher_only_after_wetlab"


def test_controls_without_multiaxis_stays_l0():
    single = EvidenceLedgerV7(
        variant_id="v1",
        axes={AXIS_STRUCTURAL: _axis(AXIS_STRUCTURAL, BAND_STRONG),
              AXIS_EVOLUTIONARY: _axis(AXIS_EVOLUTIONARY, BAND_UNRESOLVED)})  # only 1 sig axis
    control = EvidenceLedgerV7(variant_id="wt", is_control=True)
    assert overall_claim_ceiling(_bundle(single, control)) == CLAIM_L0_HYPOTHESIS
