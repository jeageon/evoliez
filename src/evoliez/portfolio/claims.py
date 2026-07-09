"""V7 ledger -> ClaimGuard bridge (ROADMAP_V7 §11 claim discipline).

V7 does not invent a second claim vocabulary. It reads the seven-axis evidence ledger and
translates it into the SHARED provenance the existing keystone ClaimGuard already reasons
over (``evoliez.ranking.claim_guard``), so a rendered V7 portfolio report is linted against
exactly the same over-claim rules as every earlier pipeline stage — one vocabulary, one
guard, no divergence.

The translation is deliberately CONSERVATIVE and fail-safe (ROADMAP_V7 §2 / §11):

  * Reference and reaction-geometry claim strengths are derived from the *strongest*
    reaction-geometry axis actually present across the bundle. A deferred / missing
    reaction-geometry axis contributes ``UNCALIBRATED`` (an honest gap, never a strength it
    did not earn); a real, banded reaction-geometry axis contributes at most
    ``HYPOTHESIS_GRADE`` pre-wet-lab. If EVERY reaction-geometry axis is deferred the run is
    flagged ``wt_reaction_geometry_sparse`` (diagnostic-only geometry).
  * Nothing here can unlock an activity / kcat / validated-lead claim: those categories are
    unlocked by ``claim_guard.evaluate`` only when ``wetlab_replicated`` is true, and that
    flag is passed through untouched (default ``False``). Pre-wet-lab the guard stays shut.
  * The portfolio-level V7 claim ceiling caps at ``L1_screening`` before wet-lab, and only
    reaches even that when the panel carries controls AND some candidate has multi-axis
    (>= 2) significant support.

Pure schema + stdlib logic; imports in the light env (no numpy needed).
"""
from __future__ import annotations

from typing import List

from evoliez.mechanism.vocab import (
    HYPOTHESIS_GRADE, MODERATE_SCREENING, UNCALIBRATED, weakest_claim,
)
from evoliez.portfolio.ledger import (
    AXIS_REACTION_GEOMETRY, BAND_CONSENSUS, BAND_CONTROL, BAND_DEFERRED, BAND_EXPLORATORY,
    BAND_SIGNIFICANT, BAND_STRONG, BAND_UNRESOLVED, CLAIM_L0_HYPOTHESIS, CLAIM_L1_SCREENING,
    LedgerBundle, strongest_band,
)
from evoliez.ranking.claim_guard import ClaimProvenance, evaluate

# A real, banded reaction-geometry axis maps to a claim strength via its band. This is a
# CEILING that pre-wet-lab is further capped at HYPOTHESIS_GRADE (see below); with replicated
# wet-lab the cap lifts and a strong band may support screening-level reference claims.
# A deferred axis maps to UNCALIBRATED — the gap contributes no strength.
_BAND_TO_STRENGTH = {
    BAND_STRONG: MODERATE_SCREENING,
    BAND_SIGNIFICANT: MODERATE_SCREENING,
    BAND_CONSENSUS: HYPOTHESIS_GRADE,
    BAND_EXPLORATORY: HYPOTHESIS_GRADE,
    BAND_CONTROL: HYPOTHESIS_GRADE,
    BAND_UNRESOLVED: HYPOTHESIS_GRADE,
    BAND_DEFERRED: UNCALIBRATED,
}


def _reaction_geometry_axes(bundle: LedgerBundle):
    """Every reaction-geometry axis across the bundle (always present per the ledger
    contract, but may be a deferred placeholder)."""
    return [led.axes[AXIS_REACTION_GEOMETRY] for led in bundle.ledgers]


def _is_deferred(axis) -> bool:
    return bool(axis.missing) or axis.band == BAND_DEFERRED


def _geometry_claim_strength(bundle: LedgerBundle, *, wetlab_replicated: bool):
    """Derive (strength, sparse) from the strongest reaction-geometry axis in the bundle.

    ``strength`` is the reference / geometry claim ceiling the ledger supports; ``sparse`` is
    True iff every reaction-geometry axis is deferred (or none exist)."""
    rg = _reaction_geometry_axes(bundle)
    live = [ax for ax in rg if not _is_deferred(ax)]
    if not live:                                        # all deferred / empty bundle
        return UNCALIBRATED, True
    best_band = strongest_band(*(ax.band for ax in live))
    strength = _BAND_TO_STRENGTH.get(best_band, UNCALIBRATED)
    if not wetlab_replicated:                           # pre-wet-lab cap: <= HYPOTHESIS_GRADE
        strength = weakest_claim(strength, HYPOTHESIS_GRADE)
    return strength, False


def ledger_claim_provenance(
    bundle: LedgerBundle,
    *,
    wetlab_replicated: bool = False,
    known_active_controls: bool = False,
    known_inactive_controls: bool = False,
    enhanced_sampling_or_qmmm: bool = False,
    only_short_md: bool = True,
) -> ClaimProvenance:
    """Build a conservative, whole-portfolio :class:`ClaimProvenance` from a V7 ledger bundle.

    Reference / geometry claim strengths derive from the strongest reaction-geometry axis
    present across the ledgers (deferred -> ``UNCALIBRATED``; real banded axis -> at most
    ``HYPOTHESIS_GRADE`` unless wet-lab). ``wt_reaction_geometry_sparse`` is set when every
    reaction-geometry axis is deferred. All wet-lab / sampling / control flags are passed
    through unchanged with conservative defaults, so pre-wet-lab the guard NEVER unlocks an
    activity or kinetic-parameter claim."""
    strength, sparse = _geometry_claim_strength(bundle, wetlab_replicated=wetlab_replicated)
    return ClaimProvenance(
        reference_claim_strength=strength,
        geometry_claim_ceiling=strength,
        wetlab_replicated=wetlab_replicated,
        only_short_md=only_short_md,
        enhanced_sampling_or_qmmm=enhanced_sampling_or_qmmm,
        known_active_controls=known_active_controls,
        known_inactive_controls=known_inactive_controls,
        wt_reaction_geometry_sparse=sparse,
    )


def portfolio_claim_allow(prov: ClaimProvenance) -> List[str]:
    """Claim categories a report built on this provenance MAY assert (sorted). Thin wrapper
    over the keystone guard so callers depend on ONE decision path."""
    return evaluate(prov).allow()


def overall_claim_ceiling(bundle: LedgerBundle) -> str:
    """Highest V7 claim ceiling the bundle supports (one of ``ledger.CLAIM_CEILINGS``).

    ``L0_hypothesis`` by default. Promoted to ``L1_screening`` only when the panel carries
    controls AND at least one candidate has multi-axis (>= 2) significant support. Never
    above ``L1_screening`` here — anything stronger requires replicated wet-lab evidence,
    which is not a property of the computational ledger."""
    has_controls = any(led.is_control for led in bundle.ledgers)
    multi_axis_support = any(
        len(led.significant_axes()) >= 2 for led in bundle.ledgers)
    if has_controls and multi_axis_support:
        return CLAIM_L1_SCREENING
    return CLAIM_L0_HYPOTHESIS
