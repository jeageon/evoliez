"""V7 multi-fidelity compute allocator (ROADMAP_V7 Phase V7-3, §5).

V7 is GPU-first but not GPU-wasteful: every variant earns a cheap Tier-0 ledger, and each
expensive tier is reserved for the *subset whose decision could still change* at that cost.
This module assigns each candidate to the CHEAPEST tier that could move its band, and — this
is Gate 3 — attaches to every non-cheap assignment a plain-language reason stating WHICH
band / lane / uncertainty signal justified spending the compute. So the run can always
"explain why each expensive calculation was run", and the whole universe is never escalated
to PMF/QM-MM.

Escalation policy (each tier is a bounded subset of the tier below it):

  * ``tier0_cheap``   — EVERY candidate. All-variant cheap evidence (CPU / light GPU).
  * ``tier1_gpu_broad`` — anything a broad GPU pass could still resolve: an axis in the
    consensus q-band, a control, a deconvolution probe, or a high-uncertainty / high-
    learning-value candidate. Capped, ranked by strongest band then lowest min-q.
  * ``tier2_focused_md`` — the mechanism hypotheses worth explicit MD: candidates that
    reached a significant/strong band or cross-axis consensus. Capped subset of Tier-1.
  * ``tier3_reaction_core`` — PMF / QM-MM-lite, the most expensive tier: only the top
    strong-band mechanism hypotheses plus unresolved-but-pivotal deconvolution probes.
    Tightly capped subset of Tier-2.

GPU hints round-robin over the configured pool (one independent job per GPU). Everything is
deterministic given the ledger contents (stable sort with a ``variant_id`` tiebreak); this
module depends only on the ledger contract + stdlib, so it runs and tests in the light env.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from evoliez.portfolio.ledger import (
    ALL_TIERS, AXIS_UNCERTAINTY, BAND_CONSENSUS, BAND_EXPLORATORY, BAND_STRONG, SEVEN_AXES,
    TIER_CHEAP, TIER_EXPENSIVE, TIER_FOCUSED, TIER_GPU_BROAD, EvidenceLedgerV7, LedgerBundle,
    band_rank,
)


@dataclass(frozen=True)
class TierBudget:
    """Per-tier caps + the GPU pool. Caps bound how many candidates reach each expensive
    tier (so no tier ever escalates the whole universe). ``gpu_pool`` is a tuple of GPU ids
    to round-robin over; empty inherits the caller's default device. ``consensus_q`` is the
    q-value threshold at which an axis is broad-validation-worthy."""

    tier1_gpu_broad_max: int = 80
    tier2_focused_md_max: int = 40
    tier3_reaction_core_max: int = 12
    gpu_pool: Tuple[int, ...] = ()
    consensus_q: float = 0.10


class _Base(BaseModel):
    model_config = {"extra": "forbid"}


class TierAssignment(_Base):
    """One candidate's fidelity assignment. ``reason`` is non-empty for every non-cheap tier
    and states which band / lane / uncertainty signal justified the spend (Gate 3)."""

    variant_id: str
    mutation: str = ""
    tier: str
    reason: str = ""
    gpu_hint: Optional[int] = None

    @model_validator(mode="after")
    def _check(self) -> "TierAssignment":
        if self.tier not in ALL_TIERS:
            raise ValueError(f"tier must be one of {ALL_TIERS}, got {self.tier!r}")
        return self


class TierPlan(_Base):
    """The full allocation: per-candidate assignments, a tier->variant_id index, and notes
    recording where caps bound. ``count(tier)`` = how many candidates were assigned to that
    tier (its highest reached fidelity)."""

    assignments: List[TierAssignment] = Field(default_factory=list)
    by_tier: Dict[str, List[str]] = Field(default_factory=dict)
    budget_notes: List[str] = Field(default_factory=list)

    def count(self, tier: str) -> int:
        return len(self.by_tier.get(tier, []))


# --- eligibility signals -------------------------------------------------------------
def _axis_min_q(led: EvidenceLedgerV7) -> float:
    """Smallest finite q-value across the seven axes (inf when no axis is calibrated)."""
    qs = [led.axes[ax].q_value for ax in SEVEN_AXES]
    finite = [q for q in qs if q is not None and math.isfinite(q)]
    return min(finite) if finite else math.inf


def _low_q_axes(led: EvidenceLedgerV7, consensus_q: float) -> List[str]:
    """Axes sitting at or below the consensus q-band."""
    out = []
    for ax in SEVEN_AXES:
        q = led.axes[ax].q_value
        if q is not None and math.isfinite(q) and q <= consensus_q:
            out.append(ax)
    return out


def _high_learning_value(led: EvidenceLedgerV7) -> bool:
    """True when the uncertainty axis flags this candidate as worth probing to learn — a
    high-learning-value / exploratory band (never a deferred / missing axis)."""
    ev = led.axes[AXIS_UNCERTAINTY]
    if ev.missing:
        return False
    return ev.band == BAND_EXPLORATORY or ev.is_significant()


def _rank_key(led: EvidenceLedgerV7) -> Tuple[int, float, str]:
    """Escalation ordering: strongest overall band first, then lowest min-q, then id."""
    return (band_rank(led.overall_band), _axis_min_q(led), led.variant_id)


def _focused_reasons(led: EvidenceLedgerV7) -> List[str]:
    """Why a candidate is worth explicit MD (empty => not focused-eligible)."""
    reasons: List[str] = []
    sig = led.significant_axes()
    if sig:
        reasons.append(f"significant/strong band on {', '.join(sig)} — explicit MD")
    if led.overall_band == BAND_CONSENSUS:
        reasons.append("cross-axis consensus — explicit MD")
    return reasons


def _gpu_broad_reasons(led: EvidenceLedgerV7, budget: TierBudget) -> List[str]:
    """Why a candidate is worth a broad GPU pass (empty => not Tier-1-eligible). A superset
    of the focused triggers, so focused-eligible candidates are always Tier-1-eligible too."""
    reasons: List[str] = []
    low = _low_q_axes(led, budget.consensus_q)
    if low:
        reasons.append(f"q<={budget.consensus_q:g} band on {', '.join(low)}")
    if led.is_control:
        reasons.append(f"control ({led.control_role or 'design-selected'})")
    if led.deconvolution_of:
        reasons.append(f"deconvolution probe of {led.deconvolution_of}")
    if _high_learning_value(led):
        reasons.append("high uncertainty / learning value")
    reasons.extend(_focused_reasons(led))
    return reasons


def _expensive_info(led: EvidenceLedgerV7) -> Tuple[bool, int, str]:
    """(eligible, priority, reason) for the reaction-core tier. Strong-band mechanism
    hypotheses have priority 0; unresolved-but-pivotal deconvolution probes priority 1."""
    if led.overall_band == BAND_STRONG:
        return (True, 0, "strong-band mechanism hypothesis — PMF/QM-MM reaction core")
    if led.deconvolution_of:
        return (True, 1,
                f"unresolved-but-pivotal deconvolution probe of {led.deconvolution_of} "
                "— reaction-core confirmation")
    return (False, 99, "")


def allocate_tiers(bundle: LedgerBundle, *, budget: TierBudget = TierBudget()) -> TierPlan:
    """Assign each candidate to the cheapest tier that could still change its decision.

    Each expensive tier is a bounded subset of the one below it (Tier-3 ⊆ Tier-2 ⊆ Tier-1 ⊆
    universe), so the whole universe is never escalated to an expensive tier. Every non-cheap
    assignment carries a reason naming the triggering band / lane / uncertainty signal.
    ``gpu_hint`` round-robins over ``budget.gpu_pool`` across the non-cheap assignments (in
    ledger order). Deterministic."""
    ledgers = list(bundle.ledgers)
    notes: List[str] = []

    # Tier-1: broad GPU pass, capped, ranked strongest-band-then-lowest-q.
    gpu_eligible = [led for led in ledgers if _gpu_broad_reasons(led, budget)]
    gpu_ranked = sorted(gpu_eligible, key=_rank_key)
    gpu_set = gpu_ranked[:budget.tier1_gpu_broad_max]
    gpu_ids = {led.variant_id for led in gpu_set}
    if len(gpu_eligible) > budget.tier1_gpu_broad_max:
        notes.append(f"tier1_gpu_broad capped at {budget.tier1_gpu_broad_max} "
                     f"({len(gpu_eligible)} eligible)")

    # Tier-2: focused MD, a capped subset of Tier-1.
    focused_eligible = [led for led in gpu_set if _focused_reasons(led)]
    focused_ranked = sorted(focused_eligible, key=_rank_key)
    focused_set = focused_ranked[:budget.tier2_focused_md_max]
    focused_ids = {led.variant_id for led in focused_set}
    if len(focused_eligible) > budget.tier2_focused_md_max:
        notes.append(f"tier2_focused_md capped at {budget.tier2_focused_md_max} "
                     f"({len(focused_eligible)} eligible)")

    # Tier-3: reaction core, a tightly capped subset of Tier-2.
    exp_candidates = []
    for led in focused_set:
        eligible, prio, reason = _expensive_info(led)
        if eligible:
            exp_candidates.append((prio, _rank_key(led), led.variant_id, reason))
    exp_candidates.sort(key=lambda t: (t[0], t[1]))
    expensive_selected = exp_candidates[:budget.tier3_reaction_core_max]
    expensive_reason = {vid: reason for (_, _, vid, reason) in expensive_selected}
    expensive_ids = set(expensive_reason)
    if len(exp_candidates) > budget.tier3_reaction_core_max:
        notes.append(f"tier3_reaction_core capped at {budget.tier3_reaction_core_max} "
                     f"({len(exp_candidates)} eligible)")

    # Assign each candidate its highest reached tier; round-robin gpu_hint over the pool.
    assignments: List[TierAssignment] = []
    by_tier: Dict[str, List[str]] = {tier: [] for tier in ALL_TIERS}
    gpu_counter = 0
    for led in ledgers:
        vid = led.variant_id
        if vid in expensive_ids:
            tier, reason = TIER_EXPENSIVE, expensive_reason[vid]
        elif vid in focused_ids:
            tier, reason = TIER_FOCUSED, "; ".join(_focused_reasons(led))
        elif vid in gpu_ids:
            tier, reason = TIER_GPU_BROAD, "; ".join(_gpu_broad_reasons(led, budget))
        else:
            tier, reason = TIER_CHEAP, "tier0 cheap all-variant baseline evidence"
        gpu_hint: Optional[int] = None
        if tier != TIER_CHEAP and budget.gpu_pool:
            gpu_hint = budget.gpu_pool[gpu_counter % len(budget.gpu_pool)]
            gpu_counter += 1
        assignments.append(TierAssignment(variant_id=vid, mutation=led.mutation, tier=tier,
                                           reason=reason, gpu_hint=gpu_hint))
        by_tier[tier].append(vid)

    notes.insert(0, "tiers: cheap={} gpu_broad={} focused={} reaction_core={}".format(
        len(by_tier[TIER_CHEAP]), len(by_tier[TIER_GPU_BROAD]),
        len(by_tier[TIER_FOCUSED]), len(by_tier[TIER_EXPENSIVE])))
    return TierPlan(assignments=assignments, by_tier=by_tier, budget_notes=notes)


def non_converged_pmf_gate(assignment_ids: List[str],
                           converged: Dict[str, bool]) -> List[str]:
    """Return the ids whose PMF did NOT converge — a caller uses this to BLOCK them from
    ranking (ROADMAP_V7 §4/§5: a non-converged PMF is not evidence and cannot enter ranking).
    An id absent from ``converged`` counts as non-converged. Order is preserved."""
    return [vid for vid in assignment_ids if not converged.get(vid, False)]
