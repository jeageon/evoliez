"""Plate-budget lane allocator + control strategy (ROADMAP_V3 D8 / V3-6).

The v2 ``multi_lane`` percentages can sum past a 96-well plate. This allocator turns
them into a HARD BUDGET: controls are reserved FIRST, evidence lanes fill the remainder
in priority order, nothing overflows, and every candidate records ALL lanes that would
have surfaced it (not just the first) so a post-experiment analysis can learn which
selection logic mattered. Capacity dropped by the budget is reported, never silent
(ROADMAP_V3 invariant: no silent caps).

"Known controls mandatory" is softened to "control strategy mandatory; known controls
when available" — a target lacking known active/inactive proceeds but is flagged
``uncalibrated``.

Duck-typed on (candidate_id, scores: dict, mutations, details: dict); light env.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


# --- plate budget --------------------------------------------------------------------
@dataclass
class PlateBudget:
    total: int = 96
    # control reservations (taken first)
    wt_replicates: int = 4
    known_active: int = 4
    known_inactive: int = 4
    negative_controls: int = 8
    uncertainty_probes: int = 8
    # evidence-lane quotas (filled after controls, in this priority order)
    consensus_high: int = 20
    mechanism_geometry_high: int = 14
    cofactor_ligand_specific: int = 10
    structural_evolutionary_high: int = 10
    # diversity fills whatever remains


@dataclass
class ControlStrategy:
    """The control candidates to reserve. Each list holds duck-typed candidates."""
    wt: List = field(default_factory=list)
    known_active: List = field(default_factory=list)
    known_inactive: List = field(default_factory=list)
    negative_computational: List = field(default_factory=list)  # stable-but-low-score probes

    @property
    def uncalibrated(self) -> bool:
        """No known active AND no known inactive -> the target cannot be calibrated."""
        return not (self.known_active or self.known_inactive)


@dataclass
class Allocation:
    selected: List
    by_lane: Dict[str, int]
    uncalibrated: bool
    dropped: int                 # candidates that qualified for a lane but didn't fit the budget
    notes: List[str] = field(default_factory=list)


def _s(c, key: str, default: float) -> float:
    v = getattr(c, "scores", {}).get(key)
    return v if isinstance(v, (int, float)) else default


def _consensus(c) -> float:
    """All-axes-good score: the MIN over normalized axis proxies (a candidate is only
    consensus-high if NO axis is weak)."""
    axes = [
        _clamp(_s(c, "stability_score", 0.0) / 2.0 + 0.5),
        _clamp(_s(c, "msa_permissiveness", 0.0)),
        _clamp(_s(c, "hbond_occupancy", 0.0)),
        _clamp(_s(c, "nac_occupancy", 0.0)),
    ]
    return min(axes)


def _clamp(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


# evidence lanes: name -> (keyfunc, quota-attr). Priority = list order.
def _lane_specs(budget: PlateBudget):
    return [
        ("consensus_high", _consensus, budget.consensus_high),
        ("mechanism_geometry_high",
         lambda c: _s(c, "nac_occupancy", 0.0) - _s(c, "catalytic_geometry_penalty", 0.0),
         budget.mechanism_geometry_high),
        ("cofactor_ligand_specific",
         lambda c: _s(c, "hbond_occupancy", 0.0) + _s(c, "interaction_gain", 0.0),
         budget.cofactor_ligand_specific),
        ("structural_evolutionary_high",
         lambda c: _s(c, "stability_score", -9.0) + _s(c, "msa_permissiveness", 0.0),
         budget.structural_evolutionary_high),
    ]


def allocate(
    candidates: Sequence, controls: Optional[ControlStrategy] = None,
    budget: Optional[PlateBudget] = None,
) -> Allocation:
    controls = controls or ControlStrategy()
    budget = budget or PlateBudget()
    selected: List = []
    selected_ids: set = set()
    by_lane: Dict[str, int] = {}
    notes: List[str] = []
    remaining = budget.total

    def _take(cands, lane: str, quota: int) -> int:
        nonlocal remaining
        n = 0
        for c in cands:
            if remaining <= 0 or n >= quota:
                break
            cid = getattr(c, "candidate_id", None)
            if cid in selected_ids:
                _record_lane(c, lane)          # already selected: still record membership
                continue
            if getattr(c, "details", None) is None:
                continue
            _record_lane(c, lane)
            c.details.setdefault("selection_lane", lane)   # primary = first (highest-priority)
            selected.append(c); selected_ids.add(cid)
            n += 1; remaining -= 1
        by_lane[lane] = by_lane.get(lane, 0) + n
        return n

    # 1) reserve controls FIRST (priority order)
    _take(controls.wt, "wt_replicates", budget.wt_replicates)
    _take(controls.known_active, "known_active", budget.known_active)
    _take(controls.known_inactive, "known_inactive", budget.known_inactive)
    _take(controls.negative_computational, "negative_control", budget.negative_controls)

    # 2) uncertainty probes — the most model-conflicted candidates
    probes = sorted(candidates, key=lambda c: _s(c, "docking_uncertainty", 0.0), reverse=True)
    _take(probes, "uncertainty_probe", budget.uncertainty_probes)

    # 3) evidence lanes (priority order), recording all_lanes membership
    for name, keyf, quota in _lane_specs(budget):
        ranked = sorted(candidates, key=keyf, reverse=True)
        # record all_lanes membership for the top-`quota` regardless of budget fit
        for c in ranked[:quota]:
            _record_lane(c, name)
        _take(ranked, name, quota)

    # 4) diversity fills the remainder
    if remaining > 0:
        _take(_by_position_diversity(candidates, selected_ids), "diversity", remaining)

    # 5) count drops: candidates that earned a lane but didn't make the budget
    qualified = {getattr(c, "candidate_id", None)
                 for c in candidates
                 if getattr(c, "details", None) and c.details.get("all_lanes")}
    dropped = len(qualified - selected_ids)
    if dropped:
        notes.append(f"{dropped} candidate(s) qualified for a lane but exceeded the "
                     f"{budget.total}-well budget (not silently hidden)")
    if controls.uncalibrated:
        notes.append("no known active/inactive controls -> target flagged UNCALIBRATED")

    return Allocation(selected=selected, by_lane=by_lane,
                      uncalibrated=controls.uncalibrated, dropped=dropped, notes=notes)


def _record_lane(c, lane: str) -> None:
    if getattr(c, "details", None) is None:
        return
    lanes = c.details.setdefault("all_lanes", [])
    if lane not in lanes:
        lanes.append(lane)


def _positions(c) -> set:
    return {getattr(m, "position", None) for m in getattr(c, "mutations", []) or []}


def _by_position_diversity(candidates, taken_ids: set) -> List:
    out, covered = [], set()
    pool = [c for c in candidates if getattr(c, "candidate_id", None) not in taken_ids]
    while pool:
        pool.sort(key=lambda c: (len(_positions(c) - covered), _s(c, "ml_score", -9.0)),
                  reverse=True)
        pick = pool.pop(0)
        covered |= _positions(pick)
        out.append(pick)
    return out
