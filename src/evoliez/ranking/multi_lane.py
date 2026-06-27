"""Multi-lane candidate selection — ML is a PRIOR, not a hard filter.

A single ML cut (s08 top_for_redocking, s09 _md_key) silently discards candidates
ML scores low — but ML enriches BINDING-validity, not catalysis (the FDH run:
AUC ml->MD-pass 0.83 vs ml->NAC 0.357; the only NAC-positive lead was not the top
ML pick). So select a UNION of lanes, each surfacing a different signal, so an ML
false negative in one lane is caught by another:

  - ml_high        : top by GNN ml_score (cheap binding-validity prior)
  - stability_high : top by ThermoMPNN stability_score (folds/keeps the fold)
  - geometry_high  : best catalytic geometry (low catalytic_geometry_penalty)
  - diversity      : maximises mutation-position coverage (explore the design space)
  - low_ml_control : a few LOW-ml candidates — the explicit false-negative probe;
    if any later passes MD/NAC, the ML hard-cutoff is demonstrably unsafe.

Each candidate is tagged with the lane that surfaced it (details["selection_lane"]).
Pure / duck-typed on (candidate_id, scores, mutations, details); unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence


@dataclass
class LaneConfig:
    enabled: bool = False
    from_ml_high: int = 25
    from_stability_high: int = 12
    from_geometry_high: int = 12
    from_diversity: int = 8
    low_ml_controls: int = 8


def _score(c, key: str, default: float) -> float:
    v = getattr(c, "scores", {}).get(key)
    return v if isinstance(v, (int, float)) else default


def _topk(cands: Sequence, key: Callable, k: int, taken: set) -> List:
    out = []
    for c in sorted(cands, key=key, reverse=True):
        if k <= 0:
            break
        if c.candidate_id in taken:
            continue
        taken.add(c.candidate_id)
        out.append(c)
        if len(out) >= k:
            break
    return out


def _positions(c) -> set:
    return {getattr(m, "position", None) for m in getattr(c, "mutations", []) or []}


def _diverse(cands: Sequence, k: int, taken: set) -> List:
    """Greedy max-coverage of mutation positions: each pick adds the most new
    positions (ties -> higher ml_score), so the lane explores breadth."""
    out: List = []
    covered: set = set()
    pool = [c for c in cands if c.candidate_id not in taken]
    while pool and len(out) < k:
        pool.sort(key=lambda c: (len(_positions(c) - covered),
                                 _score(c, "ml_score", -9.0)), reverse=True)
        pick = pool.pop(0)
        taken.add(pick.candidate_id)
        covered |= _positions(pick)
        out.append(pick)
    return out


def select_multi_lane(candidates: Sequence, cfg: LaneConfig) -> List:
    """Return the union of the lanes, tagging each candidate's
    details['selection_lane'] with the lane that surfaced it (first wins).
    Order: ml_high, stability_high, geometry_high, diversity, low_ml_control."""
    taken: set = set()
    selected: List = []

    def lane(name: str, picks: List):
        for c in picks:
            if not hasattr(c, "details") or c.details is None:
                continue
            c.details.setdefault("selection_lane", name)
            selected.append(c)

    lane("ml_high", _topk(candidates, lambda c: _score(c, "ml_score", -9.0),
                          cfg.from_ml_high, taken))
    lane("stability_high",
         _topk(candidates, lambda c: _score(c, "stability_score", -9.0),
               cfg.from_stability_high, taken))
    lane("geometry_high",
         _topk(candidates,
               lambda c: -_score(c, "catalytic_geometry_penalty", 9.0),
               cfg.from_geometry_high, taken))
    lane("diversity", _diverse(candidates, cfg.from_diversity, taken))
    lane("low_ml_control",
         _topk(candidates, lambda c: -_score(c, "ml_score", 9.0),
               cfg.low_ml_controls, taken))
    return selected


def lane_counts(selected: Sequence) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for c in selected:
        ln = getattr(c, "details", {}).get("selection_lane", "?")
        out[ln] = out.get(ln, 0) + 1
    return out
