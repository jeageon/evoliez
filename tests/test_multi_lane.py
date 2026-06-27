"""Unit tests for multi-lane candidate selection."""
from dataclasses import dataclass, field
from typing import List

from evoliez.ranking.multi_lane import LaneConfig, lane_counts, select_multi_lane


@dataclass
class _Mut:
    position: int


@dataclass
class _Cand:
    candidate_id: str
    scores: dict
    mutations: List[_Mut] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def _make(n):
    out = []
    for i in range(n):
        out.append(_Cand(
            f"c{i:02d}",
            {"ml_score": i / n, "stability_score": (n - i) / n,
             "catalytic_geometry_penalty": float(i % 5)},
            [_Mut(100 + (i % 7))]))
    return out


def test_ml_high_and_low_control_lanes():
    cands = _make(30)
    cfg = LaneConfig(from_ml_high=5, from_stability_high=0, from_geometry_high=0,
                     from_diversity=0, low_ml_controls=3)
    sel = select_multi_lane(cands, cfg)
    lane = {c.candidate_id: c.details["selection_lane"] for c in sel}
    assert all(lane[c] == "ml_high" for c in ["c29", "c28", "c27", "c26", "c25"])
    assert all(lane[c] == "low_ml_control" for c in ["c00", "c01", "c02"])


def test_stability_lane_distinct_from_ml():
    cands = _make(30)
    cfg = LaneConfig(from_ml_high=5, from_stability_high=5, from_geometry_high=0,
                     from_diversity=0, low_ml_controls=0)
    sel = select_multi_lane(cands, cfg)
    lane = {c.candidate_id: c.details["selection_lane"] for c in sel}
    # stability_score = (n-i)/n -> highest at c00; but c00.. not already in ml_high
    assert lane.get("c00") == "stability_high"


def test_no_duplicates_across_lanes():
    cands = _make(40)
    cfg = LaneConfig(from_ml_high=10, from_stability_high=10, from_geometry_high=10,
                     from_diversity=5, low_ml_controls=5)
    sel = select_multi_lane(cands, cfg)
    ids = [c.candidate_id for c in sel]
    assert len(ids) == len(set(ids))


def test_diversity_increases_position_coverage():
    cands = _make(40)
    cfg = LaneConfig(from_ml_high=0, from_stability_high=0, from_geometry_high=0,
                     from_diversity=7, low_ml_controls=0)
    sel = select_multi_lane(cands, cfg)
    covered = set()
    for c in sel:
        covered |= {m.position for m in c.mutations}
    assert len(covered) == 7          # 7 distinct positions exist; diversity finds them


def test_lane_counts_sum():
    cands = _make(40)
    cfg = LaneConfig(from_ml_high=8, from_stability_high=6, from_geometry_high=6,
                     from_diversity=4, low_ml_controls=4)
    sel = select_multi_lane(cands, cfg)
    assert sum(lane_counts(sel).values()) == len(sel)
