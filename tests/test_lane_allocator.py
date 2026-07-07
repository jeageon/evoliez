"""V3-6: plate-budget lane allocator + control strategy (ROADMAP_V3 D8)."""
from dataclasses import dataclass, field
from typing import Dict, List

from evoliez.ranking.lane_allocator import (
    Allocation, ControlStrategy, PlateBudget, allocate,
)


@dataclass
class FakeCand:
    candidate_id: str
    scores: Dict[str, float] = field(default_factory=dict)
    mutations: List = field(default_factory=list)
    details: Dict = field(default_factory=dict)


@dataclass
class FakeMut:
    position: int


def _cands(n, **score_ramp):
    out = []
    for i in range(n):
        sc = {k: (v if not callable(v) else v(i)) for k, v in score_ramp.items()}
        out.append(FakeCand(candidate_id=f"c{i}", scores=sc, mutations=[FakeMut(i % 7)]))
    return out


def test_never_exceeds_budget():
    cands = _cands(300, nac_occupancy=lambda i: (i % 100) / 100.0,
                   stability_score=lambda i: (i % 50) / 25.0,
                   hbond_occupancy=lambda i: (i % 80) / 80.0,
                   msa_permissiveness=lambda i: (i % 60) / 60.0,
                   docking_uncertainty=lambda i: (i % 30) / 30.0)
    alloc = allocate(cands, budget=PlateBudget(total=96))
    assert len(alloc.selected) <= 96
    assert len({c.candidate_id for c in alloc.selected}) == len(alloc.selected)  # deduped


def test_controls_reserved_first_and_uncalibrated_flag():
    cands = _cands(200, nac_occupancy=lambda i: i / 200.0)
    # no known controls -> uncalibrated
    ctrl = ControlStrategy(wt=[FakeCand("WT")], negative_computational=[FakeCand("neg0")])
    alloc = allocate(cands, controls=ctrl, budget=PlateBudget(total=96))
    ids = {c.candidate_id for c in alloc.selected}
    assert "WT" in ids and "neg0" in ids
    assert alloc.uncalibrated is True
    assert any("UNCALIBRATED" in n for n in alloc.notes)


def test_known_controls_clear_uncalibrated():
    ctrl = ControlStrategy(wt=[FakeCand("WT")], known_active=[FakeCand("act")],
                           known_inactive=[FakeCand("ina")])
    alloc = allocate(_cands(50, nac_occupancy=lambda i: i / 50.0), controls=ctrl)
    assert alloc.uncalibrated is False
    ids = {c.candidate_id for c in alloc.selected}
    assert {"WT", "act", "ina"} <= ids


def test_all_lanes_recorded_for_multi_lane_candidate():
    # one standout candidate strong on every axis should appear in several lanes
    star = FakeCand("star", scores=dict(nac_occupancy=1.0, stability_score=4.0,
                                        hbond_occupancy=1.0, msa_permissiveness=1.0,
                                        interaction_gain=1.0), mutations=[FakeMut(1)])
    others = _cands(40, nac_occupancy=lambda i: 0.1, stability_score=lambda i: 0.0)
    alloc = allocate([star] + others, budget=PlateBudget(total=96))
    assert len(star.details["all_lanes"]) >= 2          # in multiple lanes
    assert "selection_lane" in star.details             # a single primary lane assigned
    assert star.details["selection_lane"] in star.details["all_lanes"]


def test_dropped_reported_not_silent():
    # DISJOINT lane winners (each group strong in exactly one lane) so the lane union
    # genuinely exceeds a tiny budget -> some qualified candidates are dropped + reported.
    cands = []
    for i in range(6):  # geometry-only winners
        cands.append(FakeCand(f"geo{i}", scores=dict(nac_occupancy=1.0), mutations=[FakeMut(i)]))
    for i in range(6):  # cofactor-only winners
        cands.append(FakeCand(f"cof{i}", scores=dict(hbond_occupancy=1.0), mutations=[FakeMut(i)]))
    for i in range(6):  # structural-only winners
        cands.append(FakeCand(f"str{i}", scores=dict(stability_score=4.0), mutations=[FakeMut(i)]))
    alloc = allocate(cands, budget=PlateBudget(
        total=6, wt_replicates=0, known_active=0, known_inactive=0,
        negative_controls=0, uncertainty_probes=0,
        consensus_high=0, mechanism_geometry_high=5, cofactor_ligand_specific=5,
        structural_evolutionary_high=5))
    assert len(alloc.selected) <= 6
    assert alloc.dropped > 0
    assert any("budget" in n for n in alloc.notes)
