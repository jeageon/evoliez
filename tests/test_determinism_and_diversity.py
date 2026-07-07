"""Determinism + diversity-library ordering guards (ultra-review #18/#43/#31)."""

import pytest

from evoliez.ml.active_learning import acquisition_score, select_focused_library
from evoliez.stages.s11_final_ranking import _ranked
from evoliez.types import Candidate, Mutation


def _ac(cid, pos, score):
    c = Candidate(candidate_id=cid, mutations=[Mutation("A", pos, "L")],
                  generator="g")
    c.scores["final_score"] = score
    return c


def test_focused_library_round_robin_preserves_cluster_order():
    """3 clusters (by position//10) sized 1/3/3, library size 5: round-robin
    must serve one per cluster in score order, then refill the higher-scored
    clusters - NOT scramble when a bucket empties mid-round."""
    ranked = [
        _ac("A1", 5, 0.9),
        _ac("B1", 15, 0.8), _ac("B2", 15, 0.5), _ac("B3", 15, 0.2),
        _ac("C1", 25, 0.7), _ac("C2", 25, 0.4), _ac("C3", 25, 0.1),
    ]
    lib = select_focused_library(ranked, size=5, beta=0.0)
    assert [c.candidate_id for c in lib] == ["A1", "B1", "C1", "B2", "C2"]


def test_acquisition_score_has_no_inert_gamma_knob():
    """al_gamma multiplied an always-0.0 diversity term, so the knob did
    nothing (diversity is enforced structurally by the cluster round-robin).
    The inert gamma/diversity param is removed."""
    c = Candidate("x", [Mutation("A", 10, "L")], "g")
    c.scores["final_score"] = 1.0
    with pytest.raises(TypeError):
        acquisition_score(c, gamma=10.0)


def _fc(cid, score):
    c = Candidate(candidate_id=cid, mutations=[], generator="g")
    c.scores["final_score"] = score
    return c


def test_final_ranking_tie_breaker_is_deterministic():
    """Equal final_score must resolve by candidate_id, so rank numbers (which
    drive focused-library well assignment) are invariant to upstream order."""
    a, b, c = _fc("z", 0.5), _fc("a", 0.5), _fc("m", 0.7)
    r1 = [x.candidate_id for x in _ranked([a, b, c])]
    r2 = [x.candidate_id for x in _ranked([b, a, c])]   # different input order
    assert r1 == r2 == ["m", "a", "z"]
