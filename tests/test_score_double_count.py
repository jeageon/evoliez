"""family_interaction_score must enter the final score exactly ONCE, with the
documented ScoreWeights.family_interaction weight (ultra-review G2-20).

It was double-counted: the heuristic reranker folded 1.5x family_interaction
into ml_score (-> ml_mutation contribution) AND score.py added a dedicated
family_interaction contribution, giving an effective ~2.5x weight instead of
the documented 1.0 - breaking the transparent additive decomposition.
"""

from evoliez.config import ScoreWeights
from evoliez.ranking.score import compute_final_score
from evoliez.stages.s08_reranker import RerankerStage
from evoliez.types import Candidate

_BASE_FEATURES = {
    "interaction_gain": 0.3,
    "msa_permissiveness": 0.2,
    "conservation": 0.4,
    "n_mutations": 1,
    "dist_to_ligand": 5.0,
}


def _candidate(fam: float) -> Candidate:
    c = Candidate(candidate_id=f"c{fam}", mutations=[], generator="g")
    c.details["features"] = {**_BASE_FEATURES, "family_interaction_score": fam}
    c.scores["family_interaction_score"] = fam
    return c


def test_ml_score_does_not_absorb_family_interaction():
    """The heuristic ml_score must not depend on family_interaction_score (it
    has its own dedicated contribution)."""
    lo, hi = _candidate(0.2), _candidate(0.8)
    RerankerStage._score_heuristic(None, [lo, hi])  # self unused
    assert lo.scores["ml_score"] == hi.scores["ml_score"]


def test_family_interaction_effective_weight_equals_documented():
    """End-to-end: varying family_interaction_score changes the final score by
    exactly w.family_interaction * delta (1.0x), not ~2.5x."""
    w = ScoreWeights()
    lo, hi = _candidate(0.2), _candidate(0.8)
    RerankerStage._score_heuristic(None, [lo, hi])
    blo = compute_final_score(lo, w)
    bhi = compute_final_score(hi, w)
    assert abs((bhi.total - blo.total) - w.family_interaction * (0.8 - 0.2)) < 1e-6
