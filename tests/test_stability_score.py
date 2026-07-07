"""stability_score is no longer dead (expert review P2)."""

import inspect

from evoliez.config import ScoreWeights
from evoliez.ranking.score import compute_final_score
from evoliez.stages import s09_nonmd_validation
from evoliez.types import Candidate, Mutation


def test_s09_sets_stability_score():
    src = inspect.getsource(s09_nonmd_validation.NonMDValidationStage.run)
    assert 'cand.scores["stability_score"]' in src


def test_stability_score_contributes_positively():
    w = ScoreWeights()
    c = Candidate("c", [Mutation("A", 10, "K")], "g")
    c.scores.update({"stability_score": 1.0, "ddg_fold": 0.0})
    bd = compute_final_score(c, w)
    assert bd.contributions["stability"] == round(w.stability * 1.0, 4)
    assert bd.contributions["stability"] > 0.0

    # destabilising mutant: no positive stability, ddG penalty applies
    c2 = Candidate("c2", [Mutation("A", 10, "K")], "g")
    c2.scores.update({"stability_score": 0.0, "ddg_fold": 3.0})
    bd2 = compute_final_score(c2, w)
    assert bd2.contributions["stability"] == 0.0
    assert bd2.penalties["ddg_stability"] > 0.0
