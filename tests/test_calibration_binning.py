"""reliability_diagram must bin the prob==0 point that ECE counts
(ultra-review #34). In calibration_curve the min-max normalization pins the
lowest-scoring candidate to prob exactly 0; ECE includes it (bin 0) but the
diagram dropped it (lo < p <= hi), so the two covered different populations and
per-bin counts didn't sum to n."""

from evoliez.ml.benchmark import calibration_curve
from evoliez.ml.calibration import reliability_diagram
from evoliez.types import Candidate, Mutation


def test_reliability_diagram_includes_prob_zero_point():
    probs = [0.0, 0.05, 0.5, 0.95]
    labels = [0, 0, 1, 1]
    diag = reliability_diagram(probs, labels)
    assert sum(d["n"] for d in diag) == len(probs)   # every point binned
    bin0 = [d for d in diag if d["bin"] == "0.0-0.1"]
    assert bin0 and bin0[0]["n"] == 2                # 0.0 and 0.05 in bin 0


def test_calibration_curve_is_tagged_rank_score_reliability():
    """The min-max-normalized 'ece' is rank-score reliability, not true
    probabilistic calibration; the output must say so."""
    def _c(mut, score):
        c = Candidate(f"c{mut}", [Mutation("A", mut, "L")], "g")
        c.scores["final_score"] = score
        return c

    ranked = [_c(10, 0.9), _c(11, 0.5), _c(12, 0.1), _c(13, 0.3)]
    bench = [{"mutation": "A10L", "label": "beneficial"},
             {"mutation": "A11L", "label": "neutral"},
             {"mutation": "A12L", "label": "neutral"},
             {"mutation": "A13L", "label": "beneficial"}]
    out = calibration_curve(ranked, bench)
    assert out["metric"] == "rank_score_reliability"
