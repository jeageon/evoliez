"""Ultra-review statistical-validity fixes for the ML subsystem.

Covers four verified bugs:

1. ``_spearman`` had no tie correction -> a constant variable scored a perfect
   1.0 (e.g. ``[5,5,5]`` vs ``[1,2,3]``). Now ties get average ranks and an
   undefined correlation (constant variable / n<3) returns ``None``.
2. ``_auroc`` returned 0.0 (the *worst* score) when a class was empty, coercing
   "unmeasurable" into "failing". Now returns ``None``.
3. ``InteractionModel`` heuristic ``scale`` collapsed to 1e-3 (a near-hard
   step) when pose labels were degenerate (``mn <= mp``). Now it falls back to
   the distance spread and warns.
4. The subfamily-holdout AUROC was CIRCULAR: it scored a held group's real
   poses against decoys synthesised from that SAME group's own median, so any
   tight cluster - even pure junk - scored ~1.0. The de-circularized
   leave-one-group-out metric scores held real poses against TRAIN-consensus
   decoys, so off-manifold junk now scores near chance / 0.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from evoliez.ml.benchmark import _auroc, _spearman
from evoliez.ml.interaction_model import InteractionModel
from evoliez.ml.pose_selection import PoseRecord, select_poses
from evoliez.stages.s06b_interaction_model import InteractionModelStage


# --------------------------------------------------------------------------- #
# 1. _spearman tie handling
# --------------------------------------------------------------------------- #
def test_spearman_constant_variable_is_none_not_one():
    # The headline bug: a constant variable used to score a perfect 1.0.
    assert _spearman([5, 5, 5], [1, 2, 3]) is None
    assert _spearman([1, 2, 3], [5, 5, 5]) is None
    # Both constant -> still undefined.
    assert _spearman([2, 2, 2], [9, 9, 9]) is None


def test_spearman_too_few_points_is_none():
    assert _spearman([1, 2], [1, 2]) is None
    assert _spearman([], []) is None


def test_spearman_perfect_and_anti_correlation():
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_spearman_average_rank_tie_correction():
    # With proper average-rank ties this is a real (non-degenerate) value, and
    # it is NOT the naive 1.0 a tie-blind ranker would produce.
    rho = _spearman([1, 1, 2, 3], [1, 2, 3, 4])
    assert rho is not None
    assert 0.0 < rho < 1.0
    # Tie-aware rho must match the textbook Pearson-on-average-ranks result.
    xs, ys = [1, 1, 2, 3], [1, 2, 3, 4]
    rx = [0.5, 0.5, 2.0, 3.0]   # average ranks for the tied pair
    ry = [0.0, 1.0, 2.0, 3.0]
    n = 4
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n))
           * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    assert abs(rho - round(num / den, 4)) < 1e-9


# --------------------------------------------------------------------------- #
# 2. _auroc empty class
# --------------------------------------------------------------------------- #
def test_auroc_empty_class_is_none_not_zero():
    # all-positive and all-negative are both unmeasurable -> None, never 0.0
    assert _auroc([0.1, 0.2, 0.3], [1, 1, 1]) is None
    assert _auroc([0.1, 0.2, 0.3], [0, 0, 0]) is None
    assert _auroc([], []) is None


def test_auroc_measurable_split_is_a_number():
    # perfect separation -> 1.0; ties counted as 0.5
    assert _auroc([0.9, 0.1], [1, 0]) == 1.0
    assert _auroc([0.5, 0.5], [1, 0]) == 0.5


# --------------------------------------------------------------------------- #
# 3. interaction_model heuristic scale (degeneracy must not collapse to a step)
# --------------------------------------------------------------------------- #
def _fit_heuristic(sel):
    m = InteractionModel(cutoff=6.0, k_nearest=6)
    m.fp_dim = int(sel.consensus.shape[0])
    m.consensus = sel.consensus.astype(float)
    m._fit_heuristic(sel)
    return m


def test_heuristic_scale_does_not_collapse_on_degenerate_labels(caplog):
    rng = np.random.RandomState(1)
    base = np.linspace(0.3, 0.8, 16)
    recs = [PoseRecord("h0", base + rng.normal(0, 0.02, 16), 1.0, 0.6, 4.0)
            for _ in range(10)]
    sel = select_poses(recs, seed=5)
    # Force label degeneracy: invert labels so the near-consensus rows become
    # the negatives and the far decoys become positives -> mn <= mp.
    sel.y[:] = 1 - sel.y
    with caplog.at_level(logging.WARNING, logger="evoliez.interaction_model"):
        m = _fit_heuristic(sel)
    # OLD formula `max(1e-3, (mn-mp)/4 or std or 1)` collapsed to 1e-3 here.
    assert m.scale > 1e-2, f"scale collapsed to a hard step: {m.scale}"
    # And the degeneracy is reported, not silently swallowed.
    assert any("degenerate pose labels" in r.message for r in caplog.records)

    # No hard step: a realistic perturbation must move the score smoothly,
    # not snap between ~0 and ~1.
    near = m.thr - 0.3
    far = m.thr + 0.3
    s_near = float(1 / (1 + np.exp(-(-(near - m.thr) / m.scale))))
    s_far = float(1 / (1 + np.exp(-(-(far - m.thr) / m.scale))))
    assert 0.02 < s_far < s_near < 0.98


def test_heuristic_scale_not_saturated_in_normal_case():
    rng = np.random.RandomState(1)
    base = np.linspace(0.3, 0.8, 16)
    recs = [PoseRecord(f"h{i % 3}", base + rng.normal(0, 0.02, 16), 1.0, 0.6,
                       4.0) for i in range(30)]
    recs += [PoseRecord(f"h{i % 3}", base + 5.0 + rng.normal(0, 0.1, 16), 1.0,
                        0.2, 0.5) for i in range(6)]
    sel = select_poses(recs, seed=3)
    m = _fit_heuristic(sel)
    near = m.score_vector(np.concatenate([base + rng.normal(0, 0.05, 16),
                                          [0, 1, 4]]))
    mid = m.score_vector(np.concatenate([base + 0.5, [0, 1, 2]]))
    far = m.score_vector(np.concatenate([base + 5.0, [0, 1, 0.5]]))
    # A mid-distance pose must land strictly between the extremes - i.e. the
    # sigmoid spans the realistic distance range instead of saturating.
    assert far < mid < near
    assert mid > 0.05  # not pinned to ~0 by an over-steep scale


# --------------------------------------------------------------------------- #
# 4. De-circularized subfamily holdout (THE key proof)
# --------------------------------------------------------------------------- #
_CFG = SimpleNamespace(subfamily_holdout=True, contact_cutoff=6.0,
                       k_nearest_residues=6)
_SEL_KW = dict(select_z=2.5, outlier_z=4.0, min_decoys_per_group=4, seed=1234,
               keep_alternative_band=True, alternative_weight=0.3,
               hard_decoys_per_group=2)


def _old_circular_auroc(real, held):
    """The OLD (buggy) metric: held real poses vs decoys from the held group's
    OWN median. Reproduced here only to contrast with the fixed metric."""
    m = InteractionModel(cutoff=6.0, k_nearest=6).fit(
        select_poses(real, **_SEL_KW))
    held_sel = select_poses(held, **{**_SEL_KW, "seed": _SEL_KW["seed"] + 1})
    scores = [m.score_vector(x) for x in held_sel.X]
    return _auroc(scores, [int(v) for v in held_sel.y.tolist()])


def _make_family(rng, n_groups=3, per=12, base=None):
    base = np.linspace(0.3, 0.8, 16) if base is None else base
    return [PoseRecord(f"h{g}", base + rng.normal(0, 0.02, 16), 1.0, 0.6, 4.0)
            for g in range(n_groups) for _ in range(per)]


def test_decircularized_holdout_rejects_offmanifold_junk():
    """HEADLINE: a held group of pure junk sitting OFF the training manifold
    used to score ~1.0 (real-junk beats perturbations of itself). The
    de-circularized metric scores it against TRAIN-consensus decoys, so it now
    scores near chance / 0 - the honest answer."""
    rng = np.random.RandomState(7)
    base = np.linspace(0.3, 0.8, 16)
    real = _make_family(rng, base=base)
    # Tight junk cluster displaced off the family manifold.
    center = base + 1.0
    junk = [PoseRecord("JUNK", center + rng.normal(0, 0.02, 16), 1.0, 0.5, 2.0)
            for _ in range(24)]

    old = _old_circular_auroc(real, junk)
    gen = InteractionModelStage()._consensus_generalization(
        real + junk, _SEL_KW, _CFG)
    new_junk = gen["per_group"]["JUNK"]

    # The bug: OLD metric is high (false confidence) for off-manifold junk.
    assert old >= 0.85, f"expected the circular metric to be high, got {old}"
    # The fix: held junk no longer scores high; it is at/below chance.
    assert new_junk <= 0.5, f"de-circularized junk AUROC too high: {new_junk}"
    # And it is strictly, materially lower than the circular score.
    assert new_junk < old - 0.3


def test_decircularized_holdout_uses_train_decoys_as_negatives():
    """Sanity: the metric loops over ALL groups and averages, reporting a
    spread; a genuinely on-manifold family still generalizes well."""
    rng = np.random.RandomState(3)
    base = np.linspace(0.3, 0.8, 16)
    real = _make_family(rng, n_groups=4, per=12, base=base)
    gen = InteractionModelStage()._consensus_generalization(
        real, _SEL_KW, _CFG)
    assert gen is not None
    assert gen["n_groups"] == 4              # leave-one-group-out over ALL
    assert set(gen["per_group"]) == {"h0", "h1", "h2", "h3"}
    assert "mean" in gen and "std" in gen
    # On-manifold consensus poses are well-separated from train decoys.
    assert gen["mean"] >= 0.8


def test_decircularized_holdout_needs_three_groups():
    rng = np.random.RandomState(0)
    two = _make_family(rng, n_groups=2, per=10)
    assert InteractionModelStage()._consensus_generalization(
        two, _SEL_KW, _CFG) is None


def test_holdout_disabled_returns_none():
    rng = np.random.RandomState(0)
    real = _make_family(rng, n_groups=3, per=10)
    cfg = SimpleNamespace(subfamily_holdout=False, contact_cutoff=6.0,
                          k_nearest_residues=6)
    assert InteractionModelStage()._consensus_generalization(
        real, _SEL_KW, cfg) is None
