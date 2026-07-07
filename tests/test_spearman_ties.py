"""Spearman benchmark metric must be tie-corrected and order-invariant
(ultra-review G2-6). The headline spearman_vs_activity is what a user reads to
judge whether the model tracks measured activity; ordinal ranks made it both
numerically wrong under ties and dependent on input row order."""

from evoliez.ml.benchmark import _spearman


def test_spearman_no_ties_is_exact():
    assert abs(_spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-9
    assert abs(_spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-9


def test_spearman_is_tie_corrected():
    # pairs (5,9)(5,1)(1,2)(2,3)(3,4): the two x=5 rows are tied.
    # scipy.stats.spearmanr gives 0.2052; ordinal ranks gave 0.0.
    assert abs(_spearman([5, 5, 1, 2, 3], [9, 1, 2, 3, 4]) - 0.2052) < 1e-3


def test_spearman_is_order_invariant_under_ties():
    # swapping the two tied-x rows (their y values) must not change the result;
    # the old ordinal-rank version returned 0.0 vs 0.4 for these two orderings.
    a = _spearman([5, 5, 1, 2, 3], [9, 1, 2, 3, 4])
    b = _spearman([5, 5, 1, 2, 3], [1, 9, 2, 3, 4])
    assert abs(a - b) < 1e-9
