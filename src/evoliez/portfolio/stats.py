"""V7 statistical primitives (ROADMAP_V7 Phase V7-2) — pure numpy, no scipy.

A score becomes *selectable* only after it is calibrated against a null distribution
(ROADMAP_V7 §4). This module is the calibration engine: empirical (permutation-style)
p-values, Benjamini-Hochberg FDR q-values, standardized/population effect sizes, and
bootstrap confidence intervals. Everything is deterministic given a seed and depends only
on numpy (scipy is not in the light env), so bands are reproducible and testable locally.

Conventions:
  * Higher-is-better axes use ``empirical_p_greater`` (P(null >= observed)); higher-is-worse
    axes (uncertainty) flip the sign before calling, so a *low* uncertainty score is the
    "significant" tail.
  * Empirical p-values use the (1 + count)/(1 + n) smoothing (Phipson & Smyth 2010) so a
    p-value is never exactly 0 for a finite null — a permutation p of 0 is a lie.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

_ERFC = np.vectorize(math.erfc, otypes=[float])   # vectorized complementary error fn (no scipy)


def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def empirical_p_greater(observed: float, null: np.ndarray) -> float:
    """One-sided empirical p-value P(null >= observed) with add-one smoothing.

    ``p = (1 + #{null >= observed}) / (1 + n)`` — bounded in ``(0, 1]``, never exactly 0 for
    a finite null (Phipson-Smyth). Empty/degenerate null -> 1.0 (no evidence of a tail)."""
    if not np.isfinite(observed):
        return 1.0
    n = _as_1d(null)
    if n.size == 0:
        return 1.0
    ge = int(np.count_nonzero(n >= observed))
    return (1.0 + ge) / (1.0 + n.size)


def empirical_p_values(observed, null, *, higher_is_worse: bool = False) -> np.ndarray:
    """Vector of one-sided empirical p-values for each observed score against a shared null.

    ``higher_is_worse`` flips the orientation (uncertainty axis): the significant tail is
    the LOW end, so we test P(null <= observed) via negation. NaN observations -> p = 1.0."""
    obs = np.asarray(observed, dtype=float).ravel()
    nul = _as_1d(null)
    if higher_is_worse:
        obs = -obs
        nul = -nul
    return np.array([empirical_p_greater(float(o), nul) for o in obs])


def robust_normal_p_values(observed, null, *, higher_is_worse: bool = False,
                           min_null: int = 6) -> np.ndarray:
    """One-sided p-values against a ROBUST-GAUSSIAN empirical null (Efron's empirical-null
    idea). The count-based ``empirical_p_greater`` has a p-floor of ``1/(n+1)``; over a
    same-size candidate family that floor makes Benjamini-Hochberg mathematically unable to
    reach significance (q_min ~= 2), so a genuine outlier can never be banded. Instead we fit
    the null's CENTRE (median) and SCALE (1.4826*MAD, std fallback) — both resistant to a
    minority signal tail, so the null is estimated from the bulk — then take the analytic
    normal tail ``P(Z >= z)`` via ``math.erfc``. This gives continuous-resolution p-values, so
    a real 3-4 sigma outlier survives FDR while the bulk does not.

    ``higher_is_worse`` (uncertainty axis) flips the tail: the beneficial end is low scores.
    A tiny (< ``min_null``) or zero-spread null falls back to the count-based empirical p."""
    obs = np.asarray(observed, dtype=float).ravel()
    nul = _as_1d(null)
    if nul.size < min_null:
        return empirical_p_values(obs, nul, higher_is_worse=higher_is_worse)
    center = float(np.median(nul))
    mad = float(np.median(np.abs(nul - center))) * 1.4826
    scale = mad if mad > 1e-9 else float(np.std(nul))
    if scale <= 1e-9:
        return empirical_p_values(obs, nul, higher_is_worse=higher_is_worse)
    z = (center - obs) / scale if higher_is_worse else (obs - center) / scale
    with np.errstate(over="ignore"):
        p = 0.5 * _ERFC(z / math.sqrt(2.0))
    # NaN obs -> p = 1 (no evidence); clamp to a finite floor so BH q stays meaningful
    p = np.where(np.isfinite(p), p, 1.0)
    return np.clip(p, 1e-12, 1.0)


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values for a family of p-values.

    Standard step-up: sort ascending, ``q_(i) = p_(i) * m / i``, enforce monotonicity from
    the largest rank down, clip to 1. NaN p-values pass through as NaN (an axis with no
    p-value stays uncalibrated). Returns q-values aligned to the INPUT order."""
    p = np.asarray(pvals, dtype=float).ravel()
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    m = int(np.count_nonzero(finite))
    if m == 0:
        return out
    idx = np.flatnonzero(finite)
    pv = p[idx]
    order = np.argsort(pv, kind="mergesort")          # stable
    ranked = pv[order]
    ranks = np.arange(1, m + 1, dtype=float)
    q = ranked * m / ranks
    # enforce monotone non-decreasing q along ascending p (cumulative min from the top)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    q_in_finite_order = np.empty(m)
    q_in_finite_order[order] = q
    out[idx] = q_in_finite_order
    return out


def population_effect_size(observed: float, population) -> float:
    """Robust standardized effect size of ``observed`` vs a population: median-centred,
    scaled by a normal-consistent MAD (1.4826 * MAD). Falls back to std when MAD == 0.
    NaN / empty / zero-spread -> 0.0 (no measurable effect)."""
    pop = _as_1d(population)
    if pop.size == 0 or not np.isfinite(observed):
        return 0.0
    med = float(np.median(pop))
    mad = float(np.median(np.abs(pop - med))) * 1.4826
    scale = mad if mad > 1e-12 else float(np.std(pop))
    if scale <= 1e-12:
        return 0.0
    return float((observed - med) / scale)


def cohens_d(a, b) -> float:
    """Cohen's d between two samples (pooled-SD standardized mean difference). Degenerate
    (empty group or zero pooled variance) -> 0.0."""
    a = _as_1d(a)
    b = _as_1d(b)
    if a.size < 2 or b.size < 2:
        return 0.0
    na, nb = a.size, b.size
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    s = float(np.sqrt(pooled))
    if s <= 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / s)


def bootstrap_ci(values, *, statistic: Callable[[np.ndarray], float] = np.mean,
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``values``. Deterministic given
    ``seed``. Empty -> (nan, nan); single value -> (v, v)."""
    v = _as_1d(values)
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    stats = np.array([float(statistic(v[row])) for row in idx])
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return (lo, hi)


def permutation_null(scores, labels, *,
                     statistic: Callable[[np.ndarray, np.ndarray], float],
                     n_perm: int = 2000, seed: int = 0) -> np.ndarray:
    """Permutation null for a group-difference ``statistic(group1, group0)``: shuffle the
    boolean ``labels`` ``n_perm`` times and recompute. Deterministic given ``seed``.
    Returns the null distribution (length ``n_perm``); degenerate input -> empty array."""
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if s.size != y.size or s.size == 0 or y.all() or (~y).all():
        return np.array([])
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(y)
        out[i] = float(statistic(s[perm], s[~perm]))
    return out


def enrichment_at_k(scores, hits, k: int) -> Optional[float]:
    """Top-k enrichment = (hit rate in the top-k by score) / (overall hit rate). None when
    k exceeds n or the base rate is 0. (Mirrors ml.prior_eval.topk_enrichment; kept here so
    the portfolio calibration layer has no cross-package dependency.)"""
    s = np.asarray(scores, dtype=float).ravel()
    h = np.asarray(hits).ravel().astype(bool)
    n = s.size
    if n == 0 or k <= 0 or k > n or h.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    topk_rate = float(h[order[:k]].mean())
    base = float(h.mean())
    if base <= 0:
        return None
    return topk_rate / base
