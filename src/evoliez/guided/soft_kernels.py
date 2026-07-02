"""Soft reaction-geometry kernels for v4.

These functions replace binary NAC pass/fail language with smooth support and
uncertainty summaries. Low score is low support, not an inactivity call.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def gaussian_kernel(value: float, center: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-0.5 * ((value - center) / sigma) ** 2)


def distance_kernel(distance_A: float, *, center_A: float = 3.25, sigma_A: float = 0.65) -> float:
    return gaussian_kernel(distance_A, center_A, sigma_A)


def angle_kernel(angle_deg: float, *, center_deg: float = 165.0, sigma_deg: float = 25.0) -> float:
    return gaussian_kernel(angle_deg, center_deg, sigma_deg)


def contact_kernel(contact_score: float) -> float:
    return clamp01(contact_score)


def strain_kernel(strain_penalty: float) -> float:
    return math.exp(-max(0.0, strain_penalty))


def ensemble_fit_kernel(value: float, *, mean: float, p10: float, p90: float) -> float:
    spread = max((p90 - p10) / 2.56, 0.35)
    return gaussian_kernel(value, mean, spread)


def soft_reaction_geometry_score(
    *,
    distance_A: float,
    angle_deg: float,
    contact_score: float,
    strain_penalty: float,
    ensemble_distance_mean: float = 3.25,
    ensemble_distance_p10: float = 2.8,
    ensemble_distance_p90: float = 3.8,
) -> float:
    score = (
        distance_kernel(distance_A)
        * angle_kernel(angle_deg)
        * contact_kernel(contact_score)
        * strain_kernel(strain_penalty)
        * ensemble_fit_kernel(
            distance_A,
            mean=ensemble_distance_mean,
            p10=ensemble_distance_p10,
            p90=ensemble_distance_p90,
        )
    )
    return round(clamp01(score), 6)


def finite_difference_distance_gradient(distance_A: float, **kwargs: float) -> float:
    eps = 1e-4
    plus = soft_reaction_geometry_score(distance_A=distance_A + eps, **kwargs)
    minus = soft_reaction_geometry_score(distance_A=distance_A - eps, **kwargs)
    return (plus - minus) / (2.0 * eps)


def summarize_geometry_distribution(scores: Iterable[float], *, threshold: float = 0.25) -> Dict[str, float]:
    vals = sorted(float(s) for s in scores)
    if not vals:
        return {
            "mean_score": 0.0,
            "median_score": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "fraction_promising": 0.0,
        }
    n = len(vals)

    def pct(q: float) -> float:
        idx = (n - 1) * q
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return vals[lo] * (1.0 - frac) + vals[hi] * frac

    return {
        "mean_score": round(sum(vals) / n, 6),
        "median_score": round(pct(0.5), 6),
        "p10": round(pct(0.1), 6),
        "p90": round(pct(0.9), 6),
        "fraction_promising": round(sum(1 for v in vals if v >= threshold) / n, 6),
    }


def cutoff_sensitivity(scores: Iterable[float], cutoffs: Iterable[float] = (0.15, 0.25, 0.35)) -> Dict[str, float]:
    vals: List[float] = [float(s) for s in scores]
    if not vals:
        return {str(c): 0.0 for c in cutoffs}
    return {str(c): round(sum(1 for v in vals if v >= c) / len(vals), 6) for c in cutoffs}
