"""V4 uncertainty helpers."""

from __future__ import annotations


def uncertainty_label(score: float) -> str:
    if score >= 0.7:
        return "high_uncertainty"
    if score >= 0.45:
        return "moderate_uncertainty"
    return "low_uncertainty"
