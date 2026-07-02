"""Differentiable-friendly catalytic geometry loss facade.

The current implementation returns the same scalar as the numpy evaluator and
keeps the API ready for a later torch-backed feasibility probe.
"""

from __future__ import annotations

from evoliez.guided.soft_kernels import soft_reaction_geometry_score


def catalytic_geometry_loss(**kwargs: float) -> float:
    return 1.0 - soft_reaction_geometry_score(**kwargs)
