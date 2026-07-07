"""Per-term sensitivity of the soft reaction-geometry score (ROADMAP_V4 §7.4).

The soft geometry score is a product of five kernels, so a single scalar hides *why* a
candidate scores the way it does. This module reports the finite-difference partial of
``soft_reaction_geometry_score`` with respect to each geometric input, plus which input a
candidate's score is most sensitive to. It lets a report say "this candidate's geometry
support is dominated by the donor-acceptor angle" instead of quoting one opaque number,
and it is the ``geometry_sensitivity`` build target the roadmap lists next to the soft
kernel. The gradient machinery reuses the numpy kernels in
:mod:`evoliez.guided.soft_kernels` (no torch dependency).
"""

from __future__ import annotations

from typing import Dict

from evoliez.guided.soft_kernels import soft_reaction_geometry_score

# geometric inputs we differentiate against, with a per-input central-difference step
_STEPS: Dict[str, float] = {
    "distance_A": 1e-3,
    "angle_deg": 1e-2,
    "contact_score": 1e-3,
    "strain_penalty": 1e-3,
}


def geometry_sensitivity(
    *,
    distance_A: float,
    angle_deg: float,
    contact_score: float,
    strain_penalty: float,
    **ensemble: float,
) -> Dict[str, float]:
    """Central-difference partial derivatives of the soft geometry score.

    Returns a mapping ``input -> d(score)/d(input)``. A positive value means the score
    rises as that input increases at the given operating point; the sign therefore tells
    you which direction each geometric feature would have to move to improve support.
    Extra keyword arguments (ensemble distribution parameters) are forwarded unchanged to
    :func:`soft_reaction_geometry_score`.
    """
    base = dict(
        distance_A=distance_A,
        angle_deg=angle_deg,
        contact_score=contact_score,
        strain_penalty=strain_penalty,
        **ensemble,
    )
    out: Dict[str, float] = {}
    for name, eps in _STEPS.items():
        plus = dict(base)
        minus = dict(base)
        plus[name] = base[name] + eps
        minus[name] = base[name] - eps
        derivative = (
            soft_reaction_geometry_score(**plus) - soft_reaction_geometry_score(**minus)
        ) / (2.0 * eps)
        out[name] = round(derivative, 6)
    return out


def dominant_term(sensitivity: Dict[str, float]) -> str:
    """The input the score is most sensitive to (largest absolute partial derivative)."""
    if not sensitivity:
        raise ValueError("sensitivity mapping is empty")
    return max(sensitivity, key=lambda k: abs(sensitivity[k]))
