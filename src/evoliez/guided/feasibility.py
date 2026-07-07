"""V4 guided-Boltz feasibility gate.

This module makes the optional guided branch explicit. It returns GO only when
the required atom map, sampler access, gradient sanity, resource budget, and
negative controls all pass. Otherwise v4 stays on ReferenceEnsemble v0.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from evoliez.provenance.atom_map import ReactiveAtomMap


class FeasibilityConfig(BaseModel):
    cases: List[str] = Field(
        default_factory=lambda: [
            "WT",
            "mut_00479",
            "Q382R",
            "S340G",
            "shuffled_atom_map_negative",
            "random_geometry_target_negative",
        ]
    )
    max_gpu_memory_fraction: float = 0.75
    ensemble_sizes: List[int] = Field(default_factory=lambda: [1, 4])
    require_negative_controls: bool = True

    @model_validator(mode="after")
    def _check_budget(self) -> "FeasibilityConfig":
        if not 0 < self.max_gpu_memory_fraction <= 0.75:
            raise ValueError("guided probe must not reserve more than 75% of GPU memory")
        if not self.ensemble_sizes:
            raise ValueError("at least one ensemble size is required")
        return self


class FeasibilityResult(BaseModel):
    decision: str
    reasons: List[str] = Field(default_factory=list)
    metrics: Dict[str, object] = Field(default_factory=dict)
    negative_controls_passed: bool = False


def run_feasibility_gate(
    config: Optional[FeasibilityConfig] = None,
    *,
    atom_map: Optional[ReactiveAtomMap] = None,
    differentiable_sampler_available: bool = False,
    synthetic_gradient_ok: bool = False,
    negative_controls_passed: bool = False,
) -> FeasibilityResult:
    cfg = config or FeasibilityConfig()
    reasons: List[str] = []
    if atom_map is None or not atom_map.validated:
        reasons.append("validated atom mapping is required")
    if not differentiable_sampler_available:
        reasons.append("differentiable Boltz sampler access is not confirmed")
    if not synthetic_gradient_ok:
        reasons.append("synthetic gradient direction check not passed")
    if cfg.require_negative_controls and not negative_controls_passed:
        reasons.append("negative controls did not pass")
    decision = "GO" if not reasons else "NO_GO"
    return FeasibilityResult(
        decision=decision,
        reasons=reasons,
        metrics={
            "cases": cfg.cases,
            "ensemble_sizes": cfg.ensemble_sizes,
            "max_gpu_memory_fraction": cfg.max_gpu_memory_fraction,
        },
        negative_controls_passed=negative_controls_passed,
    )


def render_feasibility_report(result: FeasibilityResult) -> str:
    lines = [
        "# V4 guided Boltz feasibility",
        "",
        f"Decision: {result.decision}",
        "",
        "Guided Boltz is optional. ReferenceEnsemble v0 remains the default unless this gate is GO.",
        "",
        "Reasons:",
    ]
    if result.reasons:
        for reason in result.reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- all feasibility gates passed")
    lines.extend(
        [
            "",
            "Resource budget:",
            f"- max GPU memory fraction: {result.metrics.get('max_gpu_memory_fraction')}",
            f"- ensemble sizes: {result.metrics.get('ensemble_sizes')}",
        ]
    )
    return "\n".join(lines) + "\n"
