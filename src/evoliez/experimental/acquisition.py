"""Round-2 acquisition function and library proposal (ROADMAP_V4 §7.10).

After a first wet-lab round, the next library should be smaller and smarter: exploit
around measured hits, explore around false-negative / high-uncertainty lanes, keep
mechanistic diversity, and value deconvolution — while penalizing synthesis and
expression risk. This module encodes exactly the acquisition score the roadmap defines
and proposes a round-2 library from it.

Two invariants are enforced structurally:

* **Wet-lab gate** — a round-2 proposal requires a calibrated (L1+) :class:`CalibrationResult`.
  Without wet-lab data the proposal refuses to run (there is nothing to learn from yet).
* **Not final_score-only** — candidate rows carrying nothing but ``final_score`` are
  rejected, mirroring the first-round planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from evoliez.experimental.library_plan import reject_final_score_only

from .active_learning import require_calibrated_for_round2
from .calibration import CalibrationResult


@dataclass
class AcquisitionFeatures:
    """The evidence axes the round-2 acquisition score trades off."""

    expected_improvement: float = 0.0
    uncertainty_reduction: float = 0.0
    mechanistic_diversity_bonus: float = 0.0
    deconvolution_value: float = 0.0
    synthesis_risk_penalty: float = 0.0
    expression_failure_risk: float = 0.0


@dataclass
class Round2Candidate:
    mutation: str
    role: str  # "exploit" (around a measured hit) | "explore" (uncertainty / false-negative)
    features: AcquisitionFeatures
    candidate_id: str = ""


@dataclass
class Round2Proposal:
    mutation: str
    candidate_id: str
    role: str
    acquisition: float


def acquisition_score(features: AcquisitionFeatures) -> float:
    """The roadmap's acquisition function.

    ``EI + uncertainty_reduction + mechanistic_diversity_bonus + deconvolution_value
    - synthesis_risk_penalty - expression_failure_risk``.
    """
    value = (
        features.expected_improvement
        + features.uncertainty_reduction
        + features.mechanistic_diversity_bonus
        + features.deconvolution_value
        - features.synthesis_risk_penalty
        - features.expression_failure_risk
    )
    return round(value, 6)


def propose_round2_library(
    result: CalibrationResult,
    candidates: Sequence[Round2Candidate],
    *,
    size: int = 16,
    min_explore_fraction: float = 0.25,
) -> List[Round2Proposal]:
    """Propose a round-2 library, balancing exploitation and exploration.

    Requires a calibrated result (raises otherwise). Reserves at least
    ``min_explore_fraction`` of the library for exploration so the proposal never
    collapses onto only high-scoring exploitation candidates, then fills the remainder by
    descending acquisition score.
    """
    require_calibrated_for_round2(result)
    if size <= 0:
        raise ValueError("round-2 library size must be positive")

    ranked = sorted(candidates, key=lambda c: acquisition_score(c.features), reverse=True)
    exploit = [c for c in ranked if c.role == "exploit"]
    explore = [c for c in ranked if c.role == "explore"]

    n_explore = min(len(explore), max(1, round(size * min_explore_fraction))) if explore else 0
    chosen: List[Round2Candidate] = []
    chosen.extend(explore[:n_explore])
    for cand in ranked:
        if len(chosen) >= size:
            break
        if cand in chosen:
            continue
        chosen.append(cand)

    chosen.sort(key=lambda c: acquisition_score(c.features), reverse=True)
    return [
        Round2Proposal(
            mutation=c.mutation,
            candidate_id=c.candidate_id or c.mutation,
            role=c.role,
            acquisition=acquisition_score(c.features),
        )
        for c in chosen[:size]
    ]


def propose_round2_from_rows(
    result: CalibrationResult,
    rows: Sequence[dict],
    candidates: Sequence[Round2Candidate],
    *,
    size: int = 16,
) -> List[Round2Proposal]:
    """Same as :func:`propose_round2_library` but first rejects final_score-only input.

    ``rows`` is the raw candidate table the acquisition features were derived from; if it
    carries nothing but ``final_score`` the proposal is refused (ROADMAP_V4 invariant #4).
    """
    reject_final_score_only(rows)
    return propose_round2_library(result, candidates, size=size)
