"""CandidateAccommodationScore, separate from legacy final_score."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field, model_validator

from .evidence_card_v4 import EvidenceCardV4


class CandidateAccommodationScore(BaseModel):
    candidate_id: str
    mutation: str
    score: float
    components: Dict[str, float] = Field(default_factory=dict)
    uses_final_score: bool = False

    @model_validator(mode="after")
    def _check_not_final_score(self) -> "CandidateAccommodationScore":
        if self.uses_final_score:
            raise ValueError("CandidateAccommodationScore must be separate from final_score")
        return self


def compute_candidate_accommodation(card: EvidenceCardV4) -> CandidateAccommodationScore:
    components = {
        "reaction_geometry_accommodation": card.reaction_geometry_accommodation.score,
        "reference_state_accommodation": card.reference_state_accommodation.score,
        "ligand_cofactor_competence": card.ligand_cofactor_competence.score,
        "substrate_positioning": card.substrate_positioning.score,
        "structural_viability": card.structural_viability.score,
        "pose_uncertainty_penalty": 1.0 - card.pose_uncertainty.score,
    }
    score = (
        0.30 * components["reaction_geometry_accommodation"]
        + 0.20 * components["reference_state_accommodation"]
        + 0.15 * components["ligand_cofactor_competence"]
        + 0.15 * components["substrate_positioning"]
        + 0.10 * components["structural_viability"]
        + 0.10 * components["pose_uncertainty_penalty"]
    )
    return CandidateAccommodationScore(
        candidate_id=card.candidate_id,
        mutation=card.mutation,
        score=round(score, 6),
        components=components,
        uses_final_score=False,
    )
