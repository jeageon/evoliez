"""V4 evidence cards.

V4 cards keep score, confidence, evidence, and provenance separate for every
axis. They are designed for experiment planning, not activity prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from evoliez.mechanism.vocab import CONFIDENCE_LABELS, confidence_from_score


V4_EVIDENCE_AXES = (
    "structural_viability",
    "evolutionary_tolerance",
    "ligand_cofactor_competence",
    "substrate_positioning",
    "reaction_geometry_accommodation",
    "reference_state_accommodation",
    "pose_uncertainty",
    "experimental_calibration",
)


class EvidenceAxisV4(BaseModel):
    score: float = 0.0
    confidence: str = "low"
    evidence: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    @field_validator("score", mode="before")
    @classmethod
    def _reject_bool_score(cls, value):
        if isinstance(value, bool):
            raise ValueError("evidence axis score must not be a bare boolean")
        return value

    @model_validator(mode="after")
    def _check_axis(self) -> "EvidenceAxisV4":
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("evidence axis score must be in [0, 1]")
        if self.confidence not in CONFIDENCE_LABELS:
            raise ValueError(f"confidence must be one of {CONFIDENCE_LABELS}")
        return self


class EvidenceCardV4(BaseModel):
    candidate_id: str
    mutation: str
    role: str
    claim_level: str = "L0_uncalibrated"
    structural_viability: EvidenceAxisV4 = Field(default_factory=EvidenceAxisV4)
    evolutionary_tolerance: EvidenceAxisV4 = Field(default_factory=EvidenceAxisV4)
    ligand_cofactor_competence: EvidenceAxisV4 = Field(default_factory=EvidenceAxisV4)
    substrate_positioning: EvidenceAxisV4 = Field(default_factory=EvidenceAxisV4)
    reaction_geometry_accommodation: EvidenceAxisV4 = Field(default_factory=EvidenceAxisV4)
    reference_state_accommodation: EvidenceAxisV4 = Field(default_factory=EvidenceAxisV4)
    # Higher score means worse uncertainty risk.
    pose_uncertainty: EvidenceAxisV4 = Field(default_factory=EvidenceAxisV4)
    experimental_calibration: EvidenceAxisV4 = Field(default_factory=EvidenceAxisV4)
    summary_verdict: str = "screening_priority"

    @model_validator(mode="after")
    def _check_card(self) -> "EvidenceCardV4":
        if self.claim_level != "L0_uncalibrated" and not self.experimental_calibration.evidence:
            raise ValueError("non-L0 v4 cards require experimental calibration evidence")
        return self

    def axes(self) -> Dict[str, EvidenceAxisV4]:
        return {name: getattr(self, name) for name in V4_EVIDENCE_AXES}


def _conf(score: float, *, high_uncertainty: bool = False) -> str:
    base = score if not high_uncertainty else 1.0 - score
    return confidence_from_score(max(0.0, min(1.0, base)))


def _axis(
    score: float,
    confidence: Optional[str],
    evidence: List[str],
    provenance: Dict[str, Any],
    reason: str = "",
) -> EvidenceAxisV4:
    return EvidenceAxisV4(
        score=round(float(score), 6),
        confidence=confidence or _conf(float(score)),
        evidence=evidence,
        provenance=provenance,
        reason=reason,
    )


def build_evidence_card_v4(seed_candidate, accommodation: Dict[str, float]) -> EvidenceCardV4:
    cid = seed_candidate.candidate_id
    mutation = seed_candidate.mutation
    role = seed_candidate.role
    seed_prov = {
        "candidate_id": cid,
        "mutation": mutation,
        "seed_role": role,
        "source": ",".join(seed_candidate.evidence_source),
    }
    rg = float(accommodation.get("reaction_geometry_accommodation", 0.0))
    pose_unc = float(accommodation.get("pose_uncertainty", 0.5))
    low_conf_reason = "high score with low calibration confidence" if rg >= 0.25 else "low ensemble support"
    return EvidenceCardV4(
        candidate_id=cid,
        mutation=mutation,
        role=role,
        claim_level=seed_candidate.claim_level,
        structural_viability=_axis(
            accommodation.get("structural_viability", 0.0),
            None,
            ["stability_or_md_screening_signal"],
            seed_prov,
        ),
        evolutionary_tolerance=_axis(
            accommodation.get("evolutionary_tolerance", 0.0),
            None,
            ["msa_neff_prior_not_activity_evidence"],
            seed_prov,
        ),
        ligand_cofactor_competence=_axis(
            accommodation.get("ligand_cofactor_competence", 0.0),
            None,
            ["anchor_contact_retention_prior"],
            seed_prov,
        ),
        substrate_positioning=_axis(
            accommodation.get("substrate_positioning", 0.0),
            None,
            ["soft_distance_angle_contact_readout"],
            seed_prov,
        ),
        reaction_geometry_accommodation=_axis(
            rg,
            "low_to_medium" if rg >= 0.25 else "low",
            ["soft_reaction_geometry_distribution"],
            seed_prov,
            reason=low_conf_reason,
        ),
        reference_state_accommodation=_axis(
            accommodation.get("reference_state_accommodation", 0.0),
            "low_to_medium" if rg >= 0.25 else "low",
            ["reference_ensemble_v0_fit"],
            seed_prov,
        ),
        pose_uncertainty=_axis(
            pose_unc,
            _conf(pose_unc, high_uncertainty=True),
            ["pose_disagreement_recorded_as_uncertainty"],
            seed_prov,
            reason="uncertainty risk, not inactivity evidence",
        ),
        experimental_calibration=_axis(
            accommodation.get("experimental_calibration", 0.0),
            "low",
            ["no_wetlab_calibration"],
            seed_prov,
            reason="claims remain L0 until assay data are ingested",
        ),
        summary_verdict="prioritized_for_assay" if role == "tier_A_catalytic_hypothesis" else "control_or_probe",
    )
