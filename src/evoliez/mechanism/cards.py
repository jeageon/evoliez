"""v3 provenance cards (ROADMAP_V3 D2, D4, D6, D7, D9, D10).

All are pydantic v2 models with ``extra='forbid'`` so a renamed/typo'd field fails
loudly at construction — this is exactly the strict-schema layer ClaimGuard relies on
(a silently-failing guard is worse than none). Pure schema; no rdkit/md import.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .vocab import (
    CLAIM_LADDER, CONFIDENCE_LABELS, REFERENCE_TIERS, TIER_CLAIM_CEILING,
    UNCALIBRATED, claim_rank, confidence_from_score, weakest_claim,
)


class _Base(BaseModel):
    # NOTE: the claim-strength / ceiling clamp runs at CONSTRUCTION (the data-driven path — YAML /
    # JSON / dict). A post-construction ``card.claim_strength = 'strong_screening'`` would bypass
    # it, but that requires developer code (not data) and validate_assignment=True recurses
    # against the validators' own internal field assignments, so it is deliberately NOT enabled;
    # this residual is out of the data-driven threat model (Fable verification).
    model_config = {"extra": "forbid"}


# --- D2 ReferenceConfidenceCard ------------------------------------------------------
class ReferenceConfidenceCard(_Base):
    tier: str
    source: str = "unknown"
    resolution_A: Optional[float] = None
    sequence_identity_to_target: Optional[float] = None
    ligand_state: Optional[str] = None
    substrate_is_real: bool = True
    analog_identity: Optional[str] = None
    catalytic_residue_alignment_verified: bool = False
    functional_atom_mapping_verified: bool = False
    protonation_state_assigned: bool = False
    redox_state_assigned: bool = False
    active_closed_state_supported: str = "unknown"
    known_active_controls_available: bool = False
    known_inactive_controls_available: bool = False
    claim_strength: Optional[str] = None       # DERIVED if unset (see grade())

    @model_validator(mode="after")
    def _validate(self) -> "ReferenceConfidenceCard":
        if self.tier not in REFERENCE_TIERS:
            raise ValueError(f"reference tier must be one of {REFERENCE_TIERS}, got {self.tier!r}")
        if self.claim_strength is None:
            self.claim_strength = self.grade()
        elif self.claim_strength not in CLAIM_LADDER:
            raise ValueError(f"claim_strength must be one of {CLAIM_LADDER}")
        else:
            # an explicitly supplied strength may only LOWER (a manual downgrade), never EXCEED
            # the evidence-derived grade — a weak reference must not carry an inflated claim
            # strength into downstream confidence (Fable safety review, fail-open fix).
            self.claim_strength = weakest_claim(self.claim_strength, self.grade())
        return self

    def grade(self) -> str:
        """Derive claim strength from tier + analog + control availability. Each weakness
        can only LOWER the claim (weakest_claim combinator) — never raise it."""
        candidates = [TIER_CLAIM_CEILING.get(self.tier, UNCALIBRATED)]
        from .vocab import HYPOTHESIS_GRADE, MODERATE_SCREENING
        if not self.substrate_is_real:
            candidates.append(MODERATE_SCREENING)          # analog -> cap at moderate
        if not (self.known_active_controls_available or self.known_inactive_controls_available):
            candidates.append(UNCALIBRATED)                # no controls -> uncalibrated
        if not self.functional_atom_mapping_verified:
            candidates.append(HYPOTHESIS_GRADE)
        return weakest_claim(*candidates)


# --- D6 ActiveStateReferenceEnsemble -------------------------------------------------
class ReferenceMember(_Base):
    reference_id: str
    tier: str
    weight: float = 1.0
    source: str = "unknown"
    confidence_card: Optional[ReferenceConfidenceCard] = None

    @model_validator(mode="after")
    def _check_tier(self) -> "ReferenceMember":
        if self.tier not in REFERENCE_TIERS:
            raise ValueError(f"reference tier must be one of {REFERENCE_TIERS}, got {self.tier!r}")
        return self


class TermDistribution(_Base):
    median: Optional[float] = None
    iqr: Optional[float] = None
    source_references: List[str] = Field(default_factory=list)


class EnsembleDisagreement(_Base):
    geometry_variance: float = 0.0
    ligand_pose_variance: float = 0.0
    catalytic_contact_variance: float = 0.0


class ActiveStateReferenceEnsemble(_Base):
    ensemble_id: str
    mechanism_spec_id: str = "mechanism_v1"
    references: List[ReferenceMember] = Field(default_factory=list)
    distance_terms: Dict[str, TermDistribution] = Field(default_factory=dict)
    angle_terms: Dict[str, TermDistribution] = Field(default_factory=dict)
    disagreement: EnsembleDisagreement = Field(default_factory=EnsembleDisagreement)
    claim_ceiling: Optional[str] = None        # DERIVED if unset

    @model_validator(mode="after")
    def _derive_ceiling(self) -> "ActiveStateReferenceEnsemble":
        if not self.references:
            raise ValueError("an ensemble needs at least one reference")
        from .vocab import HYPOTHESIS_GRADE, MODERATE_SCREENING
        # best member tier sets the ceiling, then high disagreement caps it lower
        best = weakest_claim(*[
            (m.confidence_card.claim_strength if m.confidence_card
             else TIER_CLAIM_CEILING.get(m.tier, UNCALIBRATED))
            for m in self._best_members()])
        # disagreement across ANY of the three axes caps the ceiling — ligand-pose and
        # catalytic-contact variance were previously ignored, so a set that agreed on backbone
        # geometry but disagreed wildly on the ligand pose still claimed strong (Fable review).
        worst_var = max(self.disagreement.geometry_variance,
                        self.disagreement.ligand_pose_variance,
                        self.disagreement.catalytic_contact_variance)
        penalties = [best]
        if worst_var >= 0.25:
            penalties.append(HYPOTHESIS_GRADE)
        elif worst_var >= 0.1:
            penalties.append(MODERATE_SCREENING)
        derived = weakest_claim(*penalties)
        if self.claim_ceiling is None:
            self.claim_ceiling = derived
        elif self.claim_ceiling not in CLAIM_LADDER:
            raise ValueError(f"claim_ceiling must be one of {CLAIM_LADDER}")
        else:
            # an explicit ceiling may only TIGHTEN the derived cap, never loosen it (fail-open fix).
            self.claim_ceiling = weakest_claim(self.claim_ceiling, derived)
        return self

    def _best_members(self) -> List[ReferenceMember]:
        """The strongest-tier member(s) — the ceiling is set by the best reference, not
        diluted by weak ones (a single Tier-A beats three Tier-C)."""
        best_rank = min(claim_rank(TIER_CLAIM_CEILING.get(m.tier, UNCALIBRATED))
                        for m in self.references)
        return [m for m in self.references
                if claim_rank(TIER_CLAIM_CEILING.get(m.tier, UNCALIBRATED)) == best_rank]


# --- D4 EvidenceCard -----------------------------------------------------------------
class EvidenceAxis(_Base):
    score: float = 0.0
    confidence: str = "low"
    reason: str = ""
    evidence: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_conf(self) -> "EvidenceAxis":
        if self.confidence not in CONFIDENCE_LABELS:
            raise ValueError(f"confidence must be one of {CONFIDENCE_LABELS}, got {self.confidence!r}")
        return self


class ConfidenceModel(_Base):
    method: str = "rule_based_v1"
    components: Dict[str, float] = Field(default_factory=dict)
    final_confidence: str = "low"

    def recompute(self) -> "ConfidenceModel":
        if self.components:
            self.final_confidence = confidence_from_score(
                sum(self.components.values()) / len(self.components))
        return self


# the canonical axis names (ROADMAP_V3 D4)
EVIDENCE_AXES = (
    "structural_viability", "evolutionary_tolerance", "ligand_cofactor_competence",
    "substrate_positioning", "reaction_geometry_accommodation", "uncertainty",
)


class EvidenceCard(_Base):
    variant_id: str
    structural_viability: EvidenceAxis = Field(default_factory=EvidenceAxis)
    evolutionary_tolerance: EvidenceAxis = Field(default_factory=EvidenceAxis)
    ligand_cofactor_competence: EvidenceAxis = Field(default_factory=EvidenceAxis)
    substrate_positioning: EvidenceAxis = Field(default_factory=EvidenceAxis)
    reaction_geometry_accommodation: EvidenceAxis = Field(default_factory=EvidenceAxis)
    # uncertainty.score is "higher = worse"
    uncertainty: EvidenceAxis = Field(default_factory=EvidenceAxis)
    confidence_model: ConfidenceModel = Field(default_factory=ConfidenceModel)


# --- D7 SimulationSetupCard ----------------------------------------------------------
class LigandParam(_Base):
    redox_state: Optional[str] = None
    protonation_state: Optional[str] = None
    charge_model: Optional[str] = None
    parameter_validated: bool = False


class SimulationSetupCard(_Base):
    force_field_protein: str = "amber14sb"
    force_field_water: str = "tip3p"
    ligand_param_source: str = "gaff"
    ligand_parameters: Dict[str, LigandParam] = Field(default_factory=dict)
    distance_restraints: bool = True
    angle_restraints: bool = False             # invariant: never angle restraints
    md_solvent: str = "implicit"
    md_length_ns: float = 2.0
    md_temperature_K: float = 300.0

    @model_validator(mode="after")
    def _no_angle_restraints(self) -> "SimulationSetupCard":
        if self.angle_restraints:
            raise ValueError("angle restraints are forbidden (would manufacture NAC); "
                             "angle terms are read-only occupancy measures")
        return self

    def md_confidence_penalty(self) -> bool:
        """True if any ligand parameter is unvalidated -> MD evidence confidence is
        downgraded (ROADMAP_V3 D7 / §6 ligand_parameterization_uncertain)."""
        return any(not p.parameter_validated for p in self.ligand_parameters.values())


# --- D9 BenchmarkCard ----------------------------------------------------------------
class BenchmarkCard(_Base):
    target_id: str
    mechanism_template: str
    label_primary: str
    label_secondary: Optional[str] = None
    is_direct_kcat: bool = False
    replicate_available: bool = False
    quantitative: bool = False
    assay_context: Optional[str] = None
    reference_tiers: List[str] = Field(default_factory=list)
    substrate_or_analog: Optional[str] = None
    wt_available: bool = True
    known_active_available: bool = False
    known_inactive_available: bool = False
    allowed_claims: List[str] = Field(default_factory=list)
    prohibited_claims: List[str] = Field(default_factory=list)


# --- D10 WetLabLabelSpec -------------------------------------------------------------
class WetLabLabelSpec(_Base):
    assay_id: str
    endpoint_primary: str
    endpoint_units: str = ""
    substrates: Dict[str, List[float]] = Field(default_factory=dict)
    counter_screen: Dict[str, bool] = Field(default_factory=dict)
    expression_normalized: bool = False
    replicates_biological: int = 1
    replicates_technical: int = 1
    wt_replicates: int = 0
    derived_labels: List[str] = Field(default_factory=list)
    activity_claim_allowed_only_if_replicated: bool = True

    def replicated(self) -> bool:
        return self.replicates_biological >= 2 and self.wt_replicates >= 2
