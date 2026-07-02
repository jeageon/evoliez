"""EvoLiEZ v3 mechanism-configurable triage layer (ROADMAP_V3).

Pure-schema + claim-discipline package: mechanism templates, reaction state, and the
provenance cards that let every claim be capped to the evidence that supports it.
"""
from __future__ import annotations

from .cards import (
    ActiveStateReferenceEnsemble, BenchmarkCard, ConfidenceModel, EvidenceAxis,
    EvidenceCard, ReferenceConfidenceCard, ReferenceMember, SimulationSetupCard,
    WetLabLabelSpec,
)
from .spec import (
    CatalyticResidueSpec, GeometryCalibration, GeometryTermSpec, HostContext,
    MechanismSpec, ReactionInfo, ReactionState, list_template_keys,
)
from .vocab import (
    CLAIM_LADDER, GEOMETRY_TIERS, REFERENCE_TIERS, claim_rank, confidence_from_score,
    weakest_claim,
)

__all__ = [
    "MechanismSpec", "ReactionInfo", "ReactionState", "HostContext",
    "CatalyticResidueSpec", "GeometryCalibration", "GeometryTermSpec",
    "list_template_keys",
    "ReferenceConfidenceCard", "ReferenceMember", "ActiveStateReferenceEnsemble",
    "EvidenceCard", "EvidenceAxis", "ConfidenceModel", "SimulationSetupCard",
    "BenchmarkCard", "WetLabLabelSpec",
    "CLAIM_LADDER", "REFERENCE_TIERS", "GEOMETRY_TIERS",
    "claim_rank", "weakest_claim", "confidence_from_score",
]
