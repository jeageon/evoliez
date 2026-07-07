"""ReferenceEnsemble v0 schema and deterministic distributions."""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class ConformerValidity(BaseModel):
    clash_score: float = 0.0
    bond_strain: float = 0.0
    ligand_integrity: float = 1.0
    reactive_atom_availability: float = 1.0

    @property
    def valid(self) -> bool:
        return (
            self.clash_score <= 0.35
            and self.bond_strain <= 0.35
            and self.ligand_integrity >= 0.8
            and self.reactive_atom_availability >= 1.0
        )


class ReactionGeometryObservation(BaseModel):
    distance_A: float
    angle_deg: float
    contact_score: float = 1.0
    strain_penalty: float = 0.0


class ReferenceConformer(BaseModel):
    conformer_id: str
    source: str
    weight: float = 1.0
    geometry: ReactionGeometryObservation
    validity: ConformerValidity = Field(default_factory=ConformerValidity)
    provenance: Dict[str, str] = Field(default_factory=dict)


class ReferenceDistribution(BaseModel):
    mean: float
    median: float
    p10: float
    p90: float
    n: int


class EnsembleUncertainty(BaseModel):
    evidence_density: str = "limited"
    geometry_variance: float = 0.0
    ligand_pose_variance: float = 0.0
    reasons: List[str] = Field(default_factory=list)


class ReferenceEnsemble(BaseModel):
    ensemble_id: str
    seed_manifest_hash: str
    conformers: List[ReferenceConformer]
    distributions: Dict[str, ReferenceDistribution] = Field(default_factory=dict)
    uncertainty: EnsembleUncertainty = Field(default_factory=EnsembleUncertainty)
    claim_level: str = "L0_uncalibrated"
    insufficiency_reason: Optional[str] = None

    @model_validator(mode="after")
    def _check_ensemble(self) -> "ReferenceEnsemble":
        if len(self.conformers) < 8 and not self.insufficiency_reason:
            raise ValueError("ReferenceEnsemble v0 needs >=8 conformers or an insufficiency_reason")
        for conformer in self.conformers:
            for key in ("source_stage", "artifact_hash"):
                if key not in conformer.provenance:
                    raise ValueError(f"conformer {conformer.conformer_id} missing {key}")
        if not self.distributions:
            self.distributions = compute_distributions(self.conformers)
        return self

    @property
    def valid_conformers(self) -> List[ReferenceConformer]:
        return [c for c in self.conformers if c.validity.valid]


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = (len(vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _dist(values: List[float]) -> ReferenceDistribution:
    return ReferenceDistribution(
        mean=round(sum(values) / len(values), 4),
        median=round(statistics.median(values), 4),
        p10=round(_percentile(values, 0.10), 4),
        p90=round(_percentile(values, 0.90), 4),
        n=len(values),
    )


def compute_distributions(conformers: List[ReferenceConformer]) -> Dict[str, ReferenceDistribution]:
    valid = [c for c in conformers if c.validity.valid]
    if not valid:
        valid = conformers
    return {
        "hydride_distance_A": _dist([c.geometry.distance_A for c in valid]),
        "hydride_angle_deg": _dist([c.geometry.angle_deg for c in valid]),
        "anchor_contact": _dist([c.geometry.contact_score for c in valid]),
        "strain_penalty": _dist([c.geometry.strain_penalty for c in valid]),
    }
