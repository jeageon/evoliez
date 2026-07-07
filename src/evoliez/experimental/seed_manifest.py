"""V4 seed evidence manifest.

The manifest freezes v1-v3 observations before v4 builds new ensemble or
experimental-planning layers. It deliberately stores roles and uncertainty
flags, not only scalar ranks.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from evoliez.mechanism import vocab
from evoliez.ranking.claim_guard import ClaimProvenance


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEED_MANIFEST = REPO_ROOT / "data" / "v4_seed_evidence" / "fdh_v1_v2_v3_manifest.yaml"


class SourceRun(BaseModel):
    date: str
    path: str
    reports: List[str] = Field(default_factory=list)
    hash: str


class SeedCandidate(BaseModel):
    candidate_id: str
    mutation: str
    role: str
    evidence_source: List[str]
    evidence: Dict[str, Any] = Field(default_factory=dict)
    claim_level: str = "L0_uncalibrated"
    uncertainty_flags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_seed(self) -> "SeedCandidate":
        if self.claim_level != "L0_uncalibrated":
            raise ValueError("v4 seed evidence must remain L0 before wet-lab calibration")
        if not self.role:
            raise ValueError("seed candidate role is required")
        if not self.evidence_source:
            raise ValueError("seed candidate evidence_source is required")
        return self


class SeedControls(BaseModel):
    scalar_rank_control: List[str] = Field(default_factory=list)
    no_gain_controls: List[str] = Field(default_factory=list)
    low_ml_uncertainty_controls: List[str] = Field(default_factory=list)
    baseline: List[str] = Field(default_factory=list)


class SelectionPolicy(BaseModel):
    forbidden: List[str] = Field(default_factory=list)
    required_lanes: List[str] = Field(default_factory=list)


class SeedManifest(BaseModel):
    schema_version: float
    target_id: str
    mechanism: str = "hydride_transfer"
    claim_level_default: str = "L0_uncalibrated"
    source_runs: Dict[str, SourceRun]
    candidates: List[SeedCandidate]
    controls: SeedControls
    selection_policy: SelectionPolicy

    @model_validator(mode="after")
    def _check_manifest(self) -> "SeedManifest":
        if self.schema_version != 4.0:
            raise ValueError("schema_version must be 4.0")
        if self.claim_level_default != "L0_uncalibrated":
            raise ValueError("default v4 seed claim must be L0_uncalibrated")
        by_id = {c.candidate_id: c for c in self.candidates}
        # ROADMAP_V3 B6 — the FDH-specific lead/control invariants are gated on the FDH
        # target so the GENERIC SeedManifest schema accepts other enzymes' manifests
        # (previously these hard-coded asserts rejected any non-FDH manifest). The FDH
        # seed still freezes its known lead (mut_00479) + scalar control (Q382R).
        if self.target_id == "fdh_nadp":
            if "mut_00479" not in by_id:
                raise ValueError("mut_00479 lead seed is required")
            if by_id["mut_00479"].mutation != "I208T;R207K;R228P":
                raise ValueError("mut_00479 mutation string drifted")
            if by_id["mut_00479"].role != "tier_A_catalytic_hypothesis":
                raise ValueError("mut_00479 must stay the Tier A catalytic hypothesis")
            scalar = self.find_by_mutation("Q382R")
            if scalar is None or scalar.role != "scalar_rank_control":
                raise ValueError("Q382R must stay a scalar-rank control")
        for c in self.candidates:
            if c.role == "v2_positive_probe" and c.role == "confirmed_lead":
                raise ValueError("v2 probes must not be promoted to confirmed leads")
        return self

    def find_by_mutation(self, mutation: str) -> Optional[SeedCandidate]:
        for c in self.candidates:
            if c.mutation == mutation:
                return c
        return None


def load_seed_manifest(path: Optional[Path] = None) -> SeedManifest:
    """Load the v4 seed manifest. ROADMAP_V3 B6 — the path is configurable (arg or the
    ``EVOLIEZ_SEED_MANIFEST`` env) instead of a fixed FDH default, so a non-FDH target
    can freeze its own seed evidence."""
    if path is None:
        env = os.environ.get("EVOLIEZ_SEED_MANIFEST")
        path = Path(env) if env else DEFAULT_SEED_MANIFEST
    data = yaml.safe_load(Path(path).read_text())
    return SeedManifest(**data)


def _canonical_yaml_bytes(path: Path) -> bytes:
    data = yaml.safe_load(Path(path).read_text())
    return yaml.safe_dump(data, sort_keys=True).encode("utf-8")


def compute_manifest_hash(path: Path = DEFAULT_SEED_MANIFEST) -> str:
    return hashlib.sha256(_canonical_yaml_bytes(path)).hexdigest()


def candidate_by_id(manifest: SeedManifest, candidate_id: str) -> SeedCandidate:
    for candidate in manifest.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def candidate_by_mutation(manifest: SeedManifest, mutation: str) -> SeedCandidate:
    candidate = manifest.find_by_mutation(mutation)
    if candidate is None:
        raise KeyError(mutation)
    return candidate


def freeze_record(path: Path = DEFAULT_SEED_MANIFEST) -> Dict[str, str]:
    return {
        "manifest_path": str(path),
        "manifest_sha256": compute_manifest_hash(path),
        "schema_version": "4.0",
    }


def manifest_claim_provenance(
    manifest: SeedManifest,
    *,
    wetlab_replicated: bool = False,
    known_active_controls: bool = False,
    known_inactive_controls: bool = False,
) -> ClaimProvenance:
    """Translate v4 seed state to the existing ClaimGuard engine."""
    return ClaimProvenance(
        reference_claim_strength=vocab.UNCALIBRATED,
        geometry_claim_ceiling=vocab.UNCALIBRATED,
        ensemble_claim_ceiling=vocab.UNCALIBRATED,
        wetlab_replicated=wetlab_replicated,
        only_short_md=True,
        known_active_controls=known_active_controls,
        known_inactive_controls=known_inactive_controls,
        de_novo_pose_disagreement_high=True,
        wt_reaction_geometry_sparse=True,
        ligand_parameterization_uncertain=True,
        ml_label_source="computational_surrogate",
    )
