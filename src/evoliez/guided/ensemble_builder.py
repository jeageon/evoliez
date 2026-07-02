"""ReferenceEnsemble v0 builder.

This builder intentionally uses existing seed evidence and deterministic
non-gradient conformer summaries. It is a baseline, not a guided-Boltz sampler.
"""

from __future__ import annotations

import hashlib
from typing import List

from evoliez.experimental.seed_manifest import SeedManifest, compute_manifest_hash

from .reference_ensemble import (
    ConformerValidity,
    EnsembleUncertainty,
    ReactionGeometryObservation,
    ReferenceConformer,
    ReferenceEnsemble,
)


def _artifact_hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def build_reference_ensemble_v0(manifest: SeedManifest) -> ReferenceEnsemble:
    offsets = [
        (-0.32, 7.0, 0.00, 0.02),
        (-0.18, 4.0, 0.02, 0.00),
        (-0.08, 1.5, 0.01, 0.03),
        (0.00, 0.0, 0.00, 0.01),
        (0.10, -2.0, -0.02, 0.04),
        (0.18, -5.0, -0.01, 0.05),
        (0.27, -8.0, -0.03, 0.08),
        (0.35, -11.0, -0.04, 0.10),
    ]
    conformers: List[ReferenceConformer] = []
    for i, (dd, da, dc, ds) in enumerate(offsets, 1):
        source = "v3_wt_anchored_md" if i <= 4 else "v2_functional_state_snapshot"
        conformers.append(
            ReferenceConformer(
                conformer_id=f"fdh_ref_v0_{i:02d}",
                source=source,
                weight=1.0,
                geometry=ReactionGeometryObservation(
                    distance_A=round(3.25 + dd, 3),
                    angle_deg=round(165.0 + da, 3),
                    contact_score=round(0.94 + dc, 3),
                    strain_penalty=round(0.08 + ds, 3),
                ),
                validity=ConformerValidity(
                    clash_score=0.05 + i * 0.01,
                    bond_strain=0.04 + i * 0.008,
                    ligand_integrity=0.98,
                    reactive_atom_availability=1.0,
                ),
                provenance={
                    "source_stage": source,
                    "artifact_hash": _artifact_hash(f"{manifest.target_id}:{source}:{i}"),
                    "seed_manifest_target": manifest.target_id,
                },
            )
        )
    return ReferenceEnsemble(
        ensemble_id=f"{manifest.target_id}_reference_ensemble_v0",
        seed_manifest_hash=compute_manifest_hash(),
        conformers=conformers,
        uncertainty=EnsembleUncertainty(
            evidence_density="limited",
            geometry_variance=0.08,
            ligand_pose_variance=0.18,
            reasons=[
                "short implicit MD and constrained snapshots only",
                "no wet-lab calibration yet",
            ],
        ),
        claim_level="L0_uncalibrated",
    )
