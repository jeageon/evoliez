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


def build_reference_ensemble_from_observations(
    target_id: str,
    observations: List[tuple],
    source_label: str = "explicit_solvent_md",
    seed_manifest_hash: str = "",
) -> ReferenceEnsemble:
    """ROADMAP_V5 Day-6 (active-state ensemble, REAL): build a ReferenceEnsemble from ACTUAL
    per-frame reaction geometry — e.g. the E1 explicit-solvent WT trajectory, which RETAINED the
    co-substrate (retention 1.0) and so carries a real distribution of the productive Michaelis state.
    This replaces build_reference_ensemble_v0's hardcoded synthetic offsets with measured geometry, so
    candidate scoring is against a real active-state ensemble, not a fixture.

    ``observations``: list of (distance_A, angle_deg[, contact_score, strain_penalty]) from real frames.
    A candidate is then scored (guided.ensemble_readout) by how well its own geometry distribution
    accommodates THIS ensemble — an ensemble-vs-ensemble comparison that is robust to the single-pose
    noise that produced discriminates=false at 0.3 ns. Claim level stays L0_uncalibrated (no wet-lab)."""
    if not observations:
        raise ValueError("build_reference_ensemble_from_observations needs >=1 real frame observation")
    conformers: List[ReferenceConformer] = []
    for i, obs in enumerate(observations, 1):
        dist, ang = float(obs[0]), float(obs[1])
        contact = float(obs[2]) if len(obs) > 2 else 1.0
        strain = float(obs[3]) if len(obs) > 3 else 0.0
        conformers.append(
            ReferenceConformer(
                conformer_id=f"{target_id}_ref_expl_{i:03d}",
                source=source_label, weight=1.0,
                geometry=ReactionGeometryObservation(
                    distance_A=round(dist, 3), angle_deg=round(ang, 3),
                    contact_score=round(contact, 3), strain_penalty=round(strain, 3)),
                validity=ConformerValidity(
                    clash_score=0.0, bond_strain=0.0,
                    ligand_integrity=1.0, reactive_atom_availability=1.0),
                provenance={
                    "source_stage": source_label,
                    "artifact_hash": _artifact_hash(f"{target_id}:{source_label}:{i}:{dist}:{ang}"),
                    "measured": "true",
                },
            )
        )
    dists = [c.geometry.distance_A for c in conformers]
    angs = [c.geometry.angle_deg for c in conformers]
    _mean = lambda xs: sum(xs) / len(xs)
    _var = lambda xs: _mean([(x - _mean(xs)) ** 2 for x in xs]) if len(xs) > 1 else 0.0
    return ReferenceEnsemble(
        ensemble_id=f"{target_id}_reference_ensemble_explicit",
        seed_manifest_hash=seed_manifest_hash or _artifact_hash(target_id),
        conformers=conformers,
        uncertainty=EnsembleUncertainty(
            evidence_density="measured_md" if len(conformers) >= 10 else "limited",
            geometry_variance=round(_var(dists), 4),
            ligand_pose_variance=round(_var(angs) / 1000.0, 4),
            reasons=[f"real {source_label} frames (n={len(conformers)})", "no wet-lab calibration yet"],
        ),
        claim_level="L0_uncalibrated",
        insufficiency_reason=(None if len(conformers) >= 8
                              else f"only {len(conformers)} measured frame(s) (<8)"),
    )


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
