"""Candidate readout against ReferenceEnsemble v0."""

from __future__ import annotations

from typing import Dict

from evoliez.experimental.seed_manifest import SeedManifest

from .reference_ensemble import ReferenceEnsemble
from .soft_kernels import soft_reaction_geometry_score


_PROFILES: Dict[str, Dict[str, float]] = {
    "I208T;R207K;R228P": {
        "distance_A": 3.42,
        "angle_deg": 158.0,
        "contact_score": 0.92,
        "strain_penalty": 0.10,
        "structural_viability": 0.72,
        "pose_uncertainty": 0.55,
        "ligand_cofactor_competence": 0.86,
        "evolutionary_tolerance": 0.64,
    },
    "Q382R": {
        "distance_A": 4.95,
        "angle_deg": 103.0,
        "contact_score": 0.88,
        "strain_penalty": 0.12,
        "structural_viability": 0.92,
        "pose_uncertainty": 0.46,
        "ligand_cofactor_competence": 0.90,
        "evolutionary_tolerance": 0.78,
    },
    "S340G": {
        "distance_A": 4.10,
        "angle_deg": 87.0,
        "contact_score": 0.83,
        "strain_penalty": 0.08,
        "structural_viability": 0.84,
        "pose_uncertainty": 0.62,
        "ligand_cofactor_competence": 0.82,
        "evolutionary_tolerance": 0.58,
    },
    "G291A": {
        "distance_A": 4.35,
        "angle_deg": 99.0,
        "contact_score": 0.78,
        "strain_penalty": 0.16,
        "structural_viability": 0.74,
        "pose_uncertainty": 0.58,
        "ligand_cofactor_competence": 0.75,
        "evolutionary_tolerance": 0.52,
    },
    "P262Q;M261I;R207K": {
        "distance_A": 4.20,
        "angle_deg": 74.0,
        "contact_score": 0.82,
        "strain_penalty": 0.24,
        "structural_viability": 0.80,
        "pose_uncertainty": 0.66,
        "ligand_cofactor_competence": 0.78,
        "evolutionary_tolerance": 0.61,
    },
    "N260H": {
        "distance_A": 3.92,
        "angle_deg": 92.0,
        "contact_score": 0.80,
        "strain_penalty": 0.18,
        "structural_viability": 0.82,
        "pose_uncertainty": 0.72,
        "ligand_cofactor_competence": 0.77,
        "evolutionary_tolerance": 0.55,
    },
}


def _profile_for(mutation: str) -> Dict[str, float]:
    return dict(
        _PROFILES.get(
            mutation,
            {
                "distance_A": 3.95,
                "angle_deg": 120.0,
                "contact_score": 0.72,
                "strain_penalty": 0.18,
                "structural_viability": 0.55,
                "pose_uncertainty": 0.60,
                "ligand_cofactor_competence": 0.55,
                "evolutionary_tolerance": 0.50,
            },
        )
    )


def score_candidate_against_ensemble(
    candidate_id: str,
    mutation: str,
    ensemble: ReferenceEnsemble,
) -> Dict[str, float]:
    profile = _profile_for(mutation)
    dist = ensemble.distributions["hydride_distance_A"]
    reaction = soft_reaction_geometry_score(
        distance_A=profile["distance_A"],
        angle_deg=profile["angle_deg"],
        contact_score=profile["contact_score"],
        strain_penalty=profile["strain_penalty"],
        ensemble_distance_mean=dist.mean,
        ensemble_distance_p10=dist.p10,
        ensemble_distance_p90=dist.p90,
    )
    reference_fit = max(0.0, min(1.0, reaction * 1.15))
    return {
        "candidate_id": candidate_id,
        "mutation": mutation,
        "reaction_geometry_accommodation": round(reaction, 6),
        "reference_state_accommodation": round(reference_fit, 6),
        "substrate_positioning": round((reaction + profile["contact_score"]) / 2.0, 6),
        "structural_viability": profile["structural_viability"],
        "ligand_cofactor_competence": profile["ligand_cofactor_competence"],
        "evolutionary_tolerance": profile["evolutionary_tolerance"],
        "pose_uncertainty": profile["pose_uncertainty"],
        "experimental_calibration": 0.0,
    }


def score_manifest_candidates(manifest: SeedManifest, ensemble: ReferenceEnsemble) -> Dict[str, Dict[str, float]]:
    return {
        c.candidate_id: score_candidate_against_ensemble(c.candidate_id, c.mutation, ensemble)
        for c in manifest.candidates
    }
