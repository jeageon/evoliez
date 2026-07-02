"""v3 terminology map (ROADMAP_V3 §4 / V3-1a).

The rename from FDH-funnel vocabulary to mechanism-triage vocabulary happens at the
DISPLAY / boundary layer — internal provenance keys are kept as back-compat aliases so
existing runs (the restored fdh_5track provenance) and the 60+ v2 tests keep parsing.

  - ``display_name(key)``  -> the v3 human-facing label for any old-or-new key
  - ``normalize_key(key)`` -> the canonical v3 key for an old-or-new key (alias-safe)

Pure dict lookups; no deps. Reports call ``display_name``; loaders that want the new
canonical key call ``normalize_key``; nothing is mutated on disk.
"""
from __future__ import annotations

from typing import Dict

# old canonical key -> new canonical v3 key
OLD_TO_NEW: Dict[str, str] = {
    "reference_like": "reference_pose_accommodation",
    "displaced": "pose_uncertainty",
    "nac": "reaction_geometry_accommodation",
    "nac_score": "reaction_geometry_accommodation",
    "nac_fraction": "reaction_geometry_accommodation",
    "delta_nac": "delta_reaction_geometry_accommodation",
    "catalytic_geometry_penalty": "reaction_geometry_penalty",
    "anchored_validation": "reference_pose_accommodation_validation",
    "ml_score": "ml_evidence_prior",
    "final_rank": "triage_recommendation",
    "final_prediction": "triage_recommendation",
}

# new key -> the polished human label shown in reports/papers
NEW_TO_LABEL: Dict[str, str] = {
    "reference_pose_accommodation": "reference-pose accommodation",
    "pose_uncertainty": "alternative-pose uncertainty signal",
    "reaction_geometry_accommodation": "reaction-geometry accommodation",
    "delta_reaction_geometry_accommodation": "Δ reaction-geometry accommodation",
    "reaction_geometry_penalty": "reaction-geometry penalty",
    "reference_pose_accommodation_validation":
        "reference-pose accommodation under short local relaxation",
    "ml_evidence_prior": "ML evidence prior",
    "triage_recommendation": "triage recommendation",
}

# reverse aliases so a NEW key normalizes to itself
_NEW_KEYS = set(OLD_TO_NEW.values()) | set(NEW_TO_LABEL.keys())


def normalize_key(key: str) -> str:
    """Canonical v3 key for any old-or-new key. Unknown keys pass through unchanged."""
    if key in _NEW_KEYS:
        return key
    return OLD_TO_NEW.get(key, key)


def display_name(key: str) -> str:
    """v3 human label for any old-or-new key. Falls back to a prettified key."""
    new = normalize_key(key)
    return NEW_TO_LABEL.get(new, new.replace("_", " "))
