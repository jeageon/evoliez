"""v3 controlled vocabularies — the single source of truth for the enums every v3
card shares (ROADMAP_V3 §6 disposition policy + claim-strength ladder).

Kept dependency-free (no pydantic) so it imports in the light env and can be reused
by ClaimGuard, EvidenceCard, and the reference cards without import cycles.
"""
from __future__ import annotations

from typing import Tuple

# --- claim-strength ladder (strongest -> weakest) ------------------------------------
# ReferenceConfidenceCard.claim_strength + ClaimGuard ceilings live on this ladder.
STRONG_SCREENING = "strong_screening"
MODERATE_SCREENING = "moderate_screening"
HYPOTHESIS_GRADE = "hypothesis_grade"
UNCALIBRATED = "uncalibrated"

CLAIM_LADDER: Tuple[str, ...] = (
    STRONG_SCREENING, MODERATE_SCREENING, HYPOTHESIS_GRADE, UNCALIBRATED,
)


def claim_rank(strength: str) -> int:
    """Position on the ladder; higher index = weaker. Unknown -> weakest (fail-safe)."""
    try:
        return CLAIM_LADDER.index(strength)
    except ValueError:
        return len(CLAIM_LADDER) - 1


def weakest_claim(*strengths: str) -> str:
    """The most conservative (weakest) of the given strengths — the fail-safe combinator.
    Any unknown/garbage value is normalized to UNCALIBRATED so a typo can never leak
    through as a claim strength. No args -> UNCALIBRATED (the floor)."""
    if not strengths:
        return UNCALIBRATED
    normalized = [s if s in CLAIM_LADDER else UNCALIBRATED for s in strengths]
    return max(normalized, key=claim_rank)


# --- reference tiers (ROADMAP_V3 §2 / D2) --------------------------------------------
TIER_A = "A"   # experimental ternary / TS-analog
TIER_B = "B"   # homolog ternary transfer
TIER_C = "C"   # pose-transfer + constrained docking
TIER_D = "D"   # de-novo only
REFERENCE_TIERS: Tuple[str, ...] = (TIER_A, TIER_B, TIER_C, TIER_D)

# tier -> the BEST claim strength it can support (a ceiling, before other penalties)
TIER_CLAIM_CEILING = {
    TIER_A: STRONG_SCREENING,
    TIER_B: STRONG_SCREENING,
    TIER_C: MODERATE_SCREENING,
    TIER_D: HYPOTHESIS_GRADE,
}

# --- geometry-tolerance calibration tiers (ROADMAP_V3 D1) ----------------------------
G1 = "G1"  # WT + known-active empirical distribution (MD/QM-MM)
G2 = "G2"  # homolog active-state ensemble distribution
G3 = "G3"  # literature mechanism default
G4 = "G4"  # chemistry-informed heuristic
G5 = "G5"  # user-defined uncalibrated cutoff
GEOMETRY_TIERS: Tuple[str, ...] = (G1, G2, G3, G4, G5)

GEOMETRY_TIER_CLAIM_CEILING = {
    G1: STRONG_SCREENING,
    G2: STRONG_SCREENING,
    G3: MODERATE_SCREENING,
    G4: HYPOTHESIS_GRADE,
    G5: UNCALIBRATED,   # "diagnostic only"
}

# --- confidence labels (EvidenceCard, ROADMAP_V3 D4) ---------------------------------
CONF_HIGH = "high"
CONF_MEDIUM = "medium"
CONF_LOW_TO_MEDIUM = "low_to_medium"
CONF_LOW = "low"
CONFIDENCE_LABELS: Tuple[str, ...] = (CONF_HIGH, CONF_MEDIUM, CONF_LOW_TO_MEDIUM, CONF_LOW)


def confidence_from_score(x: float) -> str:
    """Map a [0,1] aggregate confidence score to a label (ROADMAP_V3 D4 confidence_model)."""
    if x >= 0.75:
        return CONF_HIGH
    if x >= 0.55:
        return CONF_MEDIUM
    if x >= 0.35:
        return CONF_LOW_TO_MEDIUM
    return CONF_LOW


# --- disposition policy (ROADMAP_V3 §6, shared by ClaimGuard + EvidenceCard) ----------
ABORT = "hard_abort"
DOWNGRADE = "claim_downgrade"
FLAG = "flag"
CONFIDENCE_DOWNGRADE = "confidence_downgrade"

# condition -> (disposition, note). The single table both claim_guard and the evidence
# builder import, so they never disagree on how a problem is handled.
DISPOSITION_POLICY = {
    "required_functional_atom_missing": (ABORT, "geometry uncomputable"),
    "catalytic_residue_mapping_failed": (ABORT, "mechanism score impossible"),
    "redox_or_protonation_state_missing": (ABORT, "geometry meaning changes (redox enzymes)"),
    "reference_tier_c_or_d": (DOWNGRADE, "screening still valid"),
    "no_known_controls": (FLAG, "uncalibrated; novel enzyme OK"),
    "wt_reaction_geometry_sparse": (FLAG, "diagnostic-only; reference/sampling issue likely"),
    "ligand_parameterization_uncertain": (CONFIDENCE_DOWNGRADE, "MD evidence weakened"),
    "de_novo_pose_disagreement_high": (FLAG, "uncertainty high; must not declare inactive"),
    "ml_ood_high": (CONFIDENCE_DOWNGRADE, "do not discard the candidate"),
}
