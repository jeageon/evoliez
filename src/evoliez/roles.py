"""Canonical ligand roles + per-role design objectives — GENERIC, never FDH-specific.

A functional active site holds several ligands/partners, each playing a different role; the
mutation objective AND the validation gate differ by role (ROADMAP_V2 principle 3 / §2a.2).
This is the single vocabulary so every stage agrees on what a role means:
  s01 reference state · s06 functional-state graph · s09 anchored features · s10 pose gate
  (md/pose_gate.ROLE_THRESHOLDS) · s11 role-aware scoring.

The existing `LigandInput.role` is free-form for back-compat; `normalize_role` maps it onto
the canonical set and `resolve_role` applies the positional default (primary = design_ligand,
each extra_ligand = cofactor).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# --- the canonical roles (the v2 vocabulary) ---------------------------------------
DESIGN_LIGAND = "design_ligand"
COFACTOR = "cofactor"
SUBSTRATE = "substrate"
PRODUCT = "product"
METAL = "metal"
CONTEXT = "context"
TS_PROXY = "transition_state_proxy"

CANONICAL_ROLES: Tuple[str, ...] = (
    DESIGN_LIGAND, COFACTOR, SUBSTRATE, PRODUCT, METAL, CONTEXT, TS_PROXY,
)

# free-form aliases -> canonical (so existing configs + casual strings keep working)
_ALIASES: Dict[str, str] = {
    "design": DESIGN_LIGAND, "designligand": DESIGN_LIGAND, "ligand": DESIGN_LIGAND,
    "binder": DESIGN_LIGAND, "target": DESIGN_LIGAND, "design_target": DESIGN_LIGAND,
    "cofactor": COFACTOR, "coenzyme": COFACTOR, "co_factor": COFACTOR,
    "substrate": SUBSTRATE, "reactant": SUBSTRATE,
    "product": PRODUCT,
    "metal": METAL, "ion": METAL, "metal_ion": METAL, "cation": METAL, "anion": METAL,
    "context": CONTEXT, "scaffold": CONTEXT, "structural": CONTEXT, "other": CONTEXT,
    "ts": TS_PROXY, "ts_proxy": TS_PROXY, "transition_state": TS_PROXY,
    "transitionstate": TS_PROXY, "tsproxy": TS_PROXY,
}


@dataclass(frozen=True)
class RoleObjective:
    """What 'improvement' means for a ligand in this role (drives s09/s11 scoring)."""
    role: str
    objective: str                          # human-readable summary
    optimize_affinity: bool                 # binding affinity / stability is the goal
    optimize_catalytic_geometry: bool       # near-attack / reaction geometry is the goal
    optimize_orientation_retention: bool    # keep the cofactor oriented + retained
    optimize_coordination: bool             # metal coordination geometry
    preserve_contacts: bool                 # context: keep WT conserved contacts


# the per-role objective table (ROADMAP_V2 principle 3)
ROLE_OBJECTIVES: Dict[str, RoleObjective] = {
    DESIGN_LIGAND: RoleObjective(
        DESIGN_LIGAND, "affinity + stability", True, False, False, False, False),
    COFACTOR: RoleObjective(
        COFACTOR, "orientation + retention", False, False, True, False, False),
    SUBSTRATE: RoleObjective(
        SUBSTRATE, "catalytic geometry", False, True, False, False, False),
    TS_PROXY: RoleObjective(
        TS_PROXY, "catalytic (transition-state) geometry", False, True, False, False, False),
    PRODUCT: RoleObjective(
        PRODUCT, "release / no over-binding", False, False, False, False, False),
    METAL: RoleObjective(
        METAL, "coordination geometry", False, False, False, True, False),
    CONTEXT: RoleObjective(
        CONTEXT, "conserved-contact preservation", False, False, False, False, True),
}


def normalize_role(role: Optional[str]) -> Optional[str]:
    """Map a free-form role string onto a canonical role; None if unset, None if unknown."""
    if role is None:
        return None
    key = str(role).strip().lower().replace(" ", "_").replace("-", "_")
    if key in CANONICAL_ROLES:
        return key
    return _ALIASES.get(key)


def resolve_role(role: Optional[str], *, is_primary: bool) -> str:
    """Resolve a ligand's effective role: an explicit (normalized) role wins; otherwise the
    POSITIONAL default — the primary `InputConfig.ligand` is the design ligand, every
    `extra_ligand` defaults to a cofactor (matches the existing dock/keep_as_context logic)."""
    norm = normalize_role(role)
    if norm is not None:
        return norm
    return DESIGN_LIGAND if is_primary else COFACTOR


def objective_for(role: Optional[str], *, is_primary: bool = False) -> RoleObjective:
    """The design objective for a ligand given its (free-form or canonical) role."""
    return ROLE_OBJECTIVES[resolve_role(role, is_primary=is_primary)]
