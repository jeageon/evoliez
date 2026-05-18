"""Restraint-group selection (spec section 15.5).

Classifies atoms/residues into the spec's restraint tiers. Used by the OpenMM
engine; kept separate so the policy is testable without OpenMM installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from evoliez.features.geometry import dist, ligand_centroid
from evoliez.types import Complex


@dataclass
class RestraintPlan:
    strong_backbone: List[int] = field(default_factory=list)  # distant backbone
    weak_backbone: List[int] = field(default_factory=list)  # active-site backbone
    initially_restrained: List[int] = field(default_factory=list)  # ligand-proximal
    free: List[int] = field(default_factory=list)  # pocket / mutation side chains


def build_restraint_plan(
    cx: Complex,
    mutation_positions: Sequence[int],
    *,
    pocket_radius: float = 8.0,
    active_site_radius: float = 12.0,
) -> RestraintPlan:
    plan = RestraintPlan()
    if not cx.ligand.atoms:
        plan.strong_backbone = [r.index for r in cx.structure.residues]
        return plan
    lc = ligand_centroid(cx.ligand.atoms)
    mut = set(mutation_positions)
    for r in cx.structure.residues:
        ref = r.sidechain_centroid or r.ca
        d = dist(ref, lc)
        if r.index in mut or d <= pocket_radius:
            plan.free.append(r.index)
        elif d <= active_site_radius:
            plan.weak_backbone.append(r.index)
            plan.initially_restrained.append(r.index)
        else:
            plan.strong_backbone.append(r.index)
    return plan
