"""Enzyme-mechanism layer (user priority #1).

Annotates catalytic / acid-base / nucleophile / metal / cofactor residues,
the substrate reactive ligand atom(s), the key transfer distance, and a
transition-state-like geometry score. This lets the model prefer mutations
that *keep a reactive geometry*, not just ones that bind the ligand harder.

All outputs are FEATURES / priors / penalties, never supervised labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from evoliez.adapters.mcsa import heuristic_roles, load_annotation
from evoliez.features.geometry import dist, ligand_centroid
from evoliez.types import Complex

# Ideal heavy-atom transfer distance (Å) for a reactive contact; ~H-bond /
# hydride-transfer range. Used only to score geometry plausibility.
_IDEAL_TRANSFER = 3.4
_TS_TOL = 1.6

_ROLE_KEYS = [
    "catalytic", "acid_base", "nucleophile",
    "electrophile", "metal_coord", "cofactor_binding",
]


@dataclass
class Mechanism:
    roles: Dict[int, Dict[str, bool]] = field(default_factory=dict)
    reactive_ligand_atoms: List[str] = field(default_factory=list)
    transfer_distance: float = 0.0
    transfer_residue: Optional[int] = None
    ts_geometry_score: float = 0.0  # 0..1, 1 = ideal reactive geometry
    source: str = "heuristic"

    def role_vec(self, residue_index: int) -> List[float]:
        r = self.roles.get(residue_index, {})
        return [1.0 if r.get(k) else 0.0 for k in _ROLE_KEYS]


def _reactive_ligand_atoms(cx: Complex) -> List[str]:
    """Heuristic reactive atoms: charged / polar / aromatic-ring carbons that
    sit closest to the catalytic centroid (e.g. NADP nicotinamide C4,
    phosphate O, carboxylate O)."""
    atoms = cx.ligand.atoms
    if not atoms:
        return []
    scored = []
    for a in atoms:
        w = 0.0
        if a.formal_charge != 0:
            w += 2.0
        if a.is_acceptor or a.is_donor:
            w += 1.0
        if a.aromatic:
            w += 0.5
        scored.append((w, a.id))
    scored.sort(reverse=True)
    return [aid for w, aid in scored[:4] if w > 0] or [atoms[0].id]


def annotate(
    cx: Complex,
    *,
    catalytic_positions: Sequence[int],
    cofactor: Optional[str],
    annotation_file: Optional[str] = None,
) -> Mechanism:
    residue_aa = {r.index: r.aa for r in cx.structure.residues}
    ext = load_annotation(annotation_file)
    if ext and "catalytic_residues" in ext:
        cps = [int(x) for x in ext["catalytic_residues"]]
        roles = {
            int(k): v for k, v in ext.get("roles", {}).items()
        } or heuristic_roles(cps, residue_aa, ext.get("cofactor", cofactor))
        source = "m-csa"
    else:
        cps = list(catalytic_positions)
        roles = heuristic_roles(cps, residue_aa, cofactor)
        source = "heuristic"

    reactive = _reactive_ligand_atoms(cx)
    atom_by_id = {a.id: a for a in cx.ligand.atoms}
    by_pos = {r.index: r for r in cx.structure.residues}

    best_d, best_res = 99.0, None
    for p in cps:
        r = by_pos.get(p)
        if r is None:
            continue
        ref = r.sidechain_centroid or r.ca
        for aid in reactive:
            a = atom_by_id.get(aid)
            if a is None:
                continue
            d = dist(ref, a.coord)
            if d < best_d:
                best_d, best_res = d, p

    if best_res is None:  # no catalytic residues given -> use pocket centroid
        lc = ligand_centroid(cx.ligand.atoms) if cx.ligand.atoms else (0, 0, 0)
        best_d = min(
            (dist(r.sidechain_centroid or r.ca, lc)
             for r in cx.structure.residues),
            default=99.0,
        )
    ts = max(0.0, 1.0 - abs(best_d - _IDEAL_TRANSFER) / _TS_TOL)
    return Mechanism(
        roles=roles,
        reactive_ligand_atoms=reactive,
        transfer_distance=round(best_d, 3),
        transfer_residue=best_res,
        ts_geometry_score=round(ts, 4),
        source=source,
    )


def catalytic_geometry_deviation(wt: Mechanism, mut: Mechanism) -> float:
    """Positive = mutation worsened the reactive geometry (penalty input)."""
    return round(max(0.0, wt.ts_geometry_score - mut.ts_geometry_score), 4)
