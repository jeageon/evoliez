"""Geometry helpers (spec 11.4 / 14.2): contacts, interaction typing,
catalytic-distance tracking. Pure numpy, no structure-toolkit dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from evoliez.types import LigandAtom, ProteinStructure, Residue

_POS_AA = {"K", "R", "H"}
_NEG_AA = {"D", "E"}
_AROM_AA = {"F", "Y", "W", "H"}
_HBOND_AA = {"S", "T", "N", "Q", "Y", "K", "R", "H", "D", "E", "W"}
_HYDRO_AA = {"A", "V", "L", "I", "M", "F", "W", "C"}


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    # Hot path: called millions of times in the residue x ligand-atom loops
    # (geometry.residue_ligand_contacts, interaction_descriptor). The old
    # `np.linalg.norm(np.asarray(a)-np.asarray(b))` allocated two ndarrays +
    # a norm dispatch PER call (~16s / 8.4M calls in profiling). For a
    # 3-vector np.linalg.norm == sqrt(dx^2+dy^2+dz^2) in float64, so plain
    # Python math is bitwise-identical but ~10x faster. Falls back to numpy
    # for non-length-3 inputs (defensive; not exercised on the hot path).
    try:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)
    except (IndexError, TypeError):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


@dataclass
class ResidueLigandContact:
    residue_index: int
    residue_aa: str
    ligand_atom_id: str
    distance: float
    interaction_type: str
    contact_probability: float = 1.0


def classify_interaction(res_aa: str, atom: LigandAtom, d: float) -> str:
    if d > 5.0:
        return "none"
    if atom.formal_charge < 0 and res_aa in _POS_AA and d <= 4.0:
        return "salt_bridge"
    if atom.formal_charge > 0 and res_aa in _NEG_AA and d <= 4.0:
        return "salt_bridge"
    if atom.aromatic and res_aa in _AROM_AA and d <= 4.5:
        return "aromatic"
    if (atom.is_acceptor or atom.is_donor) and res_aa in _HBOND_AA and d <= 3.5:
        return "hbond"
    if atom.is_hydrophobic and res_aa in _HYDRO_AA and d <= 4.5:
        return "hydrophobic"
    return "vdw"


def residue_ligand_contacts(
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom],
    cutoff: float = 5.0,
) -> List[ResidueLigandContact]:
    contacts: List[ResidueLigandContact] = []
    for res in structure.residues:
        ref = res.sidechain_centroid or res.ca
        best: Optional[ResidueLigandContact] = None
        for atom in ligand_atoms:
            d = dist(ref, atom.coord)
            if d <= cutoff:
                itype = classify_interaction(res.aa, atom, d)
                prob = max(0.0, 1.0 - d / cutoff)
                c = ResidueLigandContact(
                    residue_index=res.index,
                    residue_aa=res.aa,
                    ligand_atom_id=atom.id,
                    distance=round(d, 3),
                    interaction_type=itype,
                    contact_probability=round(prob, 3),
                )
                contacts.append(c)
                if best is None or d < best.distance:
                    best = c
    return contacts


def ligand_proximal_residues(
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom],
    radius: float = 8.0,
) -> List[int]:
    out: List[int] = []
    for res in structure.residues:
        ref = res.sidechain_centroid or res.ca
        if any(dist(ref, a.coord) <= radius for a in ligand_atoms):
            out.append(res.index)
    return out


def ligand_centroid(ligand_atoms: Sequence[LigandAtom]) -> tuple[float, float, float]:
    arr = np.array([a.coord for a in ligand_atoms], dtype=float)
    c = arr.mean(axis=0)
    return (float(c[0]), float(c[1]), float(c[2]))


def catalytic_distances(
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom],
    catalytic_positions: Sequence[int],
) -> Dict[str, float]:
    """Minimum residue-to-ligand distance for each catalytic residue: a
    coarse proxy for catalytic-geometry preservation (spec 14.2)."""
    by_idx = {r.index: r for r in structure.residues}
    out: Dict[str, float] = {}
    for pos in catalytic_positions:
        r = by_idx.get(pos)
        if r is None:
            continue
        ref = r.sidechain_centroid or r.ca
        out[f"cat_{pos}_min_dist"] = round(
            min((dist(ref, a.coord) for a in ligand_atoms), default=99.0), 3
        )
    return out


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    diff = a[:n] - b[:n]
    return float(np.sqrt((diff * diff).sum() / n))
