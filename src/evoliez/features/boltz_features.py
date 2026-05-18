"""Boltz-derived features (user spec sections 2-4, 8).

Everything here is a FEATURE / uncertainty weight - never a supervised label
(see ml/labels.py). Highest-priority feature is ensemble contact frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

from evoliez.features.geometry import dist, ligand_centroid
from evoliez.types import BoltzSample, LigandAtom, ProteinStructure


def _norm_plddt(v: float) -> float:
    return v / 100.0 if v > 1.5 else v


def confidence_weighted_contact(
    contact_prob: float, ligand_iptm: float, local_plddt: float, pde: float
) -> float:
    """edge_weight = contact_score x ligand_iptm x local_pLDDT x inverse_PDE
    (user §3.2). PDE is in Å (lower = better) -> inverse via 1/(1+PDE)."""
    return round(
        contact_prob
        * max(0.0, ligand_iptm)
        * max(0.0, _norm_plddt(local_plddt))
        * (1.0 / (1.0 + max(0.0, pde))),
        5,
    )


@dataclass
class EnsembleContact:
    residue_index: int
    ligand_atom_id: str
    contact_frequency: float  # fraction of samples in contact (priority #1)
    mean_distance: float
    confidence_weighted_score: float


def ensemble_contacts(
    structure: ProteinStructure,
    samples: Sequence[BoltzSample],
    *,
    cutoff: float = 6.0,
    ligand_iptm: float = 1.0,
) -> List[EnsembleContact]:
    """Per (residue, ligand atom): contact frequency across Boltz samples +
    a confidence-weighted contact score (user §2.3, §4.2, §8 priority 1)."""
    if not samples or not structure.residues:
        return []
    res_pts = [(r.index, r.sidechain_centroid or r.ca, r.plddt)
               for r in structure.residues]
    n = len(samples)
    counts: Dict[Tuple[int, str], int] = {}
    dsum: Dict[Tuple[int, str], float] = {}
    for smp in samples:
        for atom in smp.ligand_atoms:
            for (ridx, pt, _pl) in res_pts:
                d = dist(pt, atom.coord)
                if d <= cutoff:
                    key = (ridx, atom.id)
                    counts[key] = counts.get(key, 0) + 1
                    dsum[key] = dsum.get(key, 0.0) + d
    out: List[EnsembleContact] = []
    plddt_by_res = {r.index: r.plddt for r in structure.residues}
    for key, c in counts.items():
        ridx, aid = key
        freq = c / n
        md = dsum[key] / c
        cw = confidence_weighted_contact(
            contact_prob=freq,
            ligand_iptm=ligand_iptm,
            local_plddt=plddt_by_res.get(ridx, 70.0),
            pde=2.0,
        )
        out.append(EnsembleContact(ridx, aid, round(freq, 4),
                                   round(md, 3), cw))
    out.sort(key=lambda e: (-e.contact_frequency, e.mean_distance))
    return out


def pose_consensus(samples: Sequence[BoltzSample]) -> dict:
    """Ensemble agreement (user §4.1): ligand centroid / RMSD variance,
    top-vs-rest divergence."""
    if not samples:
        return {"n_samples": 0, "centroid_var": 0.0,
                "top_vs_rest_rmsd": 0.0, "ligand_rmsd_std": 0.0}
    cents = []
    for s in samples:
        if s.ligand_atoms:
            cents.append(ligand_centroid(s.ligand_atoms))
    if not cents:
        return {"n_samples": len(samples), "centroid_var": 0.0,
                "top_vs_rest_rmsd": 0.0, "ligand_rmsd_std": 0.0}
    cx = mean(c[0] for c in cents)
    cy = mean(c[1] for c in cents)
    cz = mean(c[2] for c in cents)
    cvar = mean(
        (c[0] - cx) ** 2 + (c[1] - cy) ** 2 + (c[2] - cz) ** 2 for c in cents
    )
    top = samples[0].ligand_atoms
    rest_rmsd = []
    for s in samples[1:]:
        if top and s.ligand_atoms and len(s.ligand_atoms) == len(top):
            d2 = mean(
                dist(a.coord, b.coord) ** 2
                for a, b in zip(top, s.ligand_atoms)
            )
            rest_rmsd.append(d2 ** 0.5)
    return {
        "n_samples": len(samples),
        "centroid_var": round(float(cvar), 4),
        "top_vs_rest_rmsd": round(mean(rest_rmsd), 4) if rest_rmsd else 0.0,
        "ligand_rmsd_std": round(pstdev(rest_rmsd), 4)
        if len(rest_rmsd) > 1 else 0.0,
    }


def pocket_plddt(
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom],
    *,
    radius: float = 8.0,
) -> float:
    if not ligand_atoms or not structure.residues:
        return 0.0
    lc = ligand_centroid(ligand_atoms)
    near = [
        _norm_plddt(r.plddt)
        for r in structure.residues
        if dist(r.sidechain_centroid or r.ca, lc) <= radius
    ]
    return round(mean(near), 4) if near else 0.0


def catalytic_plddt(
    structure: ProteinStructure, catalytic_positions: Sequence[int]
) -> float:
    by = {r.index: r for r in structure.residues}
    vals = [_norm_plddt(by[p].plddt) for p in catalytic_positions if p in by]
    return round(mean(vals), 4) if vals else 0.0
