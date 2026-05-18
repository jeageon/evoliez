"""Relative-vector heterogeneous graph dataset (server-grade, spec §5/§7.2).

For a complex pose: ligand-atom + nearby-residue nodes; ligand-residue and
residue-residue edges carrying r_ij = x_protein - x_ligand, |r_ij|, and the
unit vector. Self-supervised weak labels (no experimental labels):
  contact (ensemble contact frequency), interaction type (geometry),
  residue permissiveness (MSA), native residue (recovery).

Built with numpy only (torch-free) and saved as .npz so datasets can be
generated on the laptop/CI; ``to_torch`` materialises tensors at train time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from evoliez.features.boltz_features import EnsembleContact
from evoliez.features.evolutionary import AA_INDEX, PositionFeature
from evoliez.features.geometry import classify_interaction, dist
from evoliez.types import Complex

_ITYPES = ["hbond", "salt_bridge", "aromatic", "hydrophobic", "vdw", "none"]
_IT_IDX = {t: i for i, t in enumerate(_ITYPES)}


def _ligand_feat(a) -> List[float]:
    return [
        1.0,  # is-ligand flag
        float(a.formal_charge),
        float(a.partial_charge),
        float(a.aromatic),
        float(a.in_ring),
        float(a.is_donor),
        float(a.is_acceptor),
        float(a.is_hydrophobic),
        float(ord(a.element[0]) % 16) / 16.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  # residue slots (zero for ligand)
    ]


def _residue_feat(r, pf: Optional[PositionFeature], dist_lig: float,
                  is_cat: bool) -> List[float]:
    return [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # ligand slots
        pf.conservation_score if pf else 0.0,
        pf.entropy if pf else 0.0,
        pf.gap_frequency if pf else 0.0,
        round(min(dist_lig, 30.0) / 30.0, 4),
        1.0 if is_cat else 0.0,
    ]


NODE_DIM = 14
EDGE_DIM = 7  # r_ij(3) + dist(1) + unit(3)


def build_graph_sample(
    cx: Complex,
    position_features: Sequence[PositionFeature],
    ensemble_contacts: Sequence[EnsembleContact],
    *,
    radius_lr: float = 6.0,
    radius_rr: float = 8.0,
    catalytic_positions: Sequence[int] = (),
) -> Optional[Dict[str, np.ndarray]]:
    pf_by = {f.target_position: f for f in position_features
             if f.target_position is not None}
    cat = set(catalytic_positions)
    lig = cx.ligand.atoms
    if not lig or not cx.structure.residues:
        return None

    from evoliez.features.geometry import ligand_centroid

    lc = ligand_centroid(lig)
    near_res = [
        r for r in cx.structure.residues
        if dist(r.sidechain_centroid or r.ca, lc) <= radius_lr + radius_rr
    ]
    if not near_res:
        return None

    nodes_feat: List[List[float]] = []
    node_type: List[int] = []
    pos: List[list] = []
    lig_idx: Dict[str, int] = {}
    res_idx: Dict[int, int] = {}

    for a in lig:
        lig_idx[a.id] = len(nodes_feat)
        nodes_feat.append(_ligand_feat(a))
        node_type.append(0)
        pos.append(list(a.coord))
    for r in near_res:
        res_idx[r.index] = len(nodes_feat)
        d = dist(r.sidechain_centroid or r.ca, lc)
        nodes_feat.append(
            _residue_feat(r, pf_by.get(r.index), d, r.index in cat)
        )
        node_type.append(1)
        pos.append(list(r.sidechain_centroid or r.ca))

    posn = np.array(pos, dtype=np.float32)
    ei_s, ei_d, ea = [], [], []
    lr_s, lr_d, c_lab, c_w, it_lab, c_mask = [], [], [], [], [], []

    freq = {(e.residue_index, e.ligand_atom_id): e
            for e in ensemble_contacts}

    # ligand-atom -> residue edges
    for a in lig:
        ai = lig_idx[a.id]
        for r in near_res:
            ri = res_idx[r.index]
            d = float(dist(posn[ai], posn[ri]))
            if d <= radius_lr:
                rij = (posn[ri] - posn[ai]).tolist()
                u = (np.array(rij) / (d or 1.0)).tolist()
                attr = rij + [d] + u
                for s, t in ((ai, ri), (ri, ai)):
                    ei_s.append(s)
                    ei_d.append(t)
                    ea.append(attr)
                lr_s.append(ai)
                lr_d.append(ri)
                ec = freq.get((r.index, a.id))
                fval = ec.contact_frequency if ec else 0.0
                lab = 1 if fval >= 0.5 else 0
                msk = 1 if (fval >= 0.5 or fval <= 0.1) else 0  # drop ambiguous
                c_lab.append(lab)
                c_mask.append(msk)
                c_w.append(ec.confidence_weighted_score if ec else 0.1)
                it_lab.append(_IT_IDX.get(
                    classify_interaction(r.aa, a, d), _IT_IDX["vdw"]
                ))

    # residue-residue spatial edges
    for i in range(len(near_res)):
        for j in range(i + 1, len(near_res)):
            ri, rj = res_idx[near_res[i].index], res_idx[near_res[j].index]
            d = float(dist(posn[ri], posn[rj]))
            if d <= radius_rr:
                rij = (posn[rj] - posn[ri]).tolist()
                u = (np.array(rij) / (d or 1.0)).tolist()
                attr = rij + [d] + u
                for s, t in ((ri, rj), (rj, ri)):
                    ei_s.append(s)
                    ei_d.append(t)
                    ea.append(attr)

    perm_lab, native_lab = [], []
    for r in near_res:
        pf = pf_by.get(r.index)
        perm_lab.append(1.0 - (pf.conservation_score if pf else 0.5))
        native_lab.append(AA_INDEX.get(r.aa, 0))

    if not lr_s:
        return None
    return {
        "node_feat": np.array(nodes_feat, dtype=np.float32),
        "node_type": np.array(node_type, dtype=np.int64),
        "pos": posn,
        "edge_index": np.array([ei_s, ei_d], dtype=np.int64),
        "edge_attr": np.array(ea, dtype=np.float32),
        "lr_edge_index": np.array([lr_s, lr_d], dtype=np.int64),
        "contact_label": np.array(c_lab, dtype=np.int64),
        "contact_mask": np.array(c_mask, dtype=np.int64),
        "contact_weight": np.array(c_w, dtype=np.float32),
        "itype_label": np.array(it_lab, dtype=np.int64),
        "perm_label": np.array(perm_lab, dtype=np.float32),
        "native_label": np.array(native_lab, dtype=np.int64),
    }


def save_graph_dataset(
    samples: List[Dict[str, np.ndarray]], out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = []
    for i, s in enumerate(samples):
        p = out_dir / f"graph_{i:05d}.npz"
        np.savez_compressed(p, **s)
        idx.append(p.name)
    (out_dir / "index.txt").write_text("\n".join(idx) + "\n")
    return out_dir


def load_graph_dataset(out_dir: Path) -> List[Dict[str, np.ndarray]]:
    out = []
    for name in (out_dir / "index.txt").read_text().split():
        z = np.load(out_dir / name)
        out.append({k: z[k] for k in z.files})
    return out


def to_torch(sample: Dict[str, np.ndarray]):
    import torch

    return {k: torch.from_numpy(v) for k, v in sample.items()}
