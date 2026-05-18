"""Relative-vector heterogeneous graph dataset (confidence-aware).

For a complex pose: ligand-atom + nearby-residue nodes; ligand-residue and
residue-residue edges carrying r_ij, |r_ij|, unit vector, plus a per-edge
**confidence weight** (contact_freq x normalized_pLDDT x ligand_iptm x
exp(-ipDE/10)). Residue nodes carry pLDDT-derived and disorder features.

pLDDT/PAE/PDE/disorder are FEATURES / edge weights, NEVER labels. Active-site
low-pLDDT loops are kept (flagged uncertain + down-weighted), not dropped.
Self-supervised weak labels only.

numpy-only (torch-free) build + .npz export; ``to_torch`` at train time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from evoliez.adapters.disorder import DisorderTrack
from evoliez.features.boltz_features import EnsembleContact, edge_confidence
from evoliez.features.confidence import ResidueConfidence
from evoliez.features.evolutionary import AA_INDEX, PositionFeature
from evoliez.features.geometry import classify_interaction, dist, ligand_centroid
from evoliez.types import Complex

_ITYPES = ["hbond", "salt_bridge", "aromatic", "hydrophobic", "vdw", "none"]
_IT_IDX = {t: i for i, t in enumerate(_ITYPES)}

# ligand slots(9) + evolutionary(5) + confidence(7) = 21
NODE_DIM = 21
EDGE_DIM = 7  # r_ij(3) + dist(1) + unit(3)
_ZERO_CONF = [0.0] * 7


def _ligand_feat(a) -> List[float]:
    return [
        1.0, float(a.formal_charge), float(a.partial_charge),
        float(a.aromatic), float(a.in_ring), float(a.is_donor),
        float(a.is_acceptor), float(a.is_hydrophobic),
        float(ord(a.element[0]) % 16) / 16.0,
        0.0, 0.0, 0.0, 0.0, 0.0,        # evolutionary slots
        *_ZERO_CONF,                     # confidence slots
    ]


def _residue_feat(r, pf, dlig, is_cat, conf, dis_iup, dis_lc) -> List[float]:
    return [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # ligand slots
        pf.conservation_score if pf else 0.0,
        pf.entropy if pf else 0.0,
        pf.gap_frequency if pf else 0.0,
        round(min(dlig, 30.0) / 30.0, 4),
        1.0 if is_cat else 0.0,
        # confidence block (user §2A): pLDDT + disorder
        conf.plddt_norm if conf else 0.7,
        (conf.plddt_bin / 3.0) if conf else 0.66,
        conf.mean_w7 if conf else 0.7,
        round(min(conf.low_seg_len, 30) / 30.0, 4) if conf else 0.0,
        conf.frac_low_nearby if conf else 0.0,
        float(dis_iup),
        float(dis_lc),
    ]


def build_graph_sample(
    cx: Complex,
    position_features: Sequence[PositionFeature],
    ensemble_contacts: Sequence[EnsembleContact],
    *,
    radius_lr: float = 6.0,
    radius_rr: float = 8.0,
    catalytic_positions: Sequence[int] = (),
    residue_confidence: Optional[Sequence[ResidueConfidence]] = None,
    disorder: Optional[DisorderTrack] = None,
    ligand_iptm: float = 1.0,
    complex_ipde: float = 2.0,
    low_plddt_cutoff: float = 50.0,
    drop_far_low_plddt: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    pf_by = {f.target_position: f for f in position_features
             if f.target_position is not None}
    conf_by = {c.index: c for c in (residue_confidence or [])}
    cat = set(catalytic_positions)
    lig = cx.ligand.atoms
    if not lig or not cx.structure.residues:
        return None

    lc = ligand_centroid(lig)
    iup = disorder.iupred if disorder else []
    lcx = disorder.low_complexity if disorder else []

    near_res = []
    for ri, r in enumerate(cx.structure.residues):
        d = dist(r.sidechain_centroid or r.ca, lc)
        if d > radius_lr + radius_rr:
            continue
        conf = conf_by.get(r.index)
        plddt = conf.plddt_norm if conf else 0.7
        # low-pLDDT policy: drop only if far from ligand AND not catalytic;
        # near-pocket / active-site low-pLDDT loops are kept (down-weighted).
        if (drop_far_low_plddt and plddt < low_plddt_cutoff / 100.0
                and d > 10.0 and r.index not in cat):
            continue
        near_res.append((ri, r, d, conf))
    if not near_res:
        return None

    nodes_feat: List[List[float]] = []
    node_type: List[int] = []
    pos: List[list] = []
    plddt_norm: List[float] = []
    lig_idx: Dict[str, int] = {}
    res_idx: Dict[int, int] = {}

    for a in lig:
        lig_idx[a.id] = len(nodes_feat)
        nodes_feat.append(_ligand_feat(a))
        node_type.append(0)
        pos.append(list(a.coord))
        plddt_norm.append(1.0)  # trust the ligand pose coordinates
    for (ri, r, d, conf) in near_res:
        res_idx[r.index] = len(nodes_feat)
        di = iup[ri] if ri < len(iup) else 0.0
        dl = lcx[ri] if ri < len(lcx) else 0
        nodes_feat.append(
            _residue_feat(r, pf_by.get(r.index), d, r.index in cat,
                          conf, di, dl)
        )
        node_type.append(1)
        pos.append(list(r.sidechain_centroid or r.ca))
        plddt_norm.append(conf.plddt_norm if conf else 0.7)

    posn = np.array(pos, dtype=np.float32)
    ei_s, ei_d, ea, e_conf = [], [], [], []
    lr_s, lr_d, c_lab, c_w, it_lab, c_mask = [], [], [], [], [], []
    freq = {(e.residue_index, e.ligand_atom_id): e for e in ensemble_contacts}

    for a in lig:
        ai = lig_idx[a.id]
        for (_ri, r, _d, conf) in near_res:
            rj = res_idx[r.index]
            dd = float(dist(posn[ai], posn[rj]))
            if dd <= radius_lr:
                rij = (posn[rj] - posn[ai]).tolist()
                u = (np.array(rij) / (dd or 1.0)).tolist()
                attr = rij + [dd] + u
                ec = freq.get((r.index, a.id))
                fval = ec.contact_frequency if ec else 0.0
                w = edge_confidence(
                    fval, conf.plddt_norm if conf else 0.7,
                    ligand_iptm, complex_ipde,
                )
                for s, t in ((ai, rj), (rj, ai)):
                    ei_s.append(s)
                    ei_d.append(t)
                    ea.append(attr)
                    e_conf.append(w)
                lr_s.append(ai)
                lr_d.append(rj)
                lab = 1 if fval >= 0.5 else 0
                msk = 1 if (fval >= 0.5 or fval <= 0.1) else 0
                c_lab.append(lab)
                c_mask.append(msk)
                c_w.append(ec.confidence_weighted_score if ec else 0.1)
                it_lab.append(_IT_IDX.get(
                    classify_interaction(r.aa, a, dd), _IT_IDX["vdw"]
                ))

    nr = [x[1] for x in near_res]
    for i in range(len(nr)):
        for j in range(i + 1, len(nr)):
            ri2, rj2 = res_idx[nr[i].index], res_idx[nr[j].index]
            dd = float(dist(posn[ri2], posn[rj2]))
            if dd <= radius_rr:
                rij = (posn[rj2] - posn[ri2]).tolist()
                u = (np.array(rij) / (dd or 1.0)).tolist()
                attr = rij + [dd] + u
                w = round(min(plddt_norm[ri2], plddt_norm[rj2]), 5)
                for s, t in ((ri2, rj2), (rj2, ri2)):
                    ei_s.append(s)
                    ei_d.append(t)
                    ea.append(attr)
                    e_conf.append(w)

    perm_lab, native_lab, relia_lab, risk_lab = [], [], [], []
    for (_ri, r, _d, conf) in near_res:
        pf = pf_by.get(r.index)
        pn = conf.plddt_norm if conf else 0.7
        di = iup[_ri] if _ri < len(iup) else 0.0
        perm_lab.append(1.0 - (pf.conservation_score if pf else 0.5))
        native_lab.append(AA_INDEX.get(r.aa, 0))
        # Task F: coordinate reliability pseudo-label (pLDDT + low disorder)
        relia_lab.append(1 if (pn >= 0.7 and di < 0.5) else 0)
        # Task G: flexible/uncertain-pocket risk (regression, 0..1)
        risk_lab.append(round(min(1.0, 0.6 * (1.0 - pn) + 0.4 * di), 4))

    if not lr_s:
        return None
    return {
        "node_feat": np.array(nodes_feat, dtype=np.float32),
        "node_type": np.array(node_type, dtype=np.int64),
        "pos": posn,
        "plddt_norm": np.array(plddt_norm, dtype=np.float32),
        "edge_index": np.array([ei_s, ei_d], dtype=np.int64),
        "edge_attr": np.array(ea, dtype=np.float32),
        "edge_conf": np.array(e_conf, dtype=np.float32),
        "lr_edge_index": np.array([lr_s, lr_d], dtype=np.int64),
        "contact_label": np.array(c_lab, dtype=np.int64),
        "contact_mask": np.array(c_mask, dtype=np.int64),
        "contact_weight": np.array(c_w, dtype=np.float32),
        "itype_label": np.array(it_lab, dtype=np.int64),
        "perm_label": np.array(perm_lab, dtype=np.float32),
        "native_label": np.array(native_lab, dtype=np.int64),
        "relia_label": np.array(relia_lab, dtype=np.int64),
        "risk_label": np.array(risk_lab, dtype=np.float32),
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
