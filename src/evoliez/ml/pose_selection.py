"""Statistical selection of significant docking poses (data augmentation).

The homolog pool is small, so every docking pose is a candidate training
example. We keep poses whose ligand-atom interaction-distance fingerprint sits
near the **family consensus** (positives) and treat statistical outliers +
synthetic decoys as negatives. Robust statistics (median + MAD) are used so a
few wild poses do not move the consensus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from evoliez.utils.seeds import derive_seed


@dataclass
class PoseRecord:
    group_id: str  # homolog id (poses of the same homolog share this)
    fingerprint: np.ndarray
    msa_membership: float = 1.0
    identity_to_target: float = 0.0
    pred_score: float = 0.0  # per-pose docking/structure prediction score


@dataclass
class SelectionResult:
    X: np.ndarray  # (n, fingerprint + 3 extra features)
    y: np.ndarray  # binary back-compat: 1 = consensus/alternative, 0 = neg
    # Per-row training weight = Boltz pose reliability (spec: use Boltz score
    # as a SAMPLE WEIGHT, never as a label).
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # soft target in [0,1] + 4-class code (expert review #3):
    # 0 decoy/impossible | 1 outlier(weak neg) | 2 alternative/uncertain |
    # 3 consensus(strong pos). Alternative poses are NOT auto-negatives.
    soft_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pose_class: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_positive: int = 0
    n_alternative: int = 0
    n_outlier: int = 0
    n_decoy: int = 0
    n_hard_decoy: int = 0
    consensus: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _robust_z(dists: np.ndarray) -> np.ndarray:
    med = np.median(dists)
    mad = np.median(np.abs(dists - med)) or 1e-6
    return 0.6745 * (dists - med) / mad


def _augment(rec: PoseRecord) -> np.ndarray:
    return np.concatenate(
        [rec.fingerprint,
         np.array([rec.msa_membership, rec.identity_to_target, rec.pred_score],
                  dtype=float)]
    )


def select_poses(
    records: List[PoseRecord],
    *,
    select_z: float = 2.5,
    outlier_z: float = 4.0,
    min_decoys_per_group: int = 4,
    seed: int = 1234,
    keep_alternative_band: bool = True,
    alternative_weight: float = 0.3,
    hard_decoys_per_group: int = 2,
) -> SelectionResult:
    if not records:
        return SelectionResult(np.zeros((0, 0)), np.zeros(0))

    fps = np.vstack([r.fingerprint for r in records])
    consensus = np.median(fps, axis=0)
    dists = np.linalg.norm(fps - consensus, axis=1)
    z = _robust_z(dists)

    preds = np.array([r.pred_score for r in records], dtype=float)
    lo, hi = float(preds.min()), float(preds.max())
    rng_span = (hi - lo) or 1.0

    rows, soft, cls, wts = [], [], [], []
    n_pos = n_alt = n_out = 0
    for rec, zi in zip(records, z):
        reln = (rec.pred_score - lo) / rng_span
        if zi <= select_z:  # class 3: consensus, strong positive
            rows.append(_augment(rec))
            soft.append(round(0.85 + 0.15 * reln, 4))
            cls.append(3)
            wts.append(0.2 + 0.8 * reln)
            n_pos += 1
        elif zi >= outlier_z:  # class 1: real-but-far homolog pose, weak neg
            rows.append(_augment(rec))
            soft.append(0.2)
            cls.append(1)
            wts.append(0.6)
            n_out += 1
        elif keep_alternative_band:  # class 2: alternative / uncertain
            rows.append(_augment(rec))
            soft.append(0.5)            # NOT an auto-negative
            cls.append(2)
            wts.append(alternative_weight)
            n_alt += 1

    groups = sorted({r.group_id for r in records})
    fp_dim = records[0].fingerprint.shape[0]
    n_decoy = n_hard = 0
    for g in groups:
        # easy / chemically-impossible decoys: heavily perturbed geometry
        for d in range(min_decoys_per_group):
            h = derive_seed(seed, g, "decoy", str(d))
            rng = np.random.RandomState(h % (2**32 - 1))
            fp = consensus * rng.uniform(0.6, 1.6, size=fp_dim) + rng.normal(
                0, 0.4, size=fp_dim
            )
            rows.append(_aug_fp(fp))
            soft.append(0.0)
            cls.append(0)
            wts.append(1.0)
            n_decoy += 1
        # hard "plausible but wrong-pose" decoys: keep overall shape, corrupt
        # the contact pattern (right magnitude, wrong arrangement)
        for d in range(hard_decoys_per_group):
            h = derive_seed(seed, g, "hard", str(d))
            rng = np.random.RandomState(h % (2**32 - 1))
            fp = consensus.copy()
            perm = rng.permutation(fp_dim)
            fp = 0.6 * fp + 0.4 * fp[perm]  # shuffled contacts, same scale
            rows.append(_aug_fp(fp))
            soft.append(0.1)
            cls.append(0)
            wts.append(1.0)
            n_hard += 1

    if not rows:
        return SelectionResult(np.zeros((0, fp_dim + 3)), np.zeros(0),
                               consensus=consensus)
    X = np.vstack(rows)
    soft_y = np.array(soft, dtype=float)
    pose_class = np.array(cls, dtype=int)
    # binary back-compat: consensus + alternative are positive-ish (not hard
    # negatives); outlier + decoys are negative.
    y = (pose_class >= 2).astype(int)
    w = np.array(wts, dtype=float)
    return SelectionResult(
        X=X, y=y, weights=w, soft_y=soft_y, pose_class=pose_class,
        n_positive=n_pos, n_alternative=n_alt, n_outlier=n_out,
        n_decoy=n_decoy, n_hard_decoy=n_hard, consensus=consensus,
    )


def _aug_fp(fp: np.ndarray) -> np.ndarray:
    return np.concatenate([fp, np.array([0.0, 0.0, -1.0])])
