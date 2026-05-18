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
    y: np.ndarray  # 1 = family-consensus pose, 0 = outlier/decoy
    # Per-row training weight = Boltz pose reliability (spec: use Boltz score
    # as a SAMPLE WEIGHT, never as a label).
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_positive: int = 0
    n_outlier: int = 0
    n_decoy: int = 0
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
) -> SelectionResult:
    if not records:
        return SelectionResult(np.zeros((0, 0)), np.zeros(0))

    fps = np.vstack([r.fingerprint for r in records])
    consensus = np.median(fps, axis=0)
    dists = np.linalg.norm(fps - consensus, axis=1)
    z = _robust_z(dists)

    # positive sample weight = Boltz pose reliability (normalised pred_score)
    preds = np.array([r.pred_score for r in records], dtype=float)
    lo, hi = float(preds.min()), float(preds.max())
    rng_span = (hi - lo) or 1.0

    Xp, Xn, Wp, Wn = [], [], [], []
    for rec, zi in zip(records, z):
        if zi <= select_z:
            Xp.append(_augment(rec))
            Wp.append(0.2 + 0.8 * (rec.pred_score - lo) / rng_span)
        elif zi >= outlier_z:
            Xn.append(_augment(rec))
            Wn.append(1.0)
        # in-between: ambiguous, dropped to keep classes clean

    n_outlier = len(Xn)

    # Synthetic decoys: strongly perturbed fingerprints (ligand displaced /
    # interactions lost) so the classifier always has negatives.
    groups = sorted({r.group_id for r in records})
    fp_dim = records[0].fingerprint.shape[0]
    n_decoy = 0
    for g in groups:
        for d in range(min_decoys_per_group):
            h = derive_seed(seed, g, "decoy", str(d))
            rng = np.random.RandomState(h % (2**32 - 1))
            base = consensus.copy()
            # inflate distance-like features, zero-out contact fractions
            noise = rng.uniform(0.6, 1.6, size=fp_dim)
            decoy_fp = base * noise + rng.normal(0, 0.4, size=fp_dim)
            rec = PoseRecord(
                group_id=g, fingerprint=decoy_fp,
                msa_membership=0.0, identity_to_target=0.0, pred_score=-1.0,
            )
            Xn.append(_augment(rec))
            Wn.append(1.0)
            n_decoy += 1

    X = np.vstack(Xp + Xn) if (Xp or Xn) else np.zeros((0, fp_dim + 3))
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))]).astype(int)
    w = np.concatenate([np.array(Wp), np.array(Wn)]) if (Xp or Xn) else np.zeros(0)
    return SelectionResult(
        X=X, y=y, weights=w, n_positive=len(Xp), n_outlier=n_outlier,
        n_decoy=n_decoy, consensus=consensus,
    )
