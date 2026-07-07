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
    # Provenance of the pose. "boltz" (the family-geometry teacher) is the
    # default so every existing record keeps its meaning; the multi-engine
    # extension tags augmenting docking poses "gnina"/"diffdock".
    source: str = "boltz"
    # Pre-assigned class for an externally-classified row (multi-engine docking
    # poses already scored against the Boltz consensus). Empty -> let the
    # in-group robust-z statistics classify it (the Boltz default path).
    #   "weak_positive" | "hard_negative" | "strong_negative"
    role: str = ""
    # Per-row training weight for a pre-classified row (only consulted when
    # `role` is set). Boltz rows ignore this and derive the weight from
    # pred_score inside select_poses, as before.
    sample_weight: float = 1.0


@dataclass
class SelectionResult:
    X: np.ndarray  # (n, fingerprint) - interaction fingerprint only, no leakage scalars
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
    # multi-engine docking augmentation (0 unless cfg.multi_engine fed in
    # pre-classified gnina/diffdock rows).
    n_weak_positive: int = 0
    n_hard_negative: int = 0
    consensus: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _robust_z(dists: np.ndarray) -> np.ndarray:
    med = np.median(dists)
    mad = np.median(np.abs(dists - med)) or 1e-6
    return 0.6745 * (dists - med) / mad


def _augment(rec: PoseRecord) -> np.ndarray:
    # Feature matrix = the interaction fingerprint ONLY. msa_membership /
    # identity_to_target / pred_score are NOT features: decoys hardcode them to
    # fixed values distinct from real poses (see _aug_fp), so they leak the
    # real-vs-decoy label perfectly. msa_membership/identity_to_target are kept
    # on PoseRecord as metadata; pred_score is used as the per-row SAMPLE WEIGHT
    # in select_poses (its correct role per the data policy).
    return np.asarray(rec.fingerprint, dtype=float)


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

    # The family consensus is the BOLTZ teacher only: pre-classified docking
    # rows (role set, source gnina/diffdock) were already scored against the
    # Boltz consensus by multi_engine.classify_docking_poses, and must NEVER
    # move the median that defines the family-geometry prior. Boltz-only
    # statistics also keep the robust-z + pred_score normalisation byte-
    # identical when multi_engine is off (every record then has role="").
    boltz_records = [r for r in records if not r.role]
    pre_records = [r for r in records if r.role]
    stat_records = boltz_records or records  # degenerate: no Boltz rows at all

    fps = np.vstack([r.fingerprint for r in stat_records])
    consensus = np.median(fps, axis=0)
    dists = np.linalg.norm(fps - consensus, axis=1)
    z = _robust_z(dists)

    preds = np.array([r.pred_score for r in stat_records], dtype=float)
    lo, hi = float(preds.min()), float(preds.max())
    rng_span = (hi - lo) or 1.0

    rows, soft, cls, wts = [], [], [], []
    n_pos = n_alt = n_out = 0
    for rec, zi in zip(stat_records, z):
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

    # Pre-classified multi-engine docking rows: the role already encodes the
    # relationship to the Boltz consensus, so map it straight to a label +
    # carried sample weight (no robust-z reclassification).
    #   weak_positive  -> positive-ish (class 2, low weight): a docking pose that
    #                     AGREES with the family geometry — mild corroboration.
    #   hard_negative  -> negative (class 1, soft 0.15): scores well yet the
    #                     contact pattern DISAGREES — the valuable hard signal.
    #   strong_negative-> negative (class 0): clash / inverted / catalytic broken.
    n_weak_pos = n_hard_neg = 0
    for rec in pre_records:
        rows.append(_augment(rec))
        wts.append(float(rec.sample_weight))
        if rec.role == "weak_positive":
            soft.append(0.55); cls.append(2); n_weak_pos += 1
        elif rec.role == "hard_negative":
            soft.append(0.15); cls.append(1); n_hard_neg += 1
        else:  # strong_negative (or any unknown role) -> hard negative class 0
            soft.append(0.05); cls.append(0); n_hard_neg += 1

    # Decoys are generated from the BOLTZ groups + consensus only (docking rows
    # carry no homolog group and must not spawn synthetic per-group decoys).
    groups = sorted({r.group_id for r in stat_records})
    fp_dim = stat_records[0].fingerprint.shape[0]
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
        return SelectionResult(np.zeros((0, fp_dim)), np.zeros(0),
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
        n_decoy=n_decoy, n_hard_decoy=n_hard,
        n_weak_positive=n_weak_pos, n_hard_negative=n_hard_neg,
        consensus=consensus,
    )


def _aug_fp(fp: np.ndarray) -> np.ndarray:
    # decoy feature row = the (perturbed) fingerprint only; no leakage scalars.
    return np.asarray(fp, dtype=float)
