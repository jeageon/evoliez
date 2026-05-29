"""Self-supervised family interaction-geometry classifier.

Trained on (representative homolog, docking pose) examples: positives =
family-consensus poses, negatives = statistical-outlier poses + synthetic
decoys (see ``pose_selection``). Features per row = ligand-atom
interaction-distance fingerprint + [msa_membership, identity_to_target,
per-pose prediction score].

Backends, auto-selected by availability: xgboost -> sklearn logistic ->
dependency-free heuristic (distance-to-consensus logistic). All deterministic.
The [msa_membership, identity_to_target, per-pose prediction score] tail is
consumed by the trained backends only; the heuristic fallback scores on
fingerprint distance alone (see ``score_vector``).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from evoliez.features.interaction_descriptor import complex_fingerprint
from evoliez.logging_utils import get_logger
from evoliez.ml.pose_selection import SelectionResult
from evoliez.types import LigandAtom, ProteinStructure

log = get_logger("evoliez.interaction_model")


def _n_threads() -> int:
    """Shared-server thread cap (see evoliez.__init__._limit_thread_pools).

    xgboost ignores OMP_NUM_THREADS for its own pool unless n_jobs is set, and
    s08 calls predict_proba on 1 row per candidate (x39) - an unbounded pool
    there thrashes the shared box just as badly as training does.
    """
    import os

    try:
        n = int(os.environ.get("EVOLIEZ_NUM_THREADS", "") or 0)
    except ValueError:
        n = 0
    return n if n > 0 else min(4, os.cpu_count() or 4)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class InteractionModel:
    kind: str = "heuristic"
    fp_dim: int = 0
    consensus: Optional[np.ndarray] = None
    thr: float = 0.0
    scale: float = 1.0
    _est: object = None  # sklearn/xgboost estimator when kind != heuristic
    cutoff: float = 6.0
    k_nearest: int = 6
    n_bins: int = 8

    # ------------------------------------------------------------------ #
    def fit(self, sel: SelectionResult) -> "InteractionModel":
        self.fp_dim = int(sel.consensus.shape[0])
        self.consensus = sel.consensus.astype(float)
        if sel.X.shape[0] >= 8 and len(set(sel.y.tolist())) == 2:
            if self._fit_xgboost(sel) or self._fit_logistic(sel):
                return self
        self._fit_heuristic(sel)
        return self

    def _fit_xgboost(self, sel: SelectionResult) -> bool:
        try:
            import xgboost as xgb
        except Exception:
            return False
        m = xgb.XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.9, eval_metric="logloss", n_jobs=_n_threads(),
        )
        sw = sel.weights if sel.weights.size == sel.X.shape[0] else None
        m.fit(sel.X, sel.y, sample_weight=sw)
        self._est, self.kind = m, "xgboost"
        log.info("interaction model: xgboost (%d rows)", sel.X.shape[0])
        return True

    def _fit_logistic(self, sel: SelectionResult) -> bool:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
        except Exception:
            return False
        m = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500)
        )
        sw = sel.weights if sel.weights.size == sel.X.shape[0] else None
        m.fit(sel.X, sel.y, logisticregression__sample_weight=sw)
        self._est, self.kind = m, "logistic"
        log.info("interaction model: logistic (%d rows)", sel.X.shape[0])
        return True

    def _fit_heuristic(self, sel: SelectionResult) -> None:
        self.kind = "heuristic"
        if sel.X.shape[0] == 0:
            self.thr, self.scale = self.cutoff, 1.0
            return
        fp = sel.X[:, : self.fp_dim]
        d = np.linalg.norm(fp - self.consensus, axis=1)
        pos = d[sel.y == 1]
        neg = d[sel.y == 0]
        mp = float(pos.mean()) if pos.size else float(d.min())
        mn = float(neg.mean()) if neg.size else float(d.max())
        self.thr = (mp + mn) / 2.0
        # Sigmoid width. The old code did `max(1e-3, (mn-mp)/4 or std or 1)`:
        # when labels are degenerate (mn <= mp) the negative gap is *truthy*, so
        # the `or` never falls through and scale collapses to 1e-3 -> an almost
        # hard step that saturates every score at ~0 or ~1. We instead:
        #   * fall back to the overall distance spread when the pos/neg gap is
        #     non-positive or vanishing, and
        #   * never let scale fall below a fraction of that spread, so a
        #     realistic single-mutant perturbation moves along the sigmoid
        #     instead of pinning at 0.98 (gap/4 alone is far too steep when the
        #     positive cluster is tight).
        spread = float(d.std()) or float(np.median(np.abs(d - np.median(d)))) \
            or float(d.mean()) or 1.0
        gap = (mn - mp) / 4.0
        if mn <= mp:
            log.warning(
                "interaction model: degenerate pose labels (neg mean %.3f <= "
                "pos mean %.3f); using distance spread %.3f for sigmoid scale "
                "instead of a collapsed hard step", mn, mp, spread,
            )
        base = gap if gap > 1e-3 else spread
        # Widen so the transition spans the realistic distance distribution and
        # does not saturate at the extremes (tie to the full spread, not just
        # the often-tiny pos/neg gap).
        self.scale = max(base, 0.5 * spread, 1e-3)
        log.info(
            "interaction model: heuristic (thr=%.3f scale=%.3f, %d rows)",
            self.thr, self.scale, sel.X.shape[0],
        )

    # ------------------------------------------------------------------ #
    def score_vector(self, x: np.ndarray) -> float:
        """P(family-consistent) for one augmented feature row.

        Trained backends (xgboost / logistic) use the *full* augmented row,
        including the [msa_membership, identity_to_target, pred_score] tail. The
        dependency-free **heuristic** fallback, however, is purely a
        fingerprint-distance-to-consensus logistic: it reads only the first
        ``fp_dim`` columns and ignores the three extra features by design
        (there is no labelled signal to weight them against without a fitted
        estimator). Callers needing those features to influence the score must
        be on a trained backend.
        """
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if self.kind in ("xgboost", "logistic") and self._est is not None:
            try:
                return float(self._est.predict_proba(x)[0, 1])
            except Exception:
                pass
        fp = x[0, : self.fp_dim]
        d = float(np.linalg.norm(fp - self.consensus))
        return float(_sigmoid(-(d - self.thr) / self.scale))

    def score_complex(
        self,
        structure: ProteinStructure,
        ligand_atoms: Sequence[LigandAtom],
        *,
        msa_membership: float = 0.0,
        identity_to_target: float = 1.0,
        pred_score: float = 0.0,
    ) -> float:
        fp = complex_fingerprint(
            structure, ligand_atoms, cutoff=self.cutoff,
            k_nearest=self.k_nearest, n_bins=self.n_bins,
        )
        x = np.concatenate(
            [fp, np.array([msa_membership, identity_to_target, pred_score])]
        )
        return round(self.score_vector(x), 4)

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "kind": self.kind, "fp_dim": self.fp_dim,
            "consensus": self.consensus.tolist()
            if self.consensus is not None else [],
            "thr": self.thr, "scale": self.scale,
            "cutoff": self.cutoff, "k_nearest": self.k_nearest,
            "n_bins": self.n_bins,
        }
        path.write_text(json.dumps(meta, indent=2))
        if self._est is not None:
            with open(path.with_suffix(".pkl"), "wb") as fh:
                pickle.dump(self._est, fh)

    @classmethod
    def load(cls, path: Path) -> "InteractionModel":
        meta = json.loads(path.read_text())
        m = cls(
            kind=meta["kind"], fp_dim=meta["fp_dim"],
            consensus=np.array(meta["consensus"], dtype=float),
            thr=meta["thr"], scale=meta["scale"], cutoff=meta["cutoff"],
            k_nearest=meta["k_nearest"], n_bins=meta["n_bins"],
        )
        pkl = path.with_suffix(".pkl")
        if pkl.exists():
            with open(pkl, "rb") as fh:
                m._est = pickle.load(fh)
        return m
