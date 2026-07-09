"""MD-free functional-state triage: cheap/expensive contract + learnability smoke test.

ROADMAP_V2 Phase F. The v2 ML must answer ONE question:

    can CHEAP, pre-validation features predict the EXPENSIVE functional-state label
    (pose_gate verdict / ΔNAC sign / MD pass) that the heavy stages compute downstream?

If the cheap feature set carries no signal for `reference_like`, an ML triage cannot
beat the deterministic pose_gate and should not be built. So this module does two
things, both mechanically:

1. Declares the cheap/expensive split as an ALLOWLIST (cheap, may be a model input) and
   a DENYLIST (expensive, label-only). `assert_no_leakage` makes it impossible to train
   on the very quantity the model is meant to predict (the leakage failure mode: feeding
   GNINA/DiffDock/Boltz/pose-ensemble agreement in as a feature reduces the ML to a
   redundant copy of the gate).

2. Runs a small, dependency-light learnability probe on a run's provenance JSONs:
   per-feature univariate AUC, a leave-one-out logistic AUC over the cheap block, and the
   baseline AUC of the CURRENT contact-fingerprint `ml_score` against the same label. The
   new model only earns its place if cheap features beat that baseline.

Pure-numpy (no sklearn/xgboost) so it runs in the light local env. Designed to read the
provenance directory of a finished run:
    runs/<name>/reports/provenance/{generated,reranked,validated}_candidates.json
                                   /md_candidates.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# The cheap/expensive contract.  CHEAP = computable BEFORE any docking/MD, for every
# candidate (sequence / MSA / single static anchored geometry / chemistry / ThermoMPNN).
# Anything below `EXPENSIVE_FIELDS` is a LABEL source only and must never become a feature.
# --------------------------------------------------------------------------------------

# field -> provenance file it is read from (documentation + debugging)
CHEAP_FEATURES: Dict[str, str] = {
    # MSA / sequence (generated_candidates.json, s07)
    "conservation": "generated",
    "gap_freq": "generated",
    "msa_freq": "generated",
    "msa_permissiveness": "generated",
    "esm_permissive": "generated",
    "subfamily_specific": "generated",
    # mutation identity / static pocket (generated_candidates.json, s07)
    "n_mutations": "generated",
    "n_contacts": "generated",
    "ligandmpnn_logp": "generated",
    # ThermoMPNN — fast, MD-free (validated_candidates.json, s09)
    "ddg_fold": "validated",
    "stability_score": "validated",
}
# reaction-center / static-geometry distances are cheap; matched by prefix so any
# target's role set (design_ligand / catalytic / metal / water / substrate) is covered.
CHEAP_PREFIXES: Tuple[str, ...] = ("distance_to_",)

# LABEL sources — every quantity that requires docking, redock, Boltz, MD, NAC, RBFE, or a
# pose ensemble.  Forbidden as model inputs.  `ml_score` is the CURRENT model's output and is
# the baseline-to-beat, not a feature.
EXPENSIVE_FIELDS = {
    "pose_gate", "nac", "nac_delta_vs_wt", "nac_occupancy", "nac_status",
    "passed", "binding_dg", "binding_dg_status", "ligand_rmsd_mean", "pocket_rmsd_mean",
    "catalytic_distance_mean", "md_instability", "energy_drift", "hbond_occupancy",
    "md_lite_score", "md_lite_status", "rbfe", "gate_stack", "alternative_pose_boltz",
    "redock", "redocking_consistency", "d_ligand_iptm", "docking_uncertainty",
    "interaction_gain", "family_interaction_score", "ml_score",
    "catalytic_geometry_penalty",
}


# markers that make a prefix-matched distance NOT cheap: a distance measured AFTER docking /
# MD / a pose search is a label-derived quantity, not a static-geometry prior. `distance_to_`
# was the ONE prefix that bypassed the fail-safe unknown check (Fable review), so a
# `distance_to_catalytic_after_md`-style key could leak in as a feature.
_EXPENSIVE_DISTANCE_MARKERS = (
    "md", "nac", "pose", "dock", "redock", "after", "post", "dynamic",
    "rmsf", "rmsd", "drift", "traj", "boltz", "gnina", "diffdock",
    # trajectory / statistical aggregation suffixes — a distance summarised over an MD run is a
    # label, not a static prior (Fable verification: distance_to_catalytic_mean slipped through).
    "_mean", "_avg", "_median", "_min", "_max", "_std", "_var", "_final",
    "_prod", "_ns", "_ps", "_frame", "_equilib", "_occupancy",
)


def is_cheap(field_name: str) -> bool:
    if field_name in CHEAP_FEATURES:
        return True
    if field_name.startswith(CHEAP_PREFIXES):
        low = field_name.lower()
        # a static-geometry distance is cheap; a post-pose/MD one is a label, not a prior.
        return not any(mk in low for mk in _EXPENSIVE_DISTANCE_MARKERS)
    return False


def assert_no_leakage(feature_names: Sequence[str]) -> None:
    """Raise if any feature is an expensive (label-derived) quantity or is not on the
    cheap allowlist.  Call this on the exact columns of any matrix fed to a model."""
    leaked = [f for f in feature_names if f in EXPENSIVE_FIELDS]
    if leaked:
        raise ValueError(f"LEAKAGE: expensive fields used as features: {sorted(leaked)}")
    unknown = [f for f in feature_names if not is_cheap(f)]
    if unknown:
        raise ValueError(
            f"NOT ON CHEAP ALLOWLIST: {sorted(unknown)} — add to CHEAP_FEATURES/"
            f"CHEAP_PREFIXES only if it is computable before any docking/MD."
        )


# --------------------------------------------------------------------------------------
# Labels (expensive functional-state outcomes)
# --------------------------------------------------------------------------------------

def label_reference_like(rec: dict) -> Optional[int]:
    """1 if the anchored design-ligand pose stayed reference_like, else 0. None if unknown."""
    pg = (rec.get("pose_gate") or {}).get("design_ligand") or {}
    status = pg.get("status")
    if status is None:
        return None
    return 1 if status == "reference_like" else 0


def label_nac_nonneg(rec: dict) -> Optional[int]:
    """1 if ΔNAC vs WT is non-negative (reactivity not lost), else 0. None if absent."""
    d = rec.get("nac_delta_vs_wt")
    if not isinstance(d, (int, float)):
        return None
    return 1 if d >= 0 else 0


def label_md_pass(rec: dict) -> Optional[int]:
    if "passed" not in rec or rec["passed"] is None:
        return None
    return 1 if bool(rec["passed"]) else 0


LABELS = {
    "reference_like": label_reference_like,
    "nac_nonneg": label_nac_nonneg,
    "md_pass": label_md_pass,
}


# --------------------------------------------------------------------------------------
# Provenance loading + join
# --------------------------------------------------------------------------------------

def _as_float(v) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None  # None, strings, etc. -> missing


def _load(prov_dir: str, name: str) -> Dict[str, dict]:
    path = os.path.join(prov_dir, name)
    if not os.path.exists(path):
        return {}
    data = json.load(open(path))
    recs = data if isinstance(data, list) else (data.get("candidates") or data.get("records") or [])
    return {r["candidate_id"]: r for r in recs if isinstance(r, dict) and r.get("candidate_id")}


@dataclass
class Row:
    candidate_id: str
    mutation_string: str
    x: Dict[str, float]              # cheap features (NaN allowed for missing)
    labels: Dict[str, Optional[int]] # expensive outcomes
    ml_score: Optional[float]        # baseline-to-beat


def build_table(prov_dir: str) -> List[Row]:
    """Join the provenance JSONs by candidate_id. Cheap features come ONLY from the
    generated/validated records; labels + baseline ml_score come from the heavy records."""
    generated = _load(prov_dir, "generated_candidates.json")
    validated = _load(prov_dir, "validated_candidates.json")
    reranked = _load(prov_dir, "reranked_candidates.json")
    md = _load(prov_dir, "md_candidates.json")

    rows: List[Row] = []
    for cid, g in generated.items():
        v = validated.get(cid, {})
        r = reranked.get(cid, {})
        m = md.get(cid, {})

        x: Dict[str, float] = {}
        for src in (g, v):
            for k, val in src.items():
                if is_cheap(k):
                    f = _as_float(val)
                    x[k] = f if f is not None else np.nan

        # labels: prefer the s10 MD record, fall back to s09 'passed' for md_pass
        labels = {
            "reference_like": label_reference_like(m),
            "nac_nonneg": label_nac_nonneg(m),
            "md_pass": label_md_pass(m) if "passed" in m else label_md_pass(v),
        }
        rows.append(Row(
            candidate_id=cid,
            mutation_string=g.get("mutation_string", ""),
            x=x,
            labels=labels,
            ml_score=_as_float(r.get("ml_score")),
        ))
    return rows


# --------------------------------------------------------------------------------------
# Metrics (pure numpy)
# --------------------------------------------------------------------------------------

def auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """ROC-AUC via Mann-Whitney (ties = 0.5). Returns None if a class is empty."""
    pos = [s for s, y in zip(scores, labels) if y == 1 and s is not None and not _isnan(s)]
    neg = [s for s, y in zip(scores, labels) if y == 0 and s is not None and not _isnan(s)]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def _isnan(x) -> bool:
    return isinstance(x, float) and np.isnan(x)


def _logreg_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                iters: int = 800, lr: float = 0.3) -> Tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30.0, 30.0)))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def loo_logistic_auc(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> Optional[float]:
    """Leave-one-out CV AUC of a standardized logistic over the cheap block. None if there
    are too few samples or the label has no variance."""
    n = X.shape[0]
    if n < 8 or len(set(y.tolist())) < 2:
        return None
    preds = np.zeros(n)
    idx = np.arange(n)
    for i in range(n):
        tr = idx != i
        mu = np.nanmean(X[tr], axis=0)
        sd = np.nanstd(X[tr], axis=0)
        sd[sd == 0] = 1.0
        Xtr = np.nan_to_num((X[tr] - mu) / sd)
        Xte = np.nan_to_num((X[i:i + 1] - mu) / sd)
        w, b = _logreg_fit(Xtr, y[tr], l2=l2)
        preds[i] = 1.0 / (1.0 + np.exp(-(Xte @ w + b)[0]))
    return auc(preds.tolist(), y.tolist())


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

@dataclass
class LabelReport:
    label: str
    n: int
    n_pos: int
    n_neg: int
    baseline_ml_auc: Optional[float]          # current contact-fingerprint model
    loo_cheap_auc: Optional[float]            # cheap-feature logistic, LOO
    univariate: List[Tuple[str, float, int]]  # (feature, |auc-0.5|+0.5, n_usable), sorted
    note: str = ""


def assess(rows: List[Row], min_coverage: float = 0.5) -> Dict[str, LabelReport]:
    # cheap feature columns that are present often enough to be usable
    all_feats = sorted({k for r in rows for k in r.x})
    assert_no_leakage(all_feats)  # mechanical guarantee
    usable_feats = [
        f for f in all_feats
        if sum(1 for r in rows if not _isnan(r.x.get(f, np.nan))) >= min_coverage * len(rows)
    ]

    out: Dict[str, LabelReport] = {}
    for label in LABELS:
        lab_rows = [r for r in rows if r.labels.get(label) is not None]
        y = [r.labels[label] for r in lab_rows]
        n_pos, n_neg = sum(y), len(y) - sum(y)

        # univariate signal per cheap feature
        uni: List[Tuple[str, float, int]] = []
        for f in usable_feats:
            sc = [r.x.get(f, np.nan) for r in lab_rows]
            a = auc(sc, y)
            if a is not None:
                n_use = sum(1 for s in sc if not _isnan(s))
                uni.append((f, round(max(a, 1 - a), 3), n_use))
        uni.sort(key=lambda t: -t[1])

        # baseline: current ml_score vs this label
        base = auc([r.ml_score for r in lab_rows], y)

        # combined cheap-feature LOO
        loo = None
        if usable_feats and len(lab_rows) >= 8 and n_pos and n_neg:
            X = np.array([[r.x.get(f, np.nan) for f in usable_feats] for r in lab_rows])
            loo = loo_logistic_auc(X, np.array(y, dtype=float))

        note = ""
        if not lab_rows:
            note = "no labeled candidates"
        elif n_pos == 0 or n_neg == 0:
            note = "label has NO variance (all same class) -> learnability not assessable"
        elif len(lab_rows) < 8:
            note = f"only {len(lab_rows)} labeled -> directional only, not a verdict"

        out[label] = LabelReport(
            label=label, n=len(lab_rows), n_pos=n_pos, n_neg=n_neg,
            baseline_ml_auc=base, loo_cheap_auc=loo, univariate=uni[:12], note=note,
        )
    return out


def format_report(prov_dir: str, reports: Dict[str, LabelReport]) -> str:
    L = [f"# MD-free functional-state learnability smoke test", f"run: {prov_dir}", ""]
    for label, rep in reports.items():
        L.append(f"## label = {label}   (n={rep.n}, pos={rep.n_pos}, neg={rep.n_neg})")
        if rep.note:
            L.append(f"   NOTE: {rep.note}")
        L.append(f"   baseline current-ml_score AUC : {rep.baseline_ml_auc}")
        L.append(f"   cheap-feature LOO logistic AUC: {rep.loo_cheap_auc}")
        if rep.univariate:
            L.append("   top univariate |AUC| (cheap features):")
            for f, a, n in rep.univariate:
                L.append(f"     {a:>5}  {f}  (n={n})")
        L.append("")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="MD-free functional-state learnability smoke test")
    ap.add_argument("provenance_dir",
                    help="runs/<name>/reports/provenance  (needs generated/validated/"
                         "reranked_candidates.json + md_candidates.json)")
    args = ap.parse_args(argv)
    rows = build_table(args.provenance_dir)
    reports = assess(rows)
    print(format_report(args.provenance_dir, reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
