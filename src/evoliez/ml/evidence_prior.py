"""ML Evidence Prior — multi-head, label-provenance'd, split-validated (ROADMAP_V3 D5/V3-7).

The v3 ML is a PRIOR, never an activity predictor. It answers only "can cheap, pre-
validation features predict the expensive functional-state label the heavy stages
compute?" — across SEVERAL heads, each carrying its label provenance, and validated with
group splits (target / family / enzyme-class) so near-identical variants never leak
across train/test and inflate generalization.

This module is the FRAMEWORK (head registry + provenance + split policy + claim policy);
the actual retrain stays run/data-gated (a real anchored run's features), exactly like
v2 Phase F. It reuses ``learnability``'s cheap/expensive contract + label functions, so
the leakage guard still holds: no MD/pose/Boltz quantity can become a feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .learnability import LABELS, assert_no_leakage

# --- label provenance ----------------------------------------------------------------
COMPUTATIONAL_SURROGATE = "computational_surrogate"
EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class HeadSpec:
    """One ML head. ``label_key`` indexes ``learnability.LABELS``. A computational-
    surrogate head learns a deterministic validator — useful, but it may only claim to
    be a surrogate prior, NEVER an experimental activity predictor."""
    name: str
    label_key: str
    label_source: str = COMPUTATIONAL_SURROGATE
    md_used_as_label: bool = True
    allowed_claim: str = "surrogate evidence prior"
    prohibited_claim: str = "experimental activity predictor"


# the v3 heads. Note what is ABSENT: P(kcat), P(kcat/KM), TS barrier, wet-lab activity.
HEAD_SPECS: List[HeadSpec] = [
    HeadSpec("P(structural_viable)", "md_pass"),
    HeadSpec("P(reference_pose_accommodation)", "reference_like"),
    HeadSpec("P(reaction_geometry_nonneg)", "nac_nonneg"),
]

# head names that would be over-claims if they ever appeared (a mechanical guard). Includes the
# common catalysis/kinetics SYNONYMS a head could hide behind (Fable review — the denylist
# previously missed turnover/vmax/productivity/... so an activity-implying head name slipped past).
FORBIDDEN_HEAD_TOKENS = (
    "kcat", "k_cat", "km", "k_m", "activity", "barrier", "transition_state", "efficiency",
    "turnover", "vmax", "v_max", "productivity", "conversion", "reaction_rate", "rate_constant",
    "cataly",  # covers catalytic / catalysis / catalyst / catalyze (Fable verification)
    "selectivity", "specific_activity", "titer", "koff", "kon", "ic50", "flux",
    "fitness", "yield", "growth", "activation", "potency", "efficacy", "kobs",
    "product_formation", "gain_of_function",
)


def assert_heads_are_priors(heads: Sequence[HeadSpec] = tuple(HEAD_SPECS)) -> None:
    """No head may predict activity/kinetics; every head's label must be a real label
    function. Raises if a head violates the prior-only contract."""
    for h in heads:
        low = h.name.lower()
        bad = [t for t in FORBIDDEN_HEAD_TOKENS if t in low]
        if bad:
            raise ValueError(f"head {h.name!r} names a forbidden target {bad} — the ML is "
                             f"a prior, never an activity/kinetics predictor")
        if h.label_key not in LABELS:
            raise ValueError(f"head {h.name!r} label_key {h.label_key!r} not in learnability.LABELS")
        if h.label_source == EXPERIMENTAL and h.prohibited_claim == "experimental activity predictor":
            # an experimental-label head MAY claim activity — only computational ones may not
            pass


# --- generalization / split policy ---------------------------------------------------
RANDOM = "random"
TARGET = "target"
FAMILY = "family"
ENZYME_CLASS = "enzyme_class"

_SPLIT_CLAIM = {
    RANDOM: "internal interpolation only",
    TARGET: "target-level transfer estimate",
    FAMILY: "family-level generalization estimate",
    ENZYME_CLASS: "mechanism-template transfer estimate",
}


@dataclass
class GeneralizationClaim:
    split: str
    allowed_claim: str
    broad_generalization_allowed: bool
    note: str = ""


def generalization_claim(split: str, n_enzyme_classes: int) -> GeneralizationClaim:
    """The claim a given validation split supports. A broad enzyme-class generalization
    claim is PROHIBITED with fewer than 3 classes (Stage-1 = FDH + TEM-1 + glycosidase
    has exactly 3 -> only an 'initial cross-mechanism stress test')."""
    allowed = _SPLIT_CLAIM.get(split, "uncharacterized split")
    broad_ok = split == ENZYME_CLASS and n_enzyme_classes >= 4
    note = ""
    if split == ENZYME_CLASS and n_enzyme_classes < 4:
        note = ("only an initial cross-mechanism stress test; broad enzyme-class "
                "generalization is not claimable with <4 classes")
    return GeneralizationClaim(split=split, allowed_claim=allowed,
                               broad_generalization_allowed=broad_ok, note=note)


def group_split(group_by_id: Dict[str, str], holdout_groups: Sequence[str]):
    """Partition row ids into (train, test) so NO group is split across the two — the
    test holds entire groups (a target / family / class), never individual variants of a
    group seen in train. Returns (train_ids, test_ids)."""
    hold = set(holdout_groups)
    train, test = [], []
    for rid, grp in group_by_id.items():
        (test if grp in hold else train).append(rid)
    return train, test


def assert_no_group_leak(train_ids: Sequence[str], test_ids: Sequence[str],
                         group_by_id: Dict[str, str]) -> None:
    """Guard: a group present in test must not also appear in train (the leak the split
    exists to prevent)."""
    test_groups = {group_by_id[i] for i in test_ids if i in group_by_id}
    train_groups = {group_by_id[i] for i in train_ids if i in group_by_id}
    leaked = test_groups & train_groups
    if leaked:
        raise ValueError(f"group leak across split: {sorted(leaked)} in both train and test")


def feature_contract_ok(feature_names: Sequence[str]) -> bool:
    """The leakage guard still holds for the prior's features (no MD/pose/Boltz label
    quantity may be a feature). Raises via assert_no_leakage on violation."""
    assert_no_leakage(list(feature_names))
    return True


# =====================================================================================
# Trainable multi-head prior (ML strategy §3.1). The framework above stays as-is; this
# turns it into an actual, mechanism-agnostic model: pure-numpy logistic per head over the
# CHEAP feature block, group-split validated, calibrated. No activity/kinetics head can be
# fit (assert_heads_are_priors is called on fit). Reuses learnability's Row/_logreg_fit so
# the leakage guard and cheap/expensive contract are the same objects.
# =====================================================================================
import numpy as np  # noqa: E402

from .learnability import Row, _isnan, _logreg_fit, auc  # noqa: E402


def assert_single_label_source(heads: Sequence[HeadSpec]) -> None:
    """One fit call trains ONE label provenance. Mixing computed-surrogate and
    experimental labels into a single training batch is refused (their semantics and claim
    ceilings differ)."""
    sources = {h.label_source for h in heads}
    if len(sources) > 1:
        raise ValueError(
            f"a single fit trains one label source; got {sorted(sources)}. Train "
            f"computational-surrogate and experimental heads separately.")


@dataclass
class TrainedHead:
    name: str
    label_key: str
    label_source: str
    feature_names: List[str]
    mu: "np.ndarray"
    sd: "np.ndarray"
    w: "np.ndarray"
    b: float
    n_train: int
    n_pos: int
    n_neg: int
    train_auc: Optional[float] = None
    holdout_auc: Optional[float] = None

    def _prob(self, x: Dict[str, float]) -> float:
        vec = np.array([x.get(f, np.nan) for f in self.feature_names], dtype=float)
        z = np.nan_to_num((vec - self.mu) / self.sd)
        return float(1.0 / (1.0 + np.exp(-np.clip(z @ self.w + self.b, -30, 30))))

    def confidence(self, x: Dict[str, float]) -> str:
        """Prediction confidence from feature coverage + evidence strength. High score with
        low confidence is representable (EvidenceCard v4), never hidden."""
        cov = sum(1 for f in self.feature_names if not _isnan(x.get(f, np.nan))) / max(
            1, len(self.feature_names))
        strong = (self.holdout_auc or 0.0) >= 0.7 and self.n_train >= 20
        if cov >= 0.8 and strong:
            return "high"
        if cov >= 0.5 and (self.holdout_auc or self.train_auc or 0.0) >= 0.6:
            return "medium"
        return "low"


@dataclass
class MultiHeadEvidencePrior:
    heads: Dict[str, TrainedHead] = field(default_factory=dict)

    def predict_row(self, x: Dict[str, float]) -> Dict[str, dict]:
        return {
            name: {"score": round(h._prob(x), 4), "confidence": h.confidence(x)}
            for name, h in self.heads.items()
        }

    def summary(self) -> Dict[str, dict]:
        return {
            name: {
                "label_key": h.label_key, "label_source": h.label_source,
                "n_train": h.n_train, "n_pos": h.n_pos, "n_neg": h.n_neg,
                "train_auc": h.train_auc, "holdout_auc": h.holdout_auc,
            }
            for name, h in self.heads.items()
        }


def _usable_features(rows: Sequence[Row], min_coverage: float) -> List[str]:
    all_feats = sorted({k for r in rows for k in r.x})
    assert_no_leakage(all_feats)  # cheap/expensive contract, mechanically
    return [
        f for f in all_feats
        if sum(1 for r in rows if not _isnan(r.x.get(f, np.nan))) >= min_coverage * len(rows)
    ]


def fit_evidence_prior(
    rows: Sequence[Row],
    heads: Sequence[HeadSpec] = tuple(HEAD_SPECS),
    *,
    group_by_id: Optional[Dict[str, str]] = None,
    holdout_groups: Sequence[str] = (),
    min_coverage: float = 0.5,
    min_train: int = 8,
) -> MultiHeadEvidencePrior:
    """Fit one logistic head per HeadSpec over the cheap feature block.

    - ``assert_heads_are_priors``: no activity/kinetics head can be fit.
    - group split (target/family/enzyme_class) via ``holdout_groups`` — never a random
      split; ``assert_no_group_leak`` guarantees no group spans train and test.
    - a head with no label variance (all-positive: the survivorship failure mode) is
      skipped with a recorded reason rather than trained on a degenerate label.
    """
    assert_heads_are_priors(heads)
    assert_single_label_source(heads)
    feats = _usable_features(rows, min_coverage)

    group_by_id = group_by_id or {}
    holdout = set(holdout_groups)
    if holdout:
        train_ids = [r.candidate_id for r in rows if group_by_id.get(r.candidate_id) not in holdout]
        test_ids = [r.candidate_id for r in rows if group_by_id.get(r.candidate_id) in holdout]
        assert_no_group_leak(train_ids, test_ids, group_by_id)
        train_set, test_set = set(train_ids), set(test_ids)
    else:
        train_set, test_set = {r.candidate_id for r in rows}, set()

    trained: Dict[str, TrainedHead] = {}
    for h in heads:
        lab_rows = [r for r in rows if r.labels.get(h.label_key) is not None]
        tr = [r for r in lab_rows if r.candidate_id in train_set]
        y_tr = [int(r.labels[h.label_key]) for r in tr]
        n_pos, n_neg = sum(y_tr), len(y_tr) - sum(y_tr)
        if len(tr) < min_train or n_pos == 0 or n_neg == 0:
            # degenerate (too few or single-class -> the survivorship all-positive case):
            # do not fit a head on a label with no negatives.
            continue
        X = np.array([[r.x.get(f, np.nan) for f in feats] for r in tr], dtype=float)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[sd == 0] = 1.0
        Xn = np.nan_to_num((X - mu) / sd)
        w, b = _logreg_fit(Xn, np.array(y_tr, dtype=float))

        head = TrainedHead(
            name=h.name, label_key=h.label_key, label_source=h.label_source,
            feature_names=feats, mu=mu, sd=sd, w=w, b=b,
            n_train=len(tr), n_pos=n_pos, n_neg=n_neg,
        )
        head.train_auc = auc([head._prob(r.x) for r in tr], y_tr)
        te = [r for r in lab_rows if r.candidate_id in test_set]
        if te and len({int(r.labels[h.label_key]) for r in te}) == 2:
            head.holdout_auc = auc([head._prob(r.x) for r in te],
                                   [int(r.labels[h.label_key]) for r in te])
        trained[h.name] = head
    return MultiHeadEvidencePrior(heads=trained)
