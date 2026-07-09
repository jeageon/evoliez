"""V7 wet-lab calibration loop (ROADMAP_V7 Phase V7-7, Gate 7).

Once the first assay panel comes back, round 2 must be *calibrated*, not rank-based
(ROADMAP_V7 §9). Two questions decide the next library and this module answers both from
the returned hits:

  1. **Which AXES enriched hits?** For every one of the seven evidence axes we measure how
     concentrated the wet-lab hits are among the top scorers on that axis (top-k enrichment
     + a Mann-Whitney AUC). An axis that enriched hits earns weight in round 2; an axis that
     did not is demoted — the ranking learns from the assay instead of trusting the prior.
  2. **What is the false-negative rate BY LANE?** A hit that the panel ranked *below the
     selection cut* is a lane the pipeline under-prioritized (the ``low_signal`` /
     ``uncertainty`` lanes exist precisely to catch these). Counting hits-below-the-cut per
     lane tells round 2 which lanes to widen (mirrors ``experimental.metrics``).

Claim discipline (ROADMAP_V7 §11) is preserved: an assay panel yields at most *screening*
level evidence, never an activity/kcat claim. ``claim_level`` mirrors
``experimental.calibration``: ``L0_uncalibrated`` with no results, ``L1_screened`` once
results exist, and ``L2_calibrated`` only with controls AND >= 2 replicates.

Deterministic and enzyme-generic: nothing here names a residue or an enzyme — hits are
joined to ledgers by ``variant_id`` (or mutation, via the assay-label helper), and every
axis is resolved from the ledger contract. Pure numpy + pydantic; no scipy/rdkit — runs in
the light env. Reuses ``portfolio.stats.enrichment_at_k`` and ``ml.learnability.auc``; the
``portfolio`` argument is duck-typed (``.variants`` with ``.variant_id`` / ``.lane``) so
this module does not hard-depend on the sibling builder.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from evoliez.ml.assay_label import AssayLabel, load_assay_labels
from evoliez.ml.learnability import auc as mannwhitney_auc
from evoliez.portfolio.ledger import (
    AXIS_UNCERTAINTY, CHEAP_AXES, HIGHER_IS_WORSE_AXES, SEVEN_AXES,
    EvidenceLedgerV7, LedgerBundle,
)
from evoliez.portfolio.stats import enrichment_at_k

__all__ = [
    "AxisEnrichment", "LaneFalseNegative", "CalibrationReport",
    "hits_by_variant_from_labels", "hits_from_assay_file",
    "axis_enrichment", "lane_false_negative", "build_calibration_report",
    "round2_from_calibration",
]

# claim levels — mirror experimental.calibration semantics
CLAIM_L0 = "L0_uncalibrated"
CLAIM_L1 = "L1_screened"
CLAIM_L2 = "L2_calibrated"

_DEFAULT_KS: Tuple[int, ...] = (8, 16, 32)
_EXPLORE_FRACTION = 0.25

# results the calibration loop accepts: {variant_id: hit} or [(variant_id, hit), ...]
Results = Union[Dict[str, bool], Sequence[Tuple[str, object]]]


class _Base(BaseModel):
    model_config = {"extra": "forbid"}


class AxisEnrichment(_Base):
    """How concentrated the wet-lab hits are among an axis's top scorers.

    ``enrichment_top_k`` maps a cut ``k`` -> top-k enrichment (hit rate in the top-k over the
    base rate; ``None`` when ``k`` exceeds ``n`` or no hits). ``auc`` is the Mann-Whitney AUC
    of the axis score against the hit label (``> 0.5`` => the axis's good direction predicts
    hits). Higher-is-worse axes (uncertainty) are oriented to their good direction first, so
    an enrichment ``> 1`` always means "this axis, read correctly, enriched hits"."""

    axis: str
    enrichment_top_k: Dict[int, Optional[float]] = Field(default_factory=dict)
    auc: Optional[float] = None
    n: int = 0
    n_hits: int = 0


class LaneFalseNegative(_Base):
    """Per-lane hit accounting relative to a selection cut. ``hits_below_cut`` are hits whose
    panel rank fell below ``cutoff_rank`` — the lane's false negatives, i.e. hits the ranking
    under-prioritized (the round-2 signal for which lanes to widen)."""

    lane: str
    n: int = 0
    n_hits: int = 0
    hits_below_cut: int = 0


class CalibrationReport(_Base):
    """Result of one wet-lab calibration pass. ``claim_level`` never exceeds screening
    pre-controls/replicates; the axis + lane views drive a calibrated round 2."""

    n: int = 0
    n_hits: int = 0
    claim_level: str = CLAIM_L0
    axis_enrichment: List[AxisEnrichment] = Field(default_factory=list)
    lane_false_negative: List[LaneFalseNegative] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# assay-label -> hits join (enzyme-generic; reuses ml.assay_label)
# --------------------------------------------------------------------------------------

def hits_by_variant_from_labels(
    bundle: LedgerBundle,
    labels: Iterable[AssayLabel],
    *,
    threshold: Optional[float] = None,
    activity_only: bool = True,
) -> Dict[str, bool]:
    """Turn assay labels (keyed by mutation) into a ``{variant_id: hit}`` dict against a
    bundle. A label is a hit when its value clears ``threshold`` (default: the median of the
    considered values, a data-driven cut). ``activity_only`` keeps only activity/kinetic
    label types (``AssayLabel.is_activity``) so a stability readout never defines a hit.
    Only mutations that map to a ledger in the bundle are returned."""
    by_mutation: Dict[str, str] = {
        led.mutation: led.variant_id for led in bundle.ledgers if led.mutation
    }
    considered = [
        lab for lab in labels
        if lab.mutation in by_mutation and (lab.is_activity or not activity_only)
    ]
    if not considered:
        return {}
    if threshold is None:
        vals = sorted(lab.value for lab in considered)
        mid = len(vals) // 2
        threshold = vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])
    out: Dict[str, bool] = {}
    for lab in considered:
        vid = by_mutation[lab.mutation]
        # if a variant has several replicate rows, a single clearing row makes it a hit
        out[vid] = out.get(vid, False) or (lab.value >= threshold)
    return out


def hits_from_assay_file(
    path,
    bundle: LedgerBundle,
    *,
    threshold: Optional[float] = None,
    activity_only: bool = True,
) -> Dict[str, bool]:
    """Convenience wrapper: load a long-format assay CSV (``ml.assay_label.load_assay_labels``)
    and join it to the bundle via :func:`hits_by_variant_from_labels`."""
    labels = load_assay_labels(path)
    return hits_by_variant_from_labels(
        bundle, labels, threshold=threshold, activity_only=activity_only)


# --------------------------------------------------------------------------------------
# axis enrichment
# --------------------------------------------------------------------------------------

def _oriented_score(axis: str, score: float) -> float:
    """An axis score oriented so higher == "more likely a hit" (negate higher-is-worse)."""
    return -score if axis in HIGHER_IS_WORSE_AXES else score


def _normalize_results(results: Results) -> Dict[str, bool]:
    if isinstance(results, dict):
        return {str(k): bool(v) for k, v in results.items()}
    out: Dict[str, bool] = {}
    for item in results:
        vid, val = item[0], item[1]
        out[str(vid)] = bool(val)
    return out


def axis_enrichment(
    bundle: LedgerBundle,
    hits_by_variant: Dict[str, bool],
    *,
    ks: Sequence[int] = _DEFAULT_KS,
) -> List[AxisEnrichment]:
    """Per-axis top-k enrichment + AUC of the axis score against the hit labels.

    For each of the seven axes, gather the (oriented) score of every ledger that (a) carries
    that axis (not deferred/missing) and (b) has a known hit label, then compute
    ``enrichment_at_k`` for every ``k`` and a Mann-Whitney AUC. A deferred axis yields an
    honest empty entry (``n == 0``, all ``None``) — never a fabricated zero."""
    hits = _normalize_results(hits_by_variant)
    out: List[AxisEnrichment] = []
    for axis in SEVEN_AXES:
        scores: List[float] = []
        labels: List[int] = []
        for led in bundle.ledgers:
            ev = led.axes[axis]
            if ev.missing:
                continue
            hv = hits.get(led.variant_id)
            if hv is None:
                continue
            scores.append(_oriented_score(axis, ev.score))
            labels.append(1 if hv else 0)
        n = len(scores)
        n_hits = int(sum(labels))
        enr: Dict[int, Optional[float]] = {}
        au: Optional[float] = None
        if n and n_hits:
            for k in ks:
                enr[int(k)] = enrichment_at_k(scores, labels, int(k))
            au = mannwhitney_auc(scores, labels)
        else:
            enr = {int(k): None for k in ks}
        out.append(AxisEnrichment(
            axis=axis, enrichment_top_k=enr, auc=au, n=n, n_hits=n_hits))
    return out


# --------------------------------------------------------------------------------------
# lane false-negative accounting
# --------------------------------------------------------------------------------------

def lane_false_negative(
    portfolio: Any,
    hits_by_variant: Dict[str, bool],
    *,
    cutoff_rank: int,
) -> List[LaneFalseNegative]:
    """Per-lane hit counts, and how many of a lane's hits fell below ``cutoff_rank``.

    ``portfolio`` is duck-typed: any object with ``.variants``, each item exposing
    ``.variant_id`` and ``.lane``. Rank is the 1-based position in ``portfolio.variants``
    (priority order); a hit at rank ``> cutoff_rank`` is a lane false negative. Only variants
    with a known hit label are counted. Lanes are returned sorted for determinism."""
    hits = _normalize_results(hits_by_variant)
    variants = list(getattr(portfolio, "variants", []) or [])
    acc: Dict[str, Dict[str, int]] = {}
    for rank, var in enumerate(variants, start=1):
        vid = getattr(var, "variant_id", None)
        lane = getattr(var, "lane", "") or ""
        if vid is None or vid not in hits:
            continue
        bucket = acc.setdefault(lane, {"n": 0, "n_hits": 0, "hits_below_cut": 0})
        bucket["n"] += 1
        if hits[vid]:
            bucket["n_hits"] += 1
            if rank > cutoff_rank:
                bucket["hits_below_cut"] += 1
    return [
        LaneFalseNegative(lane=lane, n=b["n"], n_hits=b["n_hits"],
                          hits_below_cut=b["hits_below_cut"])
        for lane, b in sorted(acc.items())
    ]


# --------------------------------------------------------------------------------------
# report assembly
# --------------------------------------------------------------------------------------

def _claim_level(n_results: int, *, has_controls: bool, replicates: int) -> str:
    if n_results == 0:
        return CLAIM_L0
    if has_controls and replicates >= 2:
        return CLAIM_L2
    return CLAIM_L1


def build_calibration_report(
    bundle: LedgerBundle,
    portfolio: Any,
    results: Results,
    *,
    has_controls: bool = False,
    replicates: int = 1,
    cutoff_rank: Optional[int] = None,
    ks: Sequence[int] = _DEFAULT_KS,
) -> CalibrationReport:
    """Assemble a calibration report from the first assay panel.

    ``results`` is ``{variant_id: hit}`` or ``[(variant_id, hit), ...]``. ``cutoff_rank``
    defaults to the number of hits (an oracle-sized precision cut: "had we picked the top
    n_hits, which hits would we have missed?"). Claims stay at screening level; nothing here
    asserts activity."""
    hits = _normalize_results(results)
    n = len(hits)
    n_hits = int(sum(1 for v in hits.values() if v))
    claim = _claim_level(n, has_controls=has_controls, replicates=replicates)

    ax = axis_enrichment(bundle, hits, ks=ks) if n else []
    if cutoff_rank is None:
        cutoff_rank = max(1, n_hits)
    lanes = lane_false_negative(portfolio, hits, cutoff_rank=cutoff_rank) if n else []

    notes: List[str] = []
    if n == 0:
        notes.append("no assay results supplied; claims remain L0_uncalibrated "
                     "(hypothesis-grade triage).")
    else:
        enriched = [a.axis for a in ax if _axis_enriched(a)]
        if enriched:
            notes.append("statistically supported evidence band: axes enriching hits — "
                         + ", ".join(enriched) + " (screening-level, not activity-validated).")
        else:
            notes.append("no axis showed hit enrichment above the null; round 2 stays "
                         "broad / exploratory.")
        fn_lanes = [l.lane for l in lanes if l.hits_below_cut > 0]
        if fn_lanes:
            notes.append("hits fell below the selection cut in lane(s): "
                         + ", ".join(fn_lanes) + " — widen these in round 2.")
        if claim == CLAIM_L1:
            notes.append("screening-level evidence only (L1): controls and/or >= 2 "
                         "replicates required before L2_calibrated.")
    return CalibrationReport(
        n=n, n_hits=n_hits, claim_level=claim,
        axis_enrichment=ax, lane_false_negative=lanes, notes=notes,
    )


# --------------------------------------------------------------------------------------
# round-2 acquisition (calibrated, not rank-based)
# --------------------------------------------------------------------------------------

def _axis_enriched(a: AxisEnrichment) -> bool:
    if any(e is not None and e > 1.0 for e in a.enrichment_top_k.values()):
        return True
    return a.auc is not None and a.auc > 0.5


def _axis_weight(a: AxisEnrichment) -> float:
    """Round-2 weight for an axis: how much better than the null it enriched hits."""
    w = 0.0
    for e in a.enrichment_top_k.values():
        if e is not None and e > 1.0:
            w = max(w, e - 1.0)
    if a.auc is not None and a.auc > 0.5:
        w = max(w, 2.0 * (a.auc - 0.5))
    return w


def round2_from_calibration(
    report: CalibrationReport,
    bundle: LedgerBundle,
    *,
    size: int = 16,
    tested: Optional[Iterable[str]] = None,
    explore_fraction: float = _EXPLORE_FRACTION,
) -> List[dict]:
    """Propose a calibrated round-2 library from the report.

    Not-yet-tested candidates (``bundle`` minus ``tested``) are ranked by the axes the report
    says enriched hits (weighted by their enrichment margin); ``explore_fraction`` of the
    library is reserved for the highest-uncertainty candidates (learning value). Returns
    ``[{variant_id, mutation, reason}, ...]``, deterministic under a fixed input. Reasons stay
    claim-safe ("prioritized for experimental testing"), never an activity assertion. When no
    axis enriched hits, falls back to an equal-weight cheap-axis prior so round 2 is still
    populated."""
    if size <= 0:
        return []
    tested_set = {str(t) for t in (tested or [])}
    pool: List[EvidenceLedgerV7] = [
        led for led in bundle.ledgers if led.variant_id not in tested_set]
    if not pool:
        return []

    weights = {a.axis: _axis_weight(a) for a in report.axis_enrichment}
    enriched = [ax for ax, w in weights.items() if w > 0.0]
    if not enriched:
        # no calibrated signal -> equal-weight cheap-axis prior (still claim-safe)
        enriched = list(CHEAP_AXES)
        weights = {ax: 1.0 for ax in enriched}
        reason_axes = "cheap-axis prior (no axis enriched hits)"
    else:
        reason_axes = "enriched axes: " + ", ".join(sorted(enriched))

    def exploit_score(led: EvidenceLedgerV7) -> float:
        s = 0.0
        for ax in enriched:
            ev = led.axes[ax]
            if ev.missing:
                continue
            s += weights[ax] * _oriented_score(ax, ev.score)
        return s

    def uncertainty_score(led: EvidenceLedgerV7) -> float:
        ev = led.axes[AXIS_UNCERTAINTY]
        return ev.score if not ev.missing else float("-inf")

    # reserve an explore slice for the most-uncertain, then fill by exploit score
    n_explore = min(len(pool), max(0, round(size * explore_fraction))) if size > 1 else 0
    explore_ranked = sorted(pool, key=lambda l: (-uncertainty_score(l), l.variant_id))
    explore_pick = [l for l in explore_ranked
                    if uncertainty_score(l) != float("-inf")][:n_explore]
    explore_ids = {l.variant_id for l in explore_pick}

    exploit_ranked = sorted(pool, key=lambda l: (-exploit_score(l), l.variant_id))

    chosen: List[Tuple[EvidenceLedgerV7, str]] = []
    seen: set = set()
    for led in explore_pick:
        chosen.append((led, "high-uncertainty explore probe (uncertainty_learning_value)"))
        seen.add(led.variant_id)
    for led in exploit_ranked:
        if len(chosen) >= size:
            break
        if led.variant_id in seen or led.variant_id in explore_ids:
            continue
        chosen.append((led, "prioritized for experimental testing by " + reason_axes))
        seen.add(led.variant_id)

    return [
        {"variant_id": led.variant_id, "mutation": led.mutation, "reason": reason}
        for led, reason in chosen[:size]
    ]
