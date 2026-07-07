"""Calibration metrics for v4 assay results."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, pvariance
from typing import Dict, Iterable, List

from .assay_schema import AssayRecord


def hit_rate_by_lane(records: Iterable[AssayRecord], *, threshold: float = 1.1) -> Dict[str, float]:
    recs = list(records)
    wt = [r.NADP_activity for r in recs if r.lane == "baseline" or r.mutation == "WT"]
    baseline = mean(wt) if wt else 1.0
    lane_hits: Dict[str, List[bool]] = defaultdict(list)
    for r in recs:
        lane_hits[r.lane].append(r.NADP_activity >= baseline * threshold)
    return {lane: sum(vals) / len(vals) for lane, vals in lane_hits.items() if vals}


def low_ml_false_negative_rate(records: Iterable[AssayRecord], *, threshold: float = 1.1) -> float:
    recs = [r for r in records if "low_ml" in r.lane or "uncertainty" in r.lane]
    if not recs:
        return 0.0
    wt = [r.NADP_activity for r in records if r.lane == "baseline" or r.mutation == "WT"]
    baseline = mean(wt) if wt else 1.0
    winners = sum(1 for r in recs if r.NADP_activity >= baseline * threshold)
    return winners / len(recs)


def replicate_variance(records: Iterable[AssayRecord]) -> Dict[str, float]:
    by_mut: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        by_mut[r.mutation].append(r.NADP_activity)
    return {m: pvariance(vals) for m, vals in by_mut.items() if len(vals) > 1}
