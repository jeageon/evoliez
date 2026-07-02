"""Prior/selection evaluation — NOT a single AUC (ML strategy §8).

The right questions for an experiment-triage prior are: does the top-k enrich for hits,
how many true hits would a hard ML cut have thrown away (false-negative rate), and does a
low-ML control lane actually rescue hits? This module computes those, plus a calibration
summary (reusing ``ml/calibration``). All pure-python, mechanism-agnostic — it operates on
scores + binary hits + optional lane tags, never on a specific enzyme's readout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from evoliez.ml.calibration import expected_calibration_error


def _rank_desc(scores: Sequence[float]) -> List[int]:
    """Indices sorted by descending score (stable)."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def topk_enrichment(
    scores: Sequence[float], hits: Sequence[int], ks: Sequence[int] = (8, 16, 32)
) -> Dict[int, Optional[float]]:
    """Enrichment = (hit rate in top-k) / (baseline hit rate). >1 means the score enriches
    for hits; None when k exceeds n or the baseline rate is 0."""
    n = len(scores)
    base = sum(hits) / n if n else 0.0
    order = _rank_desc(scores)
    out: Dict[int, Optional[float]] = {}
    for k in ks:
        if k > n or base == 0:
            out[k] = None
            continue
        topk = order[:k]
        rate = sum(hits[i] for i in topk) / k
        out[k] = round(rate / base, 3)
    return out


def false_negative_rate(
    scores: Sequence[float], hits: Sequence[int], *, cutoff_rank: int
) -> Optional[float]:
    """Fraction of TRUE hits that fall BELOW a top-``cutoff_rank`` ML cut — i.e. would be
    discarded by a hard ml_score top-N filter. High FN rate = the cut is unsafe."""
    total_hits = sum(hits)
    if total_hits == 0:
        return None
    order = _rank_desc(scores)
    kept = set(order[:cutoff_rank])
    discarded_hits = sum(hits[i] for i in range(len(hits)) if i not in kept)
    return round(discarded_hits / total_hits, 3)


@dataclass
class LaneFNReport:
    lane: str
    n: int
    hits: int
    hits_below_ml_cut: int  # hits this lane surfaced that the ml cut would have dropped


def false_negative_rescue_by_lane(
    ml_scores: Sequence[float],
    hits: Sequence[int],
    lanes: Sequence[str],
    *,
    cutoff_rank: int,
) -> List[LaneFNReport]:
    """Per lane: how many hits it surfaced that a top-``cutoff_rank`` ml cut would have
    discarded. A non-zero count in a low-ML/uncertainty lane is the concrete proof that a
    hard ML cut is unsafe (the v2 low-ML-winner observation)."""
    order = _rank_desc(ml_scores)
    kept = set(order[:cutoff_rank])
    per: Dict[str, LaneFNReport] = {}
    for i, lane in enumerate(lanes):
        rep = per.setdefault(lane, LaneFNReport(lane=lane, n=0, hits=0, hits_below_ml_cut=0))
        rep.n += 1
        if hits[i]:
            rep.hits += 1
            if i not in kept:
                rep.hits_below_ml_cut += 1
    return sorted(per.values(), key=lambda r: -r.hits_below_ml_cut)


@dataclass
class PriorEvalReport:
    n: int
    n_hits: int
    enrichment: Dict[int, Optional[float]]
    fn_rate_at_cut: Optional[float]
    cutoff_rank: int
    ece: float
    lane_rescue: List[LaneFNReport] = field(default_factory=list)


def evaluate_prior(
    scores: Sequence[float],
    hits: Sequence[int],
    *,
    cutoff_rank: int,
    lanes: Optional[Sequence[str]] = None,
    ks: Sequence[int] = (8, 16, 32),
) -> PriorEvalReport:
    return PriorEvalReport(
        n=len(scores),
        n_hits=sum(hits),
        enrichment=topk_enrichment(scores, hits, ks),
        fn_rate_at_cut=false_negative_rate(scores, hits, cutoff_rank=cutoff_rank),
        cutoff_rank=cutoff_rank,
        ece=expected_calibration_error(list(scores), list(hits)),
        lane_rescue=(
            false_negative_rescue_by_lane(scores, hits, lanes, cutoff_rank=cutoff_rank)
            if lanes is not None else []
        ),
    )
