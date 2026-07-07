"""Model calibration + recommendation class (user §10).

A high ML score is not a high hit probability. We attach an uncertainty
estimate and a recommendation class so experimenters avoid false confidence,
and provide ECE for the benchmark suite.
"""

from __future__ import annotations

from typing import List, Sequence

from evoliez.types import Candidate


def candidate_uncertainty(cand: Candidate) -> float:
    """0 (confident) .. 1 (very uncertain) from independent risk signals."""
    s = cand.scores
    parts = [
        1.0 - s.get("redocking_consistency", 0.5),
        s.get("md_instability", 0.0),
        min(1.0, abs(s.get("ensemble_disagreement", 0.0)) / 2.0),
        1.0 - s.get("family_interaction_score", 0.5),
        max(0.0, -s.get("d_pocket_plddt", 0.0)) * 2.0,  # confidence dropped
    ]
    u = sum(parts) / len(parts)
    return round(max(0.0, min(1.0, u)), 4)


def recommendation(
    score: float, uncertainty: float, rank: int, n: int,
    geometry: float | None = None,
) -> str:
    """Recommendation class — all labels are claim-safe (no bare "strong candidate", which
    ClaimGuard forbids as catalysis-implying). A top-rank, confident candidate is a "priority
    screening candidate" only if its reaction geometry is actually positive. ``geometry`` is the
    reaction-geometry signal (e.g. ΔNAC-vs-WT, >0 = more productive than WT). geometry 0 (computed,
    dead) or None (not computed) caps the label at a structural/binding recommendation: the pipeline
    proves structure + binding, not catalysis, so a catalytically-dead or unmeasured candidate is
    never a priority. This kills the CAR-class overclaim (a geometry-0 mutant stamped "strong
    candidate") at the source, not just at report-lint time."""
    top = rank <= max(1, int(0.15 * n))
    if top and uncertainty < 0.4:
        if geometry is not None and geometry > 0:
            return "priority screening candidate"
        return "structural/binding candidate"
    if uncertainty >= 0.65:
        return "reject"
    return "uncertain candidate"


def expected_calibration_error(
    probs: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    if not probs:
        return 0.0
    n = len(probs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo < p <= hi or (b == 0 and p == 0)]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return round(ece, 4)


def reliability_diagram(
    probs: Sequence[float], labels: Sequence[int], bins: int = 10
) -> List[dict]:
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        # mirror expected_calibration_error's binning: bin 0 includes prob==0,
        # so the diagram covers the SAME population as the reported ECE and the
        # per-bin counts sum to n.
        idx = [i for i, p in enumerate(probs)
               if lo < p <= hi or (b == 0 and p == 0)]
        if idx:
            out.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": len(idx),
                "mean_conf": round(sum(probs[i] for i in idx) / len(idx), 4),
                "accuracy": round(sum(labels[i] for i in idx) / len(idx), 4),
            })
    return out
