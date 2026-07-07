"""Active-learning acquisition + diverse focused library (user §11, §15).

The focused library should not be the raw top-N (mutations cluster together);
it must be diverse so each DBTL round informs the model. Acquisition balances
predicted improvement, uncertainty (explore) and library cost; diversity is
enforced structurally by the cluster round-robin, not by a score term.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from evoliez.ml.calibration import candidate_uncertainty
from evoliez.types import Candidate


def acquisition_score(
    cand: Candidate,
    *,
    beta: float = 0.3,
    delta: float = 0.05,
) -> float:
    imp = cand.scores.get("final_score", 0.0)
    unc = candidate_uncertainty(cand)
    cost = 0.1 * (len(cand.mutations) - 1)  # multi-mutants cost more
    return round(imp + beta * unc - delta * cost, 4)


def _cluster_key(c: Candidate) -> tuple:
    """Diversity axes: position, mutant-aa chemistry, targeted ligand atom."""
    m = c.mutations[0] if c.mutations else None
    chem = "X"
    if m:
        if m.mut in "DE":
            chem = "neg"
        elif m.mut in "KRH":
            chem = "pos"
        elif m.mut in "STNQYC":
            chem = "polar"
        elif m.mut in "AVLIMFW":
            chem = "hydrophobic"
    return (
        m.position // 10 if m else -1,
        chem,
        c.details.get("near_ligand_atom", "?"),
        int(c.details.get("features", {}).get("conservation", 0) * 4),
    )


def select_focused_library(
    ranked: Sequence[Candidate], size: int, *, beta: float = 0.3,
) -> List[Candidate]:
    """Round-robin over diversity clusters so the library spans positions /
    chemistries / ligand-atom targets instead of one hot region. Diversity is
    enforced by this round-robin, not by an acquisition-score term."""
    buckets: Dict[tuple, List[Candidate]] = {}
    for c in ranked:
        c.scores["acquisition_score"] = acquisition_score(c, beta=beta)
        buckets.setdefault(_cluster_key(c), []).append(c)
    for b in buckets.values():
        b.sort(key=lambda c: -c.scores["acquisition_score"])

    order = sorted(
        buckets.keys(),
        key=lambda k: -buckets[k][0].scores["acquisition_score"],
    )
    # Round-robin over a per-round snapshot of the non-empty buckets, dropping
    # emptied buckets BETWEEN rounds. Mutating the order list mid-index (the old
    # approach) shifted `i % len(order)` onto a different bucket when one
    # emptied, scrambling the score-ordered diversity sequence.
    out: List[Candidate] = []
    active = list(order)
    while len(out) < size and active:
        for k in active:
            if len(out) >= size:
                break
            if buckets[k]:
                out.append(buckets[k].pop(0))
        active = [k for k in active if buckets[k]]
    return out
