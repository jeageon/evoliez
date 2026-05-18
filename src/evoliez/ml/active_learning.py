"""Active-learning acquisition + diverse focused library (user §11, §15).

The focused library should not be the raw top-N (mutations cluster together);
it must be diverse so each DBTL round informs the model. Acquisition balances
predicted improvement, uncertainty (explore), diversity and library cost.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from evoliez.ml.calibration import candidate_uncertainty
from evoliez.types import Candidate


def acquisition_score(
    cand: Candidate,
    *,
    beta: float = 0.3,
    gamma: float = 0.2,
    delta: float = 0.05,
    diversity: float = 0.0,
) -> float:
    imp = cand.scores.get("final_score", 0.0)
    unc = candidate_uncertainty(cand)
    cost = 0.1 * (len(cand.mutations) - 1)  # multi-mutants cost more
    return round(imp + beta * unc + gamma * diversity - delta * cost, 4)


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
    gamma: float = 0.2,
) -> List[Candidate]:
    """Round-robin over diversity clusters so the library spans positions /
    chemistries / ligand-atom targets instead of one hot region."""
    buckets: Dict[tuple, List[Candidate]] = {}
    for c in ranked:
        c.scores["acquisition_score"] = acquisition_score(
            c, beta=beta, gamma=gamma
        )
        buckets.setdefault(_cluster_key(c), []).append(c)
    for b in buckets.values():
        b.sort(key=lambda c: -c.scores["acquisition_score"])

    order = sorted(
        buckets.keys(),
        key=lambda k: -buckets[k][0].scores["acquisition_score"],
    )
    out: List[Candidate] = []
    i = 0
    while len(out) < size and order:
        k = order[i % len(order)]
        if buckets[k]:
            out.append(buckets[k].pop(0))
        else:
            order.remove(k)
            continue
        i += 1
    return out
