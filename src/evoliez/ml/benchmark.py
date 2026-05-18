"""Benchmark suite (user §9).

What matters is recovering KNOWN mutations, not just producing plausible ones:
  - top-k beneficial recovery
  - deleterious / catalytic avoidance
  - Spearman vs measured activity (if provided)
  - AUROC beneficial vs non-beneficial
Pure-python metrics (no scipy/sklearn) so it runs on the laptop.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Sequence

from evoliez.types import Candidate


def load_benchmark(path: Path) -> List[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "mutation": (r.get("mutation") or "").strip(),
                "label": (r.get("label") or "").strip().lower(),
                "activity": float(r["activity"]) if r.get("activity") else None,
            })
    return [r for r in rows if r["mutation"]]


def _spearman(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n))
           * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return round(num / den, 4) if den else 0.0


def _auroc(scores: List[float], pos: List[int]) -> float:
    p = [s for s, y in zip(scores, pos) if y == 1]
    n = [s for s, y in zip(scores, pos) if y == 0]
    if not p or not n:
        return 0.0
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0)
               for a in p for b in n)
    return round(wins / (len(p) * len(n)), 4)


def run_benchmark(
    ranked: Sequence[Candidate], bench: Sequence[dict], *, k: int = 20
) -> Dict[str, object]:
    rank_of = {c.mutation_str: i + 1 for i, c in enumerate(ranked)}
    score_of = {c.mutation_str: c.scores.get("final_score", 0.0)
                for c in ranked}
    n = len(ranked)

    benef = [b for b in bench if b["label"] == "beneficial"]
    delet = [b for b in bench if b["label"] in ("deleterious", "inactive")]

    top_k = set(c.mutation_str for c in ranked[:k])
    recovered = [b for b in benef if b["mutation"] in top_k]
    recall = round(len(recovered) / len(benef), 4) if benef else 0.0

    del_ranks = [rank_of[b["mutation"]] for b in delet
                 if b["mutation"] in rank_of]
    del_avoid = round(
        sum(r > n / 2 for r in del_ranks) / len(del_ranks), 4
    ) if del_ranks else 0.0

    common = [b for b in bench if b["mutation"] in score_of]
    spearman = 0.0
    auroc = 0.0
    if common:
        labelled = [b for b in common if b["activity"] is not None]
        if len(labelled) >= 3:
            spearman = _spearman(
                [score_of[b["mutation"]] for b in labelled],
                [b["activity"] for b in labelled],
            )
        ybin = [1 if b["label"] == "beneficial" else 0 for b in common]
        auroc = _auroc([score_of[b["mutation"]] for b in common], ybin)

    return {
        "n_benchmark": len(bench),
        "topk": k,
        "beneficial_recall_at_k": recall,
        "deleterious_avoidance": del_avoid,
        "spearman_vs_activity": spearman,
        "auroc_beneficial": auroc,
        "recovered": [b["mutation"] for b in recovered],
    }
