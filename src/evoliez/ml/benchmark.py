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


def calibration_curve(
    ranked: Sequence[Candidate], bench: Sequence[dict]
) -> Dict[str, object]:
    """Reliability of the (min-max normalised) final score vs the
    beneficial/non-beneficial label (expert review #4)."""
    from evoliez.ml.calibration import (
        expected_calibration_error,
        reliability_diagram,
    )

    label = {b["mutation"]: 1 if b["label"] == "beneficial" else 0
             for b in bench}
    pts = [(c.mutation_str, c.scores.get("final_score", 0.0))
           for c in ranked if c.mutation_str in label]
    if len(pts) < 3:
        return {"ece": 0.0, "bins": [], "n": len(pts)}
    vals = [v for _, v in pts]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    probs = [(v - lo) / span for _, v in pts]
    ys = [label[m] for m, _ in pts]
    return {
        "ece": expected_calibration_error(probs, ys),
        "bins": reliability_diagram(probs, ys),
        "n": len(pts),
    }


def validate_benchmark(
    bench: Sequence[dict], target_sequence: str = ""
) -> List[str]:
    """Warn when a benchmark mutation's wild-type letter does not match the
    target sequence (the old example had A85K while residue 85 is E)."""
    warnings: List[str] = []
    if not target_sequence:
        return warnings
    for b in bench:
        m = b["mutation"]
        try:
            wt, pos = m[0], int(m[1:-1])
        except (ValueError, IndexError):
            warnings.append(f"unparseable mutation '{m}'")
            continue
        if pos < 1 or pos > len(target_sequence):
            warnings.append(f"{m}: position out of range")
        elif target_sequence[pos - 1] != wt:
            warnings.append(
                f"{m}: WT '{wt}' != sequence '{target_sequence[pos - 1]}'"
            )
    return warnings


def run_benchmark(
    ranked: Sequence[Candidate], bench: Sequence[dict], *, k: int = 20,
    catalytic_positions: Sequence[int] = (),
    known_site: Sequence[int] = (),
    target_sequence: str = "",
    min_overlap: int = 1,
) -> Dict[str, object]:
    rank_of = {c.mutation_str: i + 1 for i, c in enumerate(ranked)}
    score_of = {c.mutation_str: c.scores.get("final_score", 0.0)
                for c in ranked}
    n = len(ranked)

    # validity guard: distinguish "model is bad" from "benchmark mismatch"
    overlap = sum(1 for b in bench if b["mutation"] in rank_of)
    warnings = validate_benchmark(bench, target_sequence)
    if overlap < min_overlap:
        warnings.append(
            f"benchmark/candidate overlap = {overlap} (< {min_overlap}); "
            f"recall/AUROC/calibration are NOT meaningful - the benchmark "
            f"mutations do not match the generated candidates"
        )
    valid = overlap >= min_overlap and not any(
        "WT '" in w for w in warnings
    )

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

    # catalytic protection: single-mutants at a catalytic position should
    # rank in the bottom half (we should NOT recommend breaking catalysis)
    cat = set(catalytic_positions)
    cat_ranked = [
        rank_of[c.mutation_str] for c in ranked
        if len(c.mutations) == 1 and c.mutations[0].position in cat
    ]
    catalytic_protection = round(
        sum(r > n / 2 for r in cat_ranked) / len(cat_ranked), 4
    ) if cat_ranked else 1.0

    # binding-site enrichment: fraction of top-k that targets a known
    # binding-site residue
    site = set(known_site)
    topk_cands = ranked[:k]
    site_hits = sum(
        1 for c in topk_cands
        if any(m.position in site for m in c.mutations)
    )
    site_enrichment = round(
        site_hits / len(topk_cands), 4
    ) if (site and topk_cands) else 0.0

    return {
        "n_benchmark": len(bench),
        "topk": k,
        "beneficial_recall_at_k": recall,
        "deleterious_avoidance": del_avoid,
        "spearman_vs_activity": spearman,
        "auroc_beneficial": auroc,
        "catalytic_protection_rate": catalytic_protection,
        "binding_site_enrichment": site_enrichment,
        "recovered": [b["mutation"] for b in recovered],
        "calibration": calibration_curve(ranked, bench),
        "overlap": overlap,
        "valid": valid,
        "warnings": warnings,
    }


def run_ablation(
    config_path: str,
    bench: Sequence[dict],
    *,
    k: int = 20,
    toggles: Sequence[str] = ("mechanism", "negative_design",
                              "subfamily_msa", "interaction_model", "md"),
    base_output: str | None = None,
) -> Dict[str, object]:
    """Run the (mock) pipeline once as baseline, then once per ablation with
    that layer disabled, and report the metric delta - the evidence that each
    layer contributes (expert review #4)."""
    import tempfile
    from pathlib import Path

    from evoliez.config import load_config
    from evoliez.context import RunContext
    from evoliez.pipeline import Pipeline
    from evoliez.utils.seeds import seed_everything

    base_output = base_output or tempfile.mkdtemp(prefix="evoliez_abl_")

    def _run(tag: str, ov: dict) -> Dict[str, object]:
        cfg = load_config(config_path, {
            "project.output_dir": str(Path(base_output) / tag),
            **ov,
        })
        seed_everything(cfg.seed)
        ctx = RunContext(cfg, allow_small_disk=True).setup()
        Pipeline().run(ctx)
        r = run_benchmark(
            ctx.get("ranked_candidates", []), bench, k=k,
            catalytic_positions=ctx.get("catalytic_positions", []),
            known_site=ctx.get("known_binding_site", []),
        )
        return r

    _OV = {
        "mechanism": {"advanced": {"mechanism": False}},
        "negative_design": {"advanced": {"negative_design": False}},
        "subfamily_msa": {"advanced": {"subfamily_msa": False}},
        "interaction_model": {"interaction_model": {"enabled": False}},
        "md": {"validation": {"md": {"enabled": False}}},
    }
    base = _run("baseline", {})
    metric = "auroc_beneficial"
    rows = [{"ablation": "full_model", metric: base.get(metric, 0.0)}]
    for t in toggles:
        if t not in _OV:
            continue
        r = _run(f"no_{t}", _OV[t])
        rows.append({
            "ablation": f"no_{t}",
            metric: r.get(metric, 0.0),
            f"delta_{metric}": round(
                r.get(metric, 0.0) - base.get(metric, 0.0), 4
            ),
        })
    return {"metric": metric, "baseline": base, "ablation": rows}
