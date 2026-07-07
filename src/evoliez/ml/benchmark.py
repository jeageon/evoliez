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


def load_external_benchmark(
    path: Path, fmt: str = "auto",
    beneficial_q: float = 0.75, deleterious_q: float = 0.25,
) -> List[dict]:
    """Load a public benchmark (ProteinGym DMS / FLIP-style) into our rows.

    Maps the mutant column ('mutant'|'mutation') and a continuous fitness
    column ('DMS_score'|'score'|'fitness'|'target'|'log_fitness'). Multi-
    mutants use ':' (ProteinGym) -> normalised to ';'. Continuous scores are
    bucketed into beneficial / neutral / deleterious by quantile so the same
    metrics apply. The user supplies the file (not bundled) - see
    docs/BENCHMARKS.md.
    """
    with open(path, newline="") as fh:
        raw = list(csv.DictReader(fh))
    if not raw:
        return []
    cols = raw[0].keys()
    mut_c = next((c for c in ("mutant", "mutation", "mutations", "variant")
                  if c in cols), None)
    sc_c = next((c for c in ("DMS_score", "score", "fitness", "target",
                             "log_fitness", "activity") if c in cols), None)
    if mut_c is None:
        raise ValueError(f"no mutant column in {list(cols)}")
    vals = []
    for r in raw:
        try:
            vals.append(float(r[sc_c])) if sc_c else None
        except (TypeError, ValueError):
            pass
    vals.sort()
    if vals:
        hi = vals[min(len(vals) - 1, int(beneficial_q * len(vals)))]
        lo = vals[max(0, int(deleterious_q * len(vals)))]
    rows = []
    for r in raw:
        mut = (r.get(mut_c) or "").strip().replace(":", ";")
        if not mut:
            continue
        act = None
        if sc_c:
            try:
                act = float(r[sc_c])
            except (TypeError, ValueError):
                act = None
        if act is None or not vals:
            label = "neutral"
        elif act >= hi:
            label = "beneficial"
        elif act <= lo:
            label = "deleterious"
        else:
            label = "neutral"
        rows.append({"mutation": mut, "label": label, "activity": act})
    return rows


def baseline_rankings(candidates):
    """Alternative orderings from existing per-candidate features (no rerun)
    so the full model can be compared to simple baselines (expert review)."""
    import random as _r

    def feat(c, k, d=0.0):
        return c.details.get("features", {}).get(k, d)

    rng = _r.Random(1234)
    rand = list(candidates)
    rng.shuffle(rand)
    return {
        "full_model": sorted(candidates,
                             key=lambda c: -c.scores.get("final_score", 0.0)),
        "random": rand,
        "conservation_only": sorted(
            candidates, key=lambda c: feat(c, "conservation", 1.0)),
        "msa_only": sorted(
            candidates, key=lambda c: -feat(c, "msa_permissiveness")),
        "interaction_only": sorted(
            candidates, key=lambda c: -feat(c, "interaction_gain")),
    }


def compare_baselines(candidates, bench, *, k: int = 20) -> Dict[str, object]:
    out = {}
    for name, ranked in baseline_rankings(candidates).items():
        r = run_benchmark(ranked, bench, k=k, min_overlap=0)
        out[name] = {
            "beneficial_recall_at_k": r["beneficial_recall_at_k"],
            "auroc_beneficial": r["auroc_beneficial"],
            "spearman_vs_activity": r["spearman_vs_activity"],
        }
    return out


def _spearman(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(v):
        # AVERAGE ranks for ties (matches scipy.stats.spearmanr): tied values
        # each receive the mean of the positions they span, so the metric is
        # numerically correct AND invariant to input row order. Ordinal ranks
        # let input order decide ties, making the reported correlation unstable.
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and v[order[j]] == v[order[i]]:
                j += 1
            avg = (i + j - 1) / 2.0  # mean of 0-based positions i .. j-1
            for p in range(i, j):
                rk[order[p]] = avg
            i = j
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
    beneficial/non-beneficial label (expert review #4).

    NOTE: this is NOT probabilistic calibration. final_score is min-max scaled
    to [0,1] - the worst candidate is pinned to 0 and the best to 1 regardless
    of true confidence - so `ece` measures the reliability of a monotonic
    rescaling of a RANKING score, not a fitted probability. The result is tagged
    `metric: rank_score_reliability` so a benchmark.json reader cannot mistake
    it for genuine probabilistic ECE. A true ECE would need a Platt/isotonic
    calibrator fit on a held-out split."""
    from evoliez.ml.calibration import (
        expected_calibration_error,
        reliability_diagram,
    )

    label = {b["mutation"]: 1 if b["label"] == "beneficial" else 0
             for b in bench}
    pts = [(c.mutation_str, c.scores.get("final_score", 0.0))
           for c in ranked if c.mutation_str in label]
    if len(pts) < 3:
        return {"ece": 0.0, "bins": [], "n": len(pts),
                "metric": "rank_score_reliability"}
    vals = [v for _, v in pts]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    probs = [(v - lo) / span for _, v in pts]
    ys = [label[m] for m, _ in pts]
    return {
        "ece": expected_calibration_error(probs, ys),
        "bins": reliability_diagram(probs, ys),
        "n": len(pts),
        "metric": "rank_score_reliability",
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
