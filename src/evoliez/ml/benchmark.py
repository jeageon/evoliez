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
from typing import Dict, List, Optional, Sequence

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


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    """Spearman rho with average-rank tie correction.

    Returns ``None`` (not 0.0) when the correlation is *undefined* rather than
    zero: n < 3, or either variable is constant (zero rank-variance). A
    constant variable has no monotone ordering, so a 1.0 / 0.0 score there is
    meaningless - ``None`` lets callers mark it N/A instead of silently
    reporting a perfect/poor correlation (e.g. ``_spearman([5,5,5],[1,2,3])``
    is ``None``, not 1.0).
    """
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def ranks(v: Sequence[float]) -> List[float]:
        # average ranks for ties: every member of a tie-group gets the mean of
        # the positions that group occupies (1-based positions averaged).
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0  # mean of tied 0-based positions
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    if vx <= 0.0 or vy <= 0.0:  # a constant variable -> correlation undefined
        return None
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    return round(num / ((vx * vy) ** 0.5), 4)


def _auroc(scores: List[float], pos: List[int]) -> Optional[float]:
    """AUROC via the Mann-Whitney U statistic (ties counted as 0.5).

    Returns ``None`` when either class is empty - the metric is *unmeasurable*,
    which is not the same as the worst possible score (0.0). Returning 0.0 here
    coerces "no negatives to rank against" into "model failed", which is what
    made the subfamily-holdout look catastrophic on degenerate splits. Callers
    must treat ``None`` as N/A.
    """
    p = [s for s, y in zip(scores, pos) if y == 1]
    n = [s for s, y in zip(scores, pos) if y == 0]
    if not p or not n:
        return None
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
    # None = not measurable (no labelled activity / a class is empty); this is
    # distinct from a genuine 0.0 metric. Downstream must treat None as N/A.
    spearman: Optional[float] = None
    auroc: Optional[float] = None
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
    base_val = base.get(metric)
    rows = [{"ablation": "full_model", metric: base_val}]
    for t in toggles:
        if t not in _OV:
            continue
        r = _run(f"no_{t}", _OV[t])
        r_val = r.get(metric)
        # delta is only defined when BOTH runs measured the metric; None means
        # "not comparable" rather than a fabricated 0.0 difference.
        delta = (round(r_val - base_val, 4)
                 if isinstance(r_val, (int, float))
                 and isinstance(base_val, (int, float)) else None)
        rows.append({
            "ablation": f"no_{t}",
            metric: r_val,
            f"delta_{metric}": delta,
        })
    return {"metric": metric, "baseline": base, "ablation": rows}
