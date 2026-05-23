"""Paper-grade benchmark recovery figure (Figure 3 of the spec).

4-panel layout that lets reviewers verify the pipeline's retrospective
recovery of literature-known mutations:

    A.  Rank distribution of known beneficial mutations (histogram)
    B.  Recall@K curve with random-baseline overlay
    C.  Per-mutation rank percentile (dot plot, beneficial vs
        deleterious vs neutral colored by label)
    D.  Ablation bars (if benchmark.json carries ablation data)

Universal: every panel reads from ``benchmark.csv`` (any enzyme) plus
``final_candidates.csv`` (pipeline output) and computes metrics inline,
so the figure renders even when the user hasn't pre-run ``evoliez
bench``. The mutation-label scheme (beneficial/neutral/deleterious) is
the same across enzymes - no PseFDH-specific assumptions.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.style import evidence_color
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)


def _coerce_recall_at_k(data: Dict[str, Any]) -> List[Tuple[int, float]]:
    """Extract (K, recall) pairs from a benchmark.json payload.

    Tolerates several shapes:
        - {"recall_at_k": {"1": 0.1, "5": 0.4, ...}}
        - {"recall_at_k": [{"k": 1, "recall": 0.1}, ...]}
        - {"recall": {"1": 0.1, ...}}
        - top-level {"1": 0.1, "5": 0.4, ...}
    """
    candidates = (
        data.get("recall_at_k"),
        data.get("recall"),
        data.get("recall_curve"),
    )
    for blob in candidates:
        if isinstance(blob, dict):
            pairs: List[Tuple[int, float]] = []
            for k, v in blob.items():
                try:
                    pairs.append((int(k), float(v)))
                except (TypeError, ValueError):
                    continue
            if pairs:
                pairs.sort(key=lambda p: p[0])
                return pairs
        if isinstance(blob, list):
            pairs2: List[Tuple[int, float]] = []
            for item in blob:
                if not isinstance(item, dict):
                    continue
                k = item.get("k") or item.get("K")
                r = item.get("recall") or item.get("value")
                try:
                    pairs2.append((int(k), float(r)))
                except (TypeError, ValueError):
                    continue
            if pairs2:
                pairs2.sort(key=lambda p: p[0])
                return pairs2

    # Last resort: top-level integer keys
    pairs3: List[Tuple[int, float]] = []
    for k, v in data.items():
        try:
            pairs3.append((int(k), float(v)))
        except (TypeError, ValueError):
            continue
    pairs3.sort(key=lambda p: p[0])
    return pairs3


def _load_benchmark_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Read ``benchmark.csv`` rows. Accepts the schema
    ``mutation,label,activity[,source]`` and tolerates extra columns.
    Bad rows are silently dropped."""
    try:
        with csv_path.open("r", newline="") as fh:
            rdr = csv.DictReader(fh)
            rows: List[Dict[str, Any]] = []
            for raw in rdr:
                mut = (raw.get("mutation") or raw.get("mutations") or "").strip()
                lbl = (raw.get("label") or "").strip().lower()
                if not mut or not lbl:
                    continue
                try:
                    act = float(raw.get("activity") or 0.0)
                except (TypeError, ValueError):
                    act = 0.0
                rows.append({"mutation": mut, "label": lbl, "activity": act})
            return rows
    except OSError:
        return []


def _load_candidate_ranks(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    """``final_candidates.csv`` -> {mutation_str: {rank, is_blocked, ...}}.

    The CSV's ``mutations`` column is the canonical mutation string
    (e.g. ``"D222N"`` for a single mutant or ``"D222Q;G286N"`` for a
    double); we key by that so benchmark lookups work directly.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with csv_path.open("r", newline="") as fh:
            for r in csv.DictReader(fh):
                mut = (r.get("mutations") or "").strip()
                if not mut:
                    continue
                try:
                    rank = int(r.get("rank") or 0)
                except (TypeError, ValueError):
                    rank = 0
                out[mut] = {
                    "rank": rank,
                    "is_blocked": str(r.get("is_blocked", "0")) in ("1", "True", "true"),
                    "evidence_class": (r.get("evidence_class") or "").strip(),
                    "final_score": r.get("final_score", ""),
                }
    except OSError:
        pass
    return out


def _compute_recall_at_k(
    beneficial_muts: List[str],
    ranked_muts_in_order: List[str],
    ks: List[int],
) -> List[float]:
    """recall@K = (beneficial mutations with rank <= K) / total beneficial."""
    if not beneficial_muts:
        return [0.0] * len(ks)
    top_at_k = {k: set(ranked_muts_in_order[:k]) for k in ks}
    total = len(beneficial_muts)
    return [
        sum(1 for m in beneficial_muts if m in top_at_k[k]) / total for k in ks
    ]


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    ks: Tuple[int, ...] = (1, 5, 10, 30, 50, 100),
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Paper-grade 4-panel benchmark recovery figure.

    Gated on ``benchmark.csv`` AND ``final_candidates.csv`` being
    present. ``benchmark.json`` is read OPPORTUNISTICALLY for ablation
    data (panel D); if absent, the ablation panel is omitted and we
    render a 3-panel layout instead.

    Universal: works for any enzyme whose benchmark CSV uses the
    standard schema (``mutation,label,activity[,source]``). No
    PseFDH-specific logic.
    """
    bench_csv = getattr(artifacts, "benchmark_csv", None)
    final_csv = getattr(artifacts, "final_candidates_csv", None)
    if bench_csv is None or final_csv is None:
        _LOGGER.warning(
            "benchmark_recovery: benchmark_csv or final_candidates_csv "
            "missing, skipping (Figure 3 needs both)"
        )
        return None
    bench_csv = Path(bench_csv)
    final_csv = Path(final_csv)
    if not bench_csv.exists() or not final_csv.exists():
        _LOGGER.warning("benchmark_recovery: input file does not exist")
        return None

    bench_rows = _load_benchmark_rows(bench_csv)
    if not bench_rows:
        _LOGGER.warning(
            "benchmark_recovery: empty / unreadable %s, skipping", bench_csv,
        )
        return None
    cand_ranks = _load_candidate_ranks(final_csv)
    if not cand_ranks:
        _LOGGER.warning(
            "benchmark_recovery: empty / unreadable %s, skipping", final_csv,
        )
        return None

    # ---- prepare per-mutation enrichment table ----------------------------
    # For each benchmark row, find its rank in the pipeline output.
    # Missing-from-CSV gets rank = inf (i.e. not recovered at all).
    n_cand = max((d["rank"] for d in cand_ranks.values()), default=0)
    enriched: List[Dict[str, Any]] = []
    for r in bench_rows:
        info = cand_ranks.get(r["mutation"])
        rank = info["rank"] if info and info["rank"] > 0 else n_cand + 1
        pct = 100.0 * (1.0 - (rank - 1) / max(1, n_cand)) if rank <= n_cand else 0.0
        enriched.append({
            **r,
            "rank": rank,
            "percentile": round(pct, 1),
            "is_blocked": (info or {}).get("is_blocked", False),
            "evidence_class": (info or {}).get("evidence_class", ""),
        })

    beneficial = [r["mutation"] for r in bench_rows if r["label"] == "beneficial"]
    deleterious = [r["mutation"] for r in bench_rows if r["label"] == "deleterious"]
    n_total_bench = len(bench_rows)
    n_recovered = sum(1 for r in enriched if r["rank"] <= n_cand)

    # ---- ranked mutation list (in CSV rank order, ACCEPTED only) ---------
    # Re-sort: lowest rank first. Use raw csv ranking which already obeys
    # the P0a accepted-then-blocked partition.
    ranked_pairs = sorted(
        [(d["rank"], m) for m, d in cand_ranks.items()
         if not d["is_blocked"]],
        key=lambda p: p[0],
    )
    ranked_muts = [m for _r, m in ranked_pairs]
    k_axis = [k for k in ks if k <= max(n_cand, 1)] or [n_cand]
    recall_vals = _compute_recall_at_k(beneficial, ranked_muts, k_axis)
    baseline = [min(1.0, k / max(1, n_cand)) for k in k_axis]

    # ---- optional ablation from benchmark.json --------------------------
    ablation: Optional[List[Tuple[str, float]]] = None
    bench_json = getattr(artifacts, "benchmark_json", None)
    if bench_json is not None and Path(bench_json).exists():
        try:
            data = json.loads(Path(bench_json).read_text())
            ab = data.get("ablation_study") or data.get("ablation") or {}
            if isinstance(ab, dict) and ab:
                rows = []
                for variant, payload in ab.items():
                    if isinstance(payload, dict):
                        val = (payload.get("auroc_beneficial")
                               or payload.get("auroc")
                               or payload.get("recall_at_k"))
                        if isinstance(val, (int, float)):
                            rows.append((str(variant), float(val)))
                if rows:
                    ablation = sorted(rows, key=lambda r: -r[1])
        except (OSError, ValueError):
            ablation = None

    # ---- render ----------------------------------------------------------
    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt   # noqa: WPS433

    n_panels = 4 if ablation else 3
    # Larger panel width + more wspace so panel titles don't run into the
    # neighbouring panel's y-axis labels; top margin reserved for the
    # suptitle via constrained_layout.
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.2 * n_panels, 4.2),
        gridspec_kw={"wspace": 0.45},
        constrained_layout=True,
    )
    ax_rank, ax_recall, ax_pct, *rest = axes
    ax_abl = rest[0] if rest else None

    # Panel A — rank distribution of beneficial vs deleterious
    ben_ranks = [r["rank"] for r in enriched if r["label"] == "beneficial"]
    del_ranks = [r["rank"] for r in enriched if r["label"] == "deleterious"]
    bins = max(8, min(20, n_cand // 10)) if n_cand > 20 else 10
    if ben_ranks:
        ax_rank.hist(ben_ranks, bins=bins, alpha=0.7,
                     color=evidence_color("Strong"), label=f"beneficial (n={len(ben_ranks)})")
    if del_ranks:
        ax_rank.hist(del_ranks, bins=bins, alpha=0.7,
                     color=evidence_color("Reject"),
                     label=f"deleterious (n={len(del_ranks)})")
    ax_rank.set_xlabel("rank in pipeline output")
    ax_rank.set_ylabel("count")
    ax_rank.set_title("A. Rank distribution", fontsize=10)
    ax_rank.legend(loc="upper right", fontsize="small")
    ax_rank.grid(True, alpha=0.3)

    # Panel B — recall@K curve
    ax_recall.plot(k_axis, recall_vals, marker="o", color="#0072B2",
                   linewidth=2.0, label="EvoLiEZ")
    ax_recall.plot(k_axis, baseline, linestyle="--", color="#999999",
                   label="random baseline (K/N)")
    ax_recall.set_xlabel("K")
    ax_recall.set_ylabel("recall@K")
    ax_recall.set_ylim(0, 1.05)
    ax_recall.set_title(
        f"B. Recall@K (n_ben={len(beneficial)}, N={n_cand})",
        fontsize=10,
    )
    ax_recall.legend(loc="lower right", fontsize="small")
    ax_recall.grid(True, alpha=0.3)

    # Panel C — per-mutation percentile (dot strip per label)
    LABEL_ORDER = ("beneficial", "neutral", "deleterious")
    LABEL_COLOR = {
        "beneficial": evidence_color("Strong"),
        "neutral":    evidence_color("Promising"),
        "deleterious": evidence_color("Reject"),
    }
    for i, lab in enumerate(LABEL_ORDER):
        ys = [r["percentile"] for r in enriched if r["label"] == lab]
        if not ys:
            continue
        xs = [i + (j - len(ys) / 2) * 0.04 for j in range(len(ys))]
        ax_pct.scatter(xs, ys, s=80, color=LABEL_COLOR[lab],
                       edgecolor="black", linewidths=0.5, zorder=3,
                       label=f"{lab} (n={len(ys)})")
        # Annotate top-3 names so reviewers can find the documented hits.
        for j, r in enumerate([r for r in enriched if r["label"] == lab][:3]):
            ax_pct.annotate(r["mutation"], (xs[j], r["percentile"]),
                            xytext=(5, 3), textcoords="offset points",
                            fontsize=7, alpha=0.8)
    ax_pct.set_xticks(range(len(LABEL_ORDER)))
    ax_pct.set_xticklabels(LABEL_ORDER, rotation=15, ha="right")
    ax_pct.set_ylabel("rank percentile (100 = best)")
    ax_pct.set_ylim(-5, 105)
    ax_pct.axhline(50, color="#999999", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_pct.set_title(
        f"C. Per-mutation percentile ({n_recovered}/{n_total_bench})",
        fontsize=10,
    )
    ax_pct.legend(loc="lower left", fontsize="small")
    ax_pct.grid(True, axis="y", alpha=0.3)

    # Panel D — ablation (when available)
    if ax_abl is not None and ablation:
        names = [n for n, _ in ablation]
        vals = [v for _, v in ablation]
        bars = ax_abl.barh(range(len(names)), vals, color="#0072B2")
        ax_abl.set_yticks(range(len(names)))
        ax_abl.set_yticklabels(names, fontsize=8)
        ax_abl.invert_yaxis()
        ax_abl.set_xlabel("AUROC / recall metric")
        ax_abl.set_xlim(0, max(1.0, max(vals) * 1.1))
        ax_abl.set_title("D. Ablation", fontsize=10)
        ax_abl.grid(True, axis="x", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax_abl.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", va="center", fontsize=7)

    # constrained_layout owns the spacing budget; suptitle sits above
    # the panel band without `y=` overriding the auto-layout.
    fig.suptitle(
        f"Benchmark recovery — {bench_csv.name}",
        fontsize=12, weight="bold",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" + constrained_layout can clash; rely on
    # constrained_layout's own margin handling so the suptitle and panel
    # titles aren't trimmed.
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return FigureSpec(
        figure_id="00_benchmark_recovery",
        section="overview",
        title="Benchmark recovery (4-panel)",
        description=(
            f"Paper-grade benchmark recovery: rank distribution + "
            f"recall@K + per-mutation percentile"
            f"{' + ablation' if ablation else ''}. "
            f"{n_recovered}/{n_total_bench} known mutations recovered in "
            f"the pipeline's top {n_cand}; "
            f"recall@10 = {recall_vals[k_axis.index(10)]:.2f} "
            f"(random baseline {baseline[k_axis.index(10)]:.2f}) "
            f"when applicable."
        ) if 10 in k_axis else (
            f"Benchmark recovery (3-panel; K=10 outside library size). "
            f"{n_recovered}/{n_total_bench} known mutations recovered."
        ),
        path=out_path,
        source_files=[bench_csv, final_csv],
        renderer="matplotlib",
        params={
            "k_values": list(k_axis),
            "recall_values": list(recall_vals),
            "random_baseline": list(baseline),
            "n_benchmark": int(n_total_bench),
            "n_beneficial": int(len(beneficial)),
            "n_deleterious": int(len(deleterious)),
            "n_recovered": int(n_recovered),
            "n_candidates": int(n_cand),
            "ablation_panel": bool(ablation),
            "benchmark_csv": bench_csv.name,
            "style": style,
        },
    )
