"""Recall@K vs K plot for the external-benchmark validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evoliez.figures.plots import apply_style_and_get_dpi
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


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Recall@K curve with a random-baseline overlay; gated on validity."""
    bench_json = getattr(artifacts, "benchmark_json", None)
    bench_csv = getattr(artifacts, "benchmark_csv", None)
    if bench_json is None or bench_csv is None:
        _LOGGER.warning(
            "benchmark_recovery: benchmark_json or benchmark_csv missing, skipping"
        )
        return None
    bench_json = Path(bench_json)
    bench_csv = Path(bench_csv)
    if not bench_json.exists() or not bench_csv.exists():
        _LOGGER.warning("benchmark_recovery: input file does not exist")
        return None

    try:
        data = json.loads(bench_json.read_text())
    except (OSError, ValueError):
        _LOGGER.warning("benchmark_recovery: failed to parse %s", bench_json)
        return None
    if not isinstance(data, dict):
        return None

    if data.get("valid") is False:
        _LOGGER.warning(
            "benchmark_recovery: benchmark.json marked invalid, skipping"
        )
        return None

    pairs = _coerce_recall_at_k(data)
    if not pairs:
        _LOGGER.warning("benchmark_recovery: no recall@K curve in payload")
        return None

    ks = [p[0] for p in pairs]
    recalls = [p[1] for p in pairs]

    # Random baseline: if we know the labelled-positive fraction we can use
    # it, otherwise fall back to K / N (uniform-prior) when N is provided.
    n_total = data.get("n_total") or data.get("library_size") or data.get("n_candidates")
    n_positives = data.get("n_positives") or data.get("n_active")
    baseline: Optional[List[float]] = None
    if isinstance(n_total, (int, float)) and n_total > 0:
        if isinstance(n_positives, (int, float)) and n_positives > 0:
            # Hypergeometric expectation simplifies to K * P / N
            baseline = [min(1.0, k * float(n_positives) / float(n_total) / max(float(n_positives), 1.0)) for k in ks]
            # That formula collapses to K/N; recall is matches/positives so:
            baseline = [min(1.0, k / float(n_total)) for k in ks]
        else:
            baseline = [min(1.0, k / float(n_total)) for k in ks]

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433

    fig, ax = plt.subplots()
    ax.plot(ks, recalls, marker="o", color="#0072B2", linewidth=2.0, label="EvoLiEZ")
    if baseline is not None:
        ax.plot(
            ks,
            baseline,
            linestyle="--",
            color="#999999",
            label="random baseline",
        )
    ax.set_xlabel("K")
    ax.set_ylabel("recall@K")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    bench_label = bench_csv.name
    ax.set_title(f"Benchmark: {bench_label}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="00_benchmark_recovery",
        section="overview",
        title="Benchmark recovery (recall@K)",
        description=(
            "Recall@K curve on the external benchmark vs. a random-pick "
            "baseline. Plot only emitted when the benchmark is marked valid."
        ),
        path=out_path,
        source_files=[bench_json, bench_csv],
        renderer="matplotlib",
        params={
            "k_values": ks,
            "recall_values": recalls,
            "benchmark_csv": bench_csv.name,
            "style": style,
        },
    )
