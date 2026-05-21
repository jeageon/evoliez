"""Donut chart of candidate counts by evidence class."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.style import EVIDENCE_COLORS, evidence_color
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

EVIDENCE_ORDER = ("Strong", "Promising", "Uncertain", "Reject")


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Donut chart of the final candidate set's evidence-class breakdown."""
    csv_path = getattr(artifacts, "final_candidates_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        _LOGGER.warning("evidence_distribution: final_candidates_csv missing")
        return None

    counts: Counter = Counter()
    try:
        with Path(csv_path).open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                ec = (row.get("evidence_class") or "Uncertain").strip() or "Uncertain"
                counts[ec] += 1
    except OSError:
        _LOGGER.warning("evidence_distribution: cannot read %s", csv_path)
        return None

    if not counts:
        _LOGGER.warning("evidence_distribution: no rows in %s", csv_path)
        return None

    ordered = [c for c in EVIDENCE_ORDER if counts.get(c, 0) > 0]
    ordered += [c for c in counts if c not in EVIDENCE_ORDER]
    values = [counts[c] for c in ordered]
    colors = [EVIDENCE_COLORS.get(c, evidence_color(c)) for c in ordered]
    total = sum(values)

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        wedgeprops={"width": 0.35, "edgecolor": "white"},
    )
    ax.set_aspect("equal")
    ax.text(
        0,
        0.05,
        f"{total}",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
    )
    ax.text(0, -0.15, "candidates", ha="center", va="center", fontsize=11)
    ax.set_title("Evidence-class distribution")

    legend_labels = [f"{c} ({counts[c]})" for c in ordered]
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="00_evidence_distribution",
        section="overview",
        title="Evidence-class distribution",
        description="Donut chart of final candidates by evidence class.",
        path=out_path,
        source_files=[Path(csv_path)],
        renderer="matplotlib",
        params={
            "counts": dict(counts),
            "total": total,
            "style": style,
        },
    )
