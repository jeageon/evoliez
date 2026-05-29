"""Stacked-bar waterfall of score components for the top-K candidates."""

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

# Order matters - components stack in this sequence so the legend reads
# consistently across runs.  The canonical ddG component key is
# ``stability_ddg`` (the production CSV column emitted by
# ``evoliez.io.report``); ``ddg_fold`` is a legacy alias kept so older
# CSVs that wrote the per-column value under ``ddg_fold`` still plot.
SCORE_COMPONENTS: Tuple[str, ...] = (
    "ml_score",
    "stability_ddg",
    "docking_score",
    "md_lite_score",
    "plif_recovery",
)

# Per-column aliases: ``canonical_key -> (other accepted column names)``.
# When reading a row we try the canonical key first, then each alias, so
# a value written under any of these headers maps onto the same stacked
# bar / legend entry.
_COMPONENT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "stability_ddg": ("ddg_fold",),
}

# Wong-palette assignment per component (kept stable so the legend is the
# same across runs).
_COMPONENT_COLORS: Dict[str, str] = {
    "ml_score": "#0072B2",
    "stability_ddg": "#E69F00",
    "docking_score": "#56B4E9",
    "md_lite_score": "#CC79A7",
    "plif_recovery": "#F0E442",
}


def _parse_breakdown(raw: str) -> Dict[str, float]:
    """Try JSON, then a loose ``k=v;k=v`` form."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            out: Dict[str, float] = {}
            for k, v in data.items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return out
    except (ValueError, TypeError):
        pass
    out2: Dict[str, float] = {}
    for token in raw.replace(",", ";").split(";"):
        token = token.strip()
        if not token or "=" not in token:
            continue
        k, v = token.split("=", 1)
        try:
            out2[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out2


def _canonical_component(key: str) -> str:
    """Map a possibly-legacy component key onto its canonical name.

    e.g. ``ddg_fold`` (legacy in-memory / old-CSV key) -> ``stability_ddg``
    (the production CSV column). Unknown keys pass through unchanged.
    """
    for canon, aliases in _COMPONENT_ALIASES.items():
        if key == canon or key in aliases:
            return canon
    return key


def _normalize_breakdown(br: Dict[str, float]) -> Dict[str, float]:
    """Collapse legacy alias keys onto their canonical component names.

    The canonical key wins if both it and an alias are present.
    """
    out: Dict[str, float] = {}
    for k, v in br.items():
        canon = _canonical_component(str(k))
        # First-seen canonical wins; an explicit canonical key (read first
        # in dict order is not guaranteed, so prefer to overwrite only when
        # the incoming key IS the canonical one).
        if canon in out and str(k) != canon:
            continue
        out[canon] = v
    return out


def _row_breakdown(row: Dict[str, str]) -> Dict[str, float]:
    """Pull score components out of a single CSV row.

    Prefers the embedded JSON ``details.score_breakdown`` if present, falling
    back to the per-column numeric values. Legacy alias keys (e.g.
    ``ddg_fold``) are normalized to canonical names (``stability_ddg``).
    """
    for key in ("score_breakdown", "details.score_breakdown"):
        raw = row.get(key)
        if raw:
            br = _parse_breakdown(raw)
            if br:
                return _normalize_breakdown(br)
    details_raw = row.get("details")
    if details_raw:
        try:
            details = json.loads(details_raw)
        except (ValueError, TypeError):
            details = None
        if isinstance(details, dict):
            br = details.get("score_breakdown")
            if isinstance(br, dict):
                out: Dict[str, float] = {}
                for k, v in br.items():
                    try:
                        out[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                if out:
                    return _normalize_breakdown(out)

    out2: Dict[str, float] = {}
    for comp in SCORE_COMPONENTS:
        # Read the canonical column first, then any legacy aliases, so a
        # value written under either header lands on the same component.
        raw = row.get(comp)
        if raw in (None, "", "NA", "nan"):
            for alias in _COMPONENT_ALIASES.get(comp, ()):  # legacy columns
                raw = row.get(alias)
                if raw not in (None, "", "NA", "nan"):
                    break
        if raw in (None, "", "NA", "nan"):
            continue
        try:
            out2[comp] = float(raw)
        except (TypeError, ValueError):
            continue
    return out2


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    top_k: int = 10,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Horizontal stacked-bar score waterfall for the top-K candidates."""
    csv_path = getattr(artifacts, "final_candidates_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        _LOGGER.warning("score_waterfall: final_candidates_csv missing")
        return None
    rows: List[Dict[str, str]] = []
    try:
        with Path(csv_path).open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                rows.append(r)
    except OSError:
        _LOGGER.warning("score_waterfall: cannot read %s", csv_path)
        return None
    if not rows:
        _LOGGER.warning("score_waterfall: %s is empty", csv_path)
        return None

    # Sort by final_score (or rank if available) descending.
    def _score_key(row: Dict[str, str]) -> float:
        for k in ("final_score", "score", "ml_score"):
            v = row.get(k)
            if v:
                try:
                    return -float(v)
                except (TypeError, ValueError):
                    continue
        rank = row.get("rank")
        try:
            return float(rank) if rank else 0.0
        except (TypeError, ValueError):
            return 0.0

    rows.sort(key=_score_key)
    rows = rows[:top_k]

    candidate_ids: List[str] = []
    breakdowns: List[Dict[str, float]] = []
    evidence_classes: List[str] = []
    for r in rows:
        cid = r.get("candidate_id") or r.get("id") or r.get("mutation") or "?"
        candidate_ids.append(str(cid))
        breakdowns.append(_row_breakdown(r))
        evidence_classes.append(r.get("evidence_class") or "Uncertain")

    # All components seen across rows, in our preferred order first.
    seen: List[str] = []
    for b in breakdowns:
        for k in b:
            if k not in seen:
                seen.append(k)
    components = [c for c in SCORE_COMPONENTS if c in seen]
    components += [c for c in seen if c not in components]
    if not components:
        _LOGGER.warning("score_waterfall: no score components found")
        return None

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433
    from matplotlib.patches import Patch
    import numpy as np
    n = len(candidate_ids)
    y_positions = np.arange(n)[::-1]  # top candidate at top

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.45 * n + 1.5)))
    # Track per-row positive / negative offsets so negative contributions
    # extend to the left of zero.
    pos_offset = np.zeros(n, dtype=float)
    neg_offset = np.zeros(n, dtype=float)
    for comp in components:
        vals = np.asarray(
            [b.get(comp, 0.0) for b in breakdowns], dtype=float
        )
        positives = np.where(vals >= 0, vals, 0.0)
        negatives = np.where(vals < 0, vals, 0.0)
        color = _COMPONENT_COLORS.get(comp, "#888888")
        if positives.any():
            ax.barh(
                y_positions,
                positives,
                left=pos_offset,
                color=color,
                edgecolor="white",
                linewidth=0.4,
            )
            pos_offset = pos_offset + positives
        if negatives.any():
            ax.barh(
                y_positions,
                negatives,
                left=neg_offset,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                hatch="//",
            )
            neg_offset = neg_offset + negatives

    # Evidence-class strip alongside the bars.
    for i, ec in enumerate(evidence_classes):
        ax.scatter(
            max(pos_offset[i], 0.0) + 0.02 * max(pos_offset.max(), 1.0),
            y_positions[i],
            color=evidence_color(ec),
            s=80,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(candidate_ids)
    ax.set_xlabel("score contribution")
    ax.set_title(f"Top-{n} candidate score breakdown")
    ax.axvline(0, color="#444444", linewidth=0.8)

    legend_handles = [
        Patch(facecolor=_COMPONENT_COLORS.get(c, "#888888"), label=c)
        for c in components
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        fontsize="small",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="07_score_waterfall",
        section="reranking",
        title=f"Top-{n} score waterfall",
        description=(
            "Horizontal stacked-bar breakdown of score contributions for the "
            "top-ranked candidates; evidence class shown beside each bar."
        ),
        path=out_path,
        source_files=[Path(csv_path)],
        renderer="matplotlib",
        params={
            "top_k": top_k,
            "n_shown": n,
            "components": components,
            "style": style,
        },
    )
