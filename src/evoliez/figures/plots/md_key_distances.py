"""Key catalytic-distance plot for the MD validation section.

Two render modes, automatically selected per the available payload in
each candidate's ``analysis.json``:

* **Time series** (preferred): if any candidate exposes
  ``key_distances`` -- a ``{distance_name: [series of floats]}`` dict --
  we emit a multi-panel line plot (one panel per distinct distance, one
  line per top-N candidate, colored by ``evidence_class``).  This is the
  expert-plan deliverable for section 8.
* **Summary bars** (fallback): when no series payload is available but
  candidates ship ``catalytic_distance_mean`` (plus optional
  ``catalytic_distance_std``), we plot horizontal bars per candidate so
  the report still gives a quantitative view of catalytic geometry.

Returns ``None`` when neither mode is feasible -- nothing on disk,
nothing in the manifest, and the section template degrades to its empty
state.  Matplotlib is imported lazily inside :func:`render` so the
module is safe to import in a barebones env.
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

# Cap the number of candidate lines per panel so the chart stays
# legible. The plan calls out "top-N candidate" -- 8 matches the
# Wong-palette length used elsewhere in the report.
_MAX_CANDIDATES = 8


def _load_analysis(md_dir: Path) -> Optional[Dict[str, Any]]:
    candidate = md_dir / "analysis.json"
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text())
    except (OSError, ValueError):
        return None


def _coerce_series(value: Any) -> Optional[List[float]]:
    """Best-effort cast ``value`` to ``List[float]`` (or ``None``)."""
    if not isinstance(value, list) or not value:
        return None
    out: List[float] = []
    for x in value:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return None
    return out


def _time_axis(data: Dict[str, Any], n: int) -> List[float]:
    for k in ("time_ps", "time", "frames_ps", "t_ps"):
        v = data.get(k)
        if isinstance(v, list) and len(v) == n:
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                continue
    return list(range(n))


def _evidence_class_for(
    artifacts: ReportArtifacts, candidate_id: str
) -> str:
    csv_path = getattr(artifacts, "final_candidates_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        return "Uncertain"
    try:
        with Path(csv_path).open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("candidate_id") == candidate_id:
                    return row.get("evidence_class") or "Uncertain"
    except OSError:
        return "Uncertain"
    return "Uncertain"


def _ordered_candidates(
    artifacts: ReportArtifacts, md_dirs: Dict[str, Path]
) -> List[str]:
    """Order candidates by ``final_score`` from the candidates CSV (best first).

    Falls back to MD-dir iteration order when the CSV is missing or
    doesn't reference the MD candidate ids.
    """
    csv_path = getattr(artifacts, "final_candidates_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        return list(md_dirs.keys())
    ranked: List[Tuple[float, str]] = []
    try:
        with Path(csv_path).open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                cid = row.get("candidate_id")
                if not cid or cid not in md_dirs:
                    continue
                try:
                    score = float(row.get("final_score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                ranked.append((-score, cid))
    except OSError:
        return list(md_dirs.keys())
    if not ranked:
        return list(md_dirs.keys())
    seen = set()
    ordered: List[str] = []
    for _, cid in sorted(ranked):
        if cid in seen:
            continue
        seen.add(cid)
        ordered.append(cid)
    # Append any MD dirs the CSV didn't mention so we never silently drop
    # a candidate that has a trajectory.
    for cid in md_dirs:
        if cid not in seen:
            ordered.append(cid)
    return ordered


def _collect_series_payloads(
    artifacts: ReportArtifacts, md_dirs: Dict[str, Path]
) -> Tuple[List[str], Dict[str, Dict[str, List[float]]], Dict[str, List[float]], List[Path]]:
    """Walk md_dirs and pull ``key_distances`` series + time axes.

    Returns
    -------
    distances:
        Ordered list of distance names encountered (first-seen order).
    per_cand:
        ``{candidate_id: {distance_name: series}}``.
    per_cand_time:
        ``{candidate_id: time_axis}`` (one shared axis per candidate,
        derived from the longest series the candidate exposed).
    sources:
        Paths to every ``analysis.json`` that contributed data.
    """
    distances: List[str] = []
    seen_dist: set = set()
    per_cand: Dict[str, Dict[str, List[float]]] = {}
    per_cand_time: Dict[str, List[float]] = {}
    sources: List[Path] = []

    for cid, md_dir in md_dirs.items():
        data = _load_analysis(Path(md_dir))
        if data is None:
            continue
        raw = data.get("key_distances")
        if not isinstance(raw, dict) or not raw:
            continue
        cand_series: Dict[str, List[float]] = {}
        for name, values in raw.items():
            series = _coerce_series(values)
            if series is None or len(series) < 2:
                continue
            cand_series[str(name)] = series
            if name not in seen_dist:
                seen_dist.add(str(name))
                distances.append(str(name))
        if not cand_series:
            continue
        per_cand[cid] = cand_series
        # Use the longest series to anchor the time axis.
        n = max(len(s) for s in cand_series.values())
        per_cand_time[cid] = _time_axis(data, n)
        sources.append(Path(md_dir) / "analysis.json")

    return distances, per_cand, per_cand_time, sources


def _collect_summary_payloads(
    md_dirs: Dict[str, Path],
) -> Tuple[List[Tuple[str, float, float]], List[Path]]:
    """Walk md_dirs for ``catalytic_distance_mean`` (+ optional ``_std``)."""
    rows: List[Tuple[str, float, float]] = []
    sources: List[Path] = []
    for cid, md_dir in md_dirs.items():
        data = _load_analysis(Path(md_dir))
        if data is None:
            continue
        mean = data.get("catalytic_distance_mean")
        if mean is None:
            continue
        try:
            mean_f = float(mean)
        except (TypeError, ValueError):
            continue
        try:
            std_f = float(data.get("catalytic_distance_std") or 0.0)
        except (TypeError, ValueError):
            std_f = 0.0
        rows.append((cid, mean_f, std_f))
        sources.append(Path(md_dir) / "analysis.json")
    return rows, sources


def _render_series(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str,
    distances: List[str],
    per_cand: Dict[str, Dict[str, List[float]]],
    per_cand_time: Dict[str, List[float]],
    sources: List[Path],
    ordered_cids: List[str],
) -> Optional[FigureSpec]:
    """Multi-panel line plot, one panel per distance."""
    keep_cids = [c for c in ordered_cids if c in per_cand][:_MAX_CANDIDATES]
    if not keep_cids:
        return None

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433

    n_panels = len(distances)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(8, max(3, 2.2 * n_panels)),
        sharex=True,
        squeeze=False,
    )

    for ax, dname in zip(axes[:, 0], distances):
        for cid in keep_cids:
            cand_series = per_cand.get(cid, {})
            series = cand_series.get(dname)
            if not series:
                continue
            t = per_cand_time.get(cid) or list(range(len(series)))
            n = min(len(t), len(series))
            if n < 2:
                continue
            ec = _evidence_class_for(artifacts, cid)
            color = evidence_color(ec)
            ax.plot(
                t[:n],
                series[:n],
                color=color,
                linewidth=1.5,
                label=cid,
                alpha=0.85,
            )
        ax.set_ylabel(f"{dname} (A)")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("time (ps)")
    axes[0, 0].set_title("MD key catalytic distances")
    if len(keep_cids) <= _MAX_CANDIDATES:
        axes[0, 0].legend(loc="upper right", fontsize="small", ncol=2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="08_md_key_distances",
        section="md",
        title="MD key catalytic distances",
        description=(
            "Time series of catalytic / key distances over the MD "
            "trajectory, one panel per distance, colored by "
            "evidence class."
        ),
        path=out_path,
        source_files=sources,
        renderer="matplotlib",
        params={
            "mode": "timeseries",
            "n_candidates": len(keep_cids),
            "n_distances": n_panels,
            "style": style,
        },
    )


def _render_summary_bars(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str,
    rows: List[Tuple[str, float, float]],
    sources: List[Path],
    ordered_cids: List[str],
) -> Optional[FigureSpec]:
    """Horizontal bar chart: catalytic distance mean +/- std per candidate."""
    by_cid: Dict[str, Tuple[float, float]] = {r[0]: (r[1], r[2]) for r in rows}
    keep_cids = [c for c in ordered_cids if c in by_cid][:_MAX_CANDIDATES]
    if not keep_cids:
        # ordered_cids may have missed CSV order; fall back to first N
        # discovered rows.
        keep_cids = [r[0] for r in rows[:_MAX_CANDIDATES]]
    if not keep_cids:
        return None

    means = [by_cid[c][0] for c in keep_cids]
    stds = [by_cid[c][1] for c in keep_cids]
    colors = [evidence_color(_evidence_class_for(artifacts, c)) for c in keep_cids]

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(keep_cids) + 1.5)))
    y_pos = list(range(len(keep_cids)))
    ax.barh(
        y_pos,
        means,
        xerr=stds,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keep_cids)
    ax.invert_yaxis()  # best at the top
    ax.set_xlabel("catalytic distance (A)")
    ax.set_title("MD catalytic distance (mean +/- std)")
    ax.grid(True, axis="x", alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="08_md_key_distances",
        section="md",
        title="MD key catalytic distances",
        description=(
            "Mean catalytic distance per candidate (error bars: std). "
            "Time-series key_distances unavailable - showing summary "
            "statistics from analysis.json."
        ),
        path=out_path,
        source_files=sources,
        renderer="matplotlib",
        params={
            "mode": "summary_bars",
            "n_candidates": len(keep_cids),
            "style": style,
        },
    )


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Key-distance plot for the MD validation section.

    Returns ``None`` when no MD ``analysis.json`` has either a usable
    ``key_distances`` mapping or a ``catalytic_distance_mean`` summary.
    """
    md_dirs = getattr(artifacts, "md_dirs", None) or {}
    if not md_dirs:
        _LOGGER.warning("md_key_distances: no MD directories, skipping")
        return None

    ordered = _ordered_candidates(artifacts, md_dirs)

    distances, per_cand, per_cand_time, series_sources = _collect_series_payloads(
        artifacts, md_dirs
    )
    if distances and per_cand:
        spec = _render_series(
            artifacts,
            out_path,
            style=style,
            distances=distances,
            per_cand=per_cand,
            per_cand_time=per_cand_time,
            sources=series_sources,
            ordered_cids=ordered,
        )
        if spec is not None:
            return spec

    rows, summary_sources = _collect_summary_payloads(md_dirs)
    if rows:
        return _render_summary_bars(
            artifacts,
            out_path,
            style=style,
            rows=rows,
            sources=summary_sources,
            ordered_cids=ordered,
        )

    _LOGGER.warning(
        "md_key_distances: no analysis.json had key_distances or "
        "catalytic_distance_mean, skipping"
    )
    return None
