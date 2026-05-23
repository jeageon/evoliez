"""Multi-panel MD RMSD time series (ligand + pocket)."""

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

STABILITY_THRESHOLD_A = 2.5  # angstrom dashed-line cutoff


def _load_analysis(md_dir: Path) -> Optional[Dict[str, Any]]:
    """Return parsed ``analysis.json`` for a candidate's MD directory."""
    candidate = md_dir / "analysis.json"
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text())
    except (OSError, ValueError):
        return None


def _series(
    data: Dict[str, Any],
    *,
    series_keys: Tuple[str, ...],
    summary_keys: Tuple[str, ...],
) -> Optional[List[float]]:
    for k in series_keys:
        v = data.get(k)
        if isinstance(v, list) and v:
            try:
                return [float(x) for x in v]
            except (TypeError, ValueError):
                continue
    # Summary-only payloads don't qualify - we need a real time series.
    for k in summary_keys:
        if k in data:
            return None
    return None


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


def _render_summary_bars(
    artifacts: ReportArtifacts,
    summary: List[Tuple[str, float, float, float, float]],
    sources: List[Path],
    out_path: Path,
    style: str,
) -> Optional[FigureSpec]:
    """Fallback layout when ``analysis.json`` lacks RMSD time-series.

    Renders a 2-panel horizontal-bar chart: each row is a candidate;
    bars show mean RMSD with a marker at the final-frame RMSD so the
    drift over the run is still visually obvious. Same color-by-
    evidence-class scheme as the time-series path.
    """
    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433

    # Sort by ligand-mean asc so the most-stable candidates land at the
    # top of the chart (easier to scan for paper figure).
    summary_sorted = sorted(summary, key=lambda r: r[1])
    # Cap to top-30 so the figure stays legible.
    if len(summary_sorted) > 30:
        summary_sorted = summary_sorted[:30]

    cand_ids = [r[0] for r in summary_sorted]
    lig_means = [r[1] for r in summary_sorted]
    lig_finals = [r[2] for r in summary_sorted]
    poc_means = [r[3] for r in summary_sorted]
    poc_finals = [r[4] for r in summary_sorted]
    colors = [evidence_color(_evidence_class_for(artifacts, c)) for c in cand_ids]

    n = len(cand_ids)
    height = max(3.0, 0.25 * n + 1.5)
    fig, (ax_lig, ax_poc) = plt.subplots(1, 2, figsize=(10, height), sharey=True)
    y_pos = list(range(n))

    ax_lig.barh(y_pos, lig_means, color=colors, alpha=0.7, label="mean")
    ax_lig.scatter(lig_finals, y_pos, color="black", marker="|", s=80,
                   linewidths=1.5, zorder=5, label="final")
    ax_lig.axvline(STABILITY_THRESHOLD_A, color="#D55E00", linestyle="--",
                   linewidth=1.0, alpha=0.7, label=f"{STABILITY_THRESHOLD_A} Å cutoff")
    ax_lig.set_xlabel("ligand RMSD (Å)")
    ax_lig.set_yticks(y_pos)
    ax_lig.set_yticklabels(cand_ids, fontsize="x-small")
    ax_lig.invert_yaxis()
    ax_lig.set_title("ligand stability (mean ─ final ▮)")
    ax_lig.grid(True, axis="x", alpha=0.3)
    ax_lig.legend(loc="lower right", fontsize="x-small")

    ax_poc.barh(y_pos, poc_means, color=colors, alpha=0.7)
    ax_poc.scatter(poc_finals, y_pos, color="black", marker="|", s=80,
                   linewidths=1.5, zorder=5)
    ax_poc.axvline(STABILITY_THRESHOLD_A, color="#D55E00", linestyle="--",
                   linewidth=1.0, alpha=0.7)
    ax_poc.set_xlabel("pocket RMSD (Å)")
    ax_poc.set_title("pocket stability")
    ax_poc.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"MD RMSD summary ({n} candidate{'s' if n != 1 else ''}) — "
        "time-series unavailable, falling back to mean+final"
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return FigureSpec(
        figure_id="08_md_rmsd_timeseries",
        section="md",
        title="MD RMSD summary",
        description=(
            f"Mean (bar) and final-frame (marker) RMSD for {n} MD-validated "
            "candidates. Time-series unavailable in analysis.json; this "
            "summary view is the fallback. Coloured by evidence class; the "
            f"{STABILITY_THRESHOLD_A} Å dashed line marks the stability cutoff."
        ),
        path=out_path,
        source_files=sources,
        renderer="matplotlib",
        params={
            "n_candidates": n,
            "threshold_A": STABILITY_THRESHOLD_A,
            "style": style,
            "mode": "summary_bars",
        },
    )


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Two-panel ligand + pocket RMSD time series across MD runs."""
    md_dirs = getattr(artifacts, "md_dirs", None) or {}
    if not md_dirs:
        _LOGGER.warning("md_rmsd: no MD directories, skipping")
        return None

    series_per_cand: List[
        Tuple[str, List[float], List[float], Optional[List[float]]]
    ] = []
    summary_per_cand: List[Tuple[str, float, float, float, float]] = []
    sources: List[Path] = []
    for cand_id, md_dir in md_dirs.items():
        data = _load_analysis(Path(md_dir))
        if data is None:
            continue
        lig = _series(
            data,
            series_keys=("ligand_rmsd_series", "ligand_rmsd", "lig_rmsd_series"),
            summary_keys=("ligand_rmsd_mean", "ligand_rmsd_max"),
        )
        pocket = _series(
            data,
            series_keys=("pocket_rmsd_series", "pocket_rmsd", "binding_site_rmsd_series"),
            summary_keys=("pocket_rmsd_mean", "pocket_rmsd_max"),
        )
        if lig and pocket:
            n = min(len(lig), len(pocket))
            if n >= 2:
                lig = lig[:n]
                pocket = pocket[:n]
                t = _time_axis(data, n)
                series_per_cand.append((cand_id, lig, pocket, t))
                sources.append(Path(md_dir) / "analysis.json")
                continue
        # Summary-stats fallback: production md/analysis.json only carries
        # mean/final per-RMSD (see md/analysis.to_json). Plot as a
        # horizontal bar showing mean ± (final - mean)/2 instead of
        # honest-skipping the whole section.
        lig_mean = data.get("ligand_rmsd_mean")
        lig_final = data.get("ligand_rmsd_final", lig_mean)
        poc_mean = data.get("pocket_rmsd_mean")
        if lig_mean is None or poc_mean is None:
            continue
        try:
            summary_per_cand.append((
                cand_id,
                float(lig_mean),
                float(lig_final or lig_mean),
                float(poc_mean),
                float(data.get("pocket_rmsd_final", poc_mean) or poc_mean),
            ))
            sources.append(Path(md_dir) / "analysis.json")
        except (TypeError, ValueError):
            continue

    if not series_per_cand and not summary_per_cand:
        _LOGGER.warning(
            "md_rmsd: no MD analysis.json had usable RMSD fields, skipping"
        )
        return None

    # Branch: time-series path renders the original 2-panel figure; the
    # summary-only fallback renders a different layout (grouped bars).
    if not series_per_cand:
        return _render_summary_bars(
            artifacts, summary_per_cand, sources, out_path, style,
        )

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433

    fig, (ax_lig, ax_poc) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True
    )

    for cand_id, lig, pocket, t in series_per_cand:
        ec = _evidence_class_for(artifacts, cand_id)
        color = evidence_color(ec)
        ax_lig.plot(t, lig, color=color, linewidth=1.5, label=cand_id, alpha=0.85)
        ax_poc.plot(t, pocket, color=color, linewidth=1.5, label=cand_id, alpha=0.85)

    for ax, ylabel in ((ax_lig, "ligand RMSD (Å)"), (ax_poc, "pocket RMSD (Å)")):
        ax.axhline(
            STABILITY_THRESHOLD_A,
            color="#D55E00",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    ax_poc.set_xlabel("time (ps)")
    ax_lig.set_title("MD stability (ligand vs pocket)")

    if len(series_per_cand) <= 8:
        ax_lig.legend(loc="upper right", fontsize="small", ncol=2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="08_md_rmsd_timeseries",
        section="md",
        title="MD RMSD time series",
        description=(
            "Ligand and pocket RMSD trajectories per MD-validated candidate, "
            "with the 2.5 Å stability cutoff."
        ),
        path=out_path,
        source_files=sources,
        renderer="matplotlib",
        params={
            "n_candidates": len(series_per_cand),
            "threshold_A": STABILITY_THRESHOLD_A,
            "style": style,
        },
    )
