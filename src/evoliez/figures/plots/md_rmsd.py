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
        if not lig:
            continue
        pocket = _series(
            data,
            series_keys=("pocket_rmsd_series", "pocket_rmsd", "binding_site_rmsd_series"),
            summary_keys=("pocket_rmsd_mean", "pocket_rmsd_max"),
        )
        if not pocket:
            continue
        # Trim to common length so the two panels share an x-axis.
        n = min(len(lig), len(pocket))
        if n < 2:
            continue
        lig = lig[:n]
        pocket = pocket[:n]
        t = _time_axis(data, n)
        series_per_cand.append((cand_id, lig, pocket, t))
        sources.append(Path(md_dir) / "analysis.json")

    if not series_per_cand:
        _LOGGER.warning(
            "md_rmsd: no MD analysis.json had time-series arrays, skipping"
        )
        return None

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
