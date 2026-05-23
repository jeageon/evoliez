"""Per-residue conservation heatmap with designable + catalytic overlays."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)


def _load_conservation(path: Path) -> Optional[List[float]]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    # Shape 1: flat list.
    if isinstance(data, list):
        try:
            return [float(x) for x in data] or None
        except (TypeError, ValueError):
            return None
    if not isinstance(data, dict):
        return None
    # Shape 2 (legacy): {"conservation": [...], "scores": [...]}.
    flat = data.get("conservation") or data.get("scores")
    if isinstance(flat, list) and flat:
        try:
            return [float(x) for x in flat]
        except (TypeError, ValueError):
            return None
    # Shape 3 (production - s03_msa actually writes this):
    #   {"1": {"conservation": 0.59, "entropy": ..., "allowed_aa": [...]},
    #    "2": {"conservation": 0.45, ...}, ...}
    # Numerically-sorted keys -> per-position conservation list.
    try:
        pos_keys = sorted(data.keys(), key=lambda k: int(k))
    except (TypeError, ValueError):
        return None
    if not pos_keys:
        return None
    cons: List[float] = []
    for key in pos_keys:
        entry = data[key]
        if not isinstance(entry, dict):
            return None
        val = entry.get("conservation")
        if val is None:
            return None
        try:
            cons.append(float(val))
        except (TypeError, ValueError):
            return None
    return cons or None


def _load_designable_positions(run_dir: Optional[Path]) -> List[int]:
    """Best-effort read of designable positions from ``graph_features.json``."""
    if run_dir is None:
        return []
    candidates = [
        Path(run_dir) / "graph_features.json",
        Path(run_dir) / "features" / "graph_features.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        positions = data.get("designable") or data.get("designable_positions")
        if isinstance(positions, list):
            try:
                return [int(p) for p in positions]
            except (TypeError, ValueError):
                continue
    return []


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    catalytic: Optional[Iterable[int]] = None,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render a 1D conservation heatmap with annotation tracks."""
    cons_path = getattr(artifacts, "conservation_json", None)
    if cons_path is None:
        _LOGGER.warning("conservation: conservation_json missing, skipping")
        return None
    scores = _load_conservation(Path(cons_path))
    if scores is None:
        _LOGGER.warning("conservation: failed to parse %s", cons_path)
        return None

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433
    import numpy as np
    n_pos = len(scores)
    positions = np.arange(1, n_pos + 1)

    catalytic_list: List[int] = list(catalytic) if catalytic else []
    designable = _load_designable_positions(getattr(artifacts, "run_dir", None))

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(6.0, n_pos * 0.08), 2.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.25},
        sharex=True,
    )
    ax_heat, ax_track = axes

    heat = np.asarray(scores, dtype=float).reshape(1, -1)
    im = ax_heat.imshow(
        heat,
        aspect="auto",
        cmap="viridis",
        extent=(0.5, n_pos + 0.5, 0, 1),
        vmin=0.0,
        vmax=1.0,
    )
    ax_heat.set_yticks([])
    ax_heat.set_ylabel("conservation")
    fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.01)
    ax_heat.set_title("Per-residue conservation")

    for pos in catalytic_list:
        if 1 <= pos <= n_pos:
            ax_heat.axvline(pos, color="#D55E00", linewidth=1.2, alpha=0.9)

    ax_track.set_ylim(-0.5, 1.5)
    ax_track.set_yticks([0, 1])
    ax_track.set_yticklabels(["designable", "catalytic"])
    for pos in designable:
        if 1 <= pos <= n_pos:
            ax_track.scatter(pos, 0, color="#0072B2", s=10, marker="s")
    for pos in catalytic_list:
        if 1 <= pos <= n_pos:
            ax_track.scatter(pos, 1, color="#D55E00", s=14, marker="^")
    ax_track.set_xlabel("residue position")
    ax_track.set_xlim(0.5, n_pos + 0.5)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="02_conservation_heatmap",
        section="msa",
        title="Per-residue conservation",
        description=(
            "Position-wise conservation heatmap (viridis), catalytic and "
            "designable positions overlaid."
        ),
        path=out_path,
        source_files=[Path(cons_path)],
        renderer="matplotlib",
        params={
            "n_positions": n_pos,
            "n_catalytic": len(catalytic_list),
            "n_designable": len(designable),
            "style": style,
        },
    )
