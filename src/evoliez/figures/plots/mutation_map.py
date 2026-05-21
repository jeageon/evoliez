"""1D sequence track marking catalytic, binding-site, designable residues."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

_CATEGORY_COLORS = {
    "catalytic":    "#D55E00",  # red/vermillion
    "binding_site": "#009E73",  # green
    "designable":   "#0072B2",  # blue
    "fixed":        "#999999",  # grey
}

# Track Y offsets (top -> bottom)
_TRACK_ORDER = ("catalytic", "binding_site", "designable", "fixed")


def _read_fasta_length(path: Path) -> Optional[int]:
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    seq_chunks: List[str] = []
    started = False
    for line in text.splitlines():
        if line.startswith(">"):
            if started:
                break
            started = True
            continue
        if started:
            seq_chunks.append(line.strip())
    seq = "".join(seq_chunks)
    return len(seq) if seq else None


def _read_graph_features(run_dir: Optional[Path]) -> Dict[str, List[int]]:
    """Pull catalytic / binding-site / designable / fixed positions."""
    if run_dir is None:
        return {}
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
        if not isinstance(data, dict):
            continue
        out: Dict[str, List[int]] = {}
        for key, alt in (
            ("catalytic", ("catalytic_positions", "catalytic_residues")),
            ("binding_site", ("binding_site_positions", "binding_site")),
            ("designable", ("designable_positions", "designable")),
            ("fixed", ("fixed_positions", "fixed")),
        ):
            for k in (key, *alt):
                v = data.get(k)
                if isinstance(v, list):
                    try:
                        out[key] = [int(x) for x in v]
                    except (TypeError, ValueError):
                        continue
                    break
        if out:
            return out
    return {}


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Horizontal sequence track with per-category position markers."""
    fasta = getattr(artifacts, "target_fasta", None)
    if fasta is None or not Path(fasta).exists():
        _LOGGER.warning("mutation_map: target_fasta missing, skipping")
        return None
    n = _read_fasta_length(Path(fasta))
    if not n:
        _LOGGER.warning("mutation_map: failed to read sequence length")
        return None

    positions = _read_graph_features(getattr(artifacts, "run_dir", None))
    if not positions:
        _LOGGER.warning("mutation_map: no graph_features.json found, skipping")
        return None

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(max(7.0, n * 0.03), 2.8))
    # Backbone strip
    ax.axhspan(-0.4, 0.4, color="#eeeeee")
    ax.plot([1, n], [0, 0], color="#bbbbbb", linewidth=1.0)

    y_offsets: Dict[str, float] = {}
    for i, cat in enumerate(_TRACK_ORDER):
        y_offsets[cat] = -(i + 1) * 0.8

    drawn: List[str] = []
    for cat in _TRACK_ORDER:
        pts = positions.get(cat)
        if not pts:
            continue
        drawn.append(cat)
        y = y_offsets[cat]
        ax.scatter(
            pts,
            [y] * len(pts),
            color=_CATEGORY_COLORS[cat],
            s=28,
            edgecolors="black",
            linewidths=0.3,
            marker="o" if cat != "catalytic" else "^",
        )
        ax.text(
            -0.01 * n - 1,
            y,
            cat.replace("_", " "),
            ha="right",
            va="center",
            fontsize="small",
        )

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(min(y_offsets.values()) - 0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("residue position")
    ax.set_title(f"Mutation design space (sequence length = {n})")
    ax.grid(True, axis="x", alpha=0.3)

    legend_handles = [
        Patch(facecolor=_CATEGORY_COLORS[c], label=c.replace("_", " "))
        for c in drawn
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize="small")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="06_mutation_design_space",
        section="mutation",
        title="Mutation design space",
        description=(
            "1D sequence track marking catalytic, binding-site, designable, "
            "and fixed positions."
        ),
        path=out_path,
        source_files=[Path(fasta)],
        renderer="matplotlib",
        params={
            "sequence_length": n,
            "counts": {k: len(v) for k, v in positions.items()},
            "style": style,
        },
    )
