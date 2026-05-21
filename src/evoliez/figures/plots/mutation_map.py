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


def _coerce_int_list(v: Any) -> Optional[List[int]]:
    if not isinstance(v, list):
        return None
    try:
        return [int(x) for x in v]
    except (TypeError, ValueError):
        return None


def _read_graph_features_file(path: Optional[Path]) -> Dict[str, List[int]]:
    """Parse a ``graph_features.json`` payload into our four canonical keys."""
    if path is None or not Path(path).exists():
        return {}
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, List[int]] = {}
    for key, alt in (
        ("catalytic", ("catalytic_positions", "catalytic_residues")),
        ("binding_site", ("binding_site_positions", "binding_site")),
        ("designable", ("designable_positions", "designable")),
        ("fixed", ("fixed_positions", "fixed")),
    ):
        for k in (key, *alt):
            coerced = _coerce_int_list(data.get(k))
            if coerced is not None:
                out[key] = coerced
                break
    return out


def _read_graph_features(run_dir: Optional[Path]) -> Dict[str, List[int]]:
    """Legacy fallback: look for ``graph_features.json`` next to the run dir.

    Kept for backward-compat with the pre-discovery-polish layout where
    callers passed the raw run_dir. New code should prefer
    :func:`_read_graph_features_file` on an explicit artifact path.
    """
    if run_dir is None:
        return {}
    for candidate in (
        Path(run_dir) / "graph_features.json",
        Path(run_dir) / "features" / "graph_features.json",
        Path(run_dir) / "interaction_graphs" / "graph_features.json",
    ):
        out = _read_graph_features_file(candidate)
        if out:
            return out
    return {}


def _positions_from_state_meta(state_path: Optional[Path]) -> Dict[str, List[int]]:
    """Pull ``designable_positions`` / ``catalytic_positions`` from ``_state.json``."""
    if state_path is None or not Path(state_path).exists():
        return {}
    try:
        doc = json.loads(Path(state_path).read_text())
    except (OSError, ValueError):
        return {}
    meta = doc.get("meta") if isinstance(doc, dict) else None
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, List[int]] = {}
    for cat, key in (
        ("catalytic", "catalytic_positions"),
        ("binding_site", "binding_site_positions"),
        ("designable", "designable_positions"),
        ("fixed", "fixed_positions"),
    ):
        coerced = _coerce_int_list(meta.get(key))
        if coerced is not None:
            out[cat] = coerced
    return out


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

    # Sources, in order:
    #   1. artifacts.graph_features_json (the canonical interaction_graphs
    #      output path, populated by discovery).
    #   2. legacy graph_features.json in the run_dir root / features/.
    #   3. _state.json meta keys (catalytic/designable/binding/fixed _positions).
    # Whichever first yields a non-empty dict wins; we then *merge* the
    # state-meta positions in to fill any gaps (e.g., graph_features only
    # has designable but meta also has catalytic).
    positions: Dict[str, List[int]] = _read_graph_features_file(
        getattr(artifacts, "graph_features_json", None)
    )
    if not positions:
        positions = _read_graph_features(getattr(artifacts, "run_dir", None))

    meta_positions = _positions_from_state_meta(getattr(artifacts, "state_json", None))
    for cat, pts in meta_positions.items():
        positions.setdefault(cat, pts)

    # As long as we have a sequence length we can still render a useful
    # backbone strip - empty position lists just produce a sparse track.
    if not positions:
        _LOGGER.info(
            "mutation_map: no position annotations found; rendering empty track"
        )

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
