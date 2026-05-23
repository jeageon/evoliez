"""Section 5 - Interaction fingerprint matrix heatmap.

Reads ``ml_datasets/fingerprint_matrix.csv`` (written by stage s06b -
:func:`evoliez.stages.s06b_interaction_model._export_fingerprint_matrix`).
Rows are individual Boltz poses (``<group_id>__p<idx>``) and columns are
the residue-by-atom-type interaction-distance fingerprint features
(``feature_0``, ``feature_1``, ...).  Rendered as a matplotlib ``imshow``
heatmap with the Wong-style viridis colourmap so it composes with the
rest of the report.

Renderer contract::

    render(artifacts, out_path, *, style="presentation", **kwargs)
        -> Optional[FigureSpec]

Returns ``None`` when the CSV is missing / empty / unparseable - the
builder converts that into a "section skipped" note rather than
crashing.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

# Group → display color for the vertical separators / group banner labels.
# Wong-style palette (colorblind-safe) reused from evoliez.figures.style.
_GROUP_COLORS = {
    "dist_hist": "#0072B2",  # blue
    "itype":     "#D55E00",  # vermilion
    "summary":   "#009E73",  # bluish-green
    "kshell":    "#CC79A7",  # reddish-purple
}
_GROUP_PRETTY = {
    "dist_hist": "distance histogram",
    "itype":     "interaction type",
    "summary":   "summary",
    "kshell":    "k-shell distances",
}

# When the matrix has more rows than this we sub-sample to the rows with
# the highest variance - a 5000-pose ensemble produces an unreadable
# heatmap, and the high-variance rows are the diagnostically useful ones.
_MAX_ROWS = 50


def _load_matrix(
    csv_path: Path,
) -> Optional[Tuple[List[str], "object"]]:
    """Read ``fingerprint_matrix.csv`` -> (pose_ids, numpy 2D float matrix).

    Returns ``None`` on any parse failure or when the matrix would be
    empty (no rows or zero feature columns).
    """
    try:
        import numpy as np  # noqa: WPS433 - deferred matplotlib world
    except ImportError:
        _LOGGER.warning("fingerprint_heatmap: numpy unavailable")
        return None

    try:
        with Path(csv_path).open("r", newline="") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                return None
            # Schema: pose_id, group_id, feature_0, feature_1, ...
            if len(header) < 3:
                return None
            pose_ids: List[str] = []
            data: List[List[float]] = []
            for row in reader:
                if len(row) < 3:
                    continue
                pose_ids.append(row[0])
                try:
                    data.append([float(v) for v in row[2:]])
                except ValueError:
                    # Skip a malformed row rather than abort the whole figure.
                    pose_ids.pop()
                    continue
    except OSError:
        return None

    if not data:
        return None
    try:
        mat = np.asarray(data, dtype=float)
    except ValueError:
        return None
    if mat.ndim != 2 or mat.size == 0:
        return None
    return pose_ids, mat


def _load_feature_labels(csv_path: Path, n_features: int) -> List[dict]:
    """Resolve per-feature semantic labels from the sibling
    ``fingerprint_matrix_meta.json`` (written by s06b). Returns ``[]``
    when the meta file is missing or its label count doesn't match the
    matrix - the renderer then falls back to numeric x-ticks.
    """
    meta_path = csv_path.parent / "fingerprint_matrix_meta.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return []
    labels = meta.get("feature_labels") or []
    if not isinstance(labels, list) or len(labels) != n_features:
        return []
    # Be defensive: ensure each entry has the expected keys.
    cleaned: List[dict] = []
    for entry in labels:
        if not isinstance(entry, dict):
            return []
        cleaned.append({
            "label": str(entry.get("label", "?")),
            "group": str(entry.get("group", "")),
            "index_in_group": int(entry.get("index_in_group", 0)),
        })
    return cleaned


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render the interaction-fingerprint matrix heatmap.

    Layout: rows = poses (truncated id labels), columns = fingerprint
    feature indices, colour = feature value with a viridis colour map.
    """
    csv_path = getattr(artifacts, "fingerprint_matrix_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        _LOGGER.info("fingerprint_heatmap: fingerprint_matrix.csv missing")
        return None

    loaded = _load_matrix(Path(csv_path))
    if loaded is None:
        _LOGGER.warning(
            "fingerprint_heatmap: failed to parse %s", csv_path
        )
        return None
    pose_ids, mat = loaded

    dpi = apply_style_and_get_dpi(style)
    import numpy as np  # noqa: WPS433
    from matplotlib import pyplot as plt  # noqa: WPS433
    from matplotlib.patches import Rectangle  # noqa: WPS433

    n_rows_full, n_features = mat.shape

    # Sub-sample rows by descending row variance when the matrix is huge.
    if n_rows_full > _MAX_ROWS:
        variances = np.var(mat, axis=1)
        top_idx = np.argsort(variances)[-_MAX_ROWS:][::-1]
        mat = mat[top_idx]
        pose_ids = [pose_ids[i] for i in top_idx]
    n_rows = mat.shape[0]

    # Sensible figure size: wider with more features, taller with more
    # rows, but capped so we never produce a multi-meg PNG.
    fig_w = min(12.0, max(6.0, 0.05 * n_features + 4.0))
    fig_h = min(10.0, max(2.5, 0.18 * n_rows + 1.5))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        mat,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_ylabel("pose")
    # Extra pad leaves room for the colored group banner that the
    # semantic-xticks branch adds above the heatmap.
    ax.set_title("Interaction fingerprint matrix", pad=18)

    # Truncate pose labels so they don't fight the layout.
    short_labels = [
        (pid[:10] + "..." if len(pid) > 13 else pid) for pid in pose_ids
    ]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(short_labels, fontsize=7)

    # ---- semantic x-axis (P1f) ---------------------------------------
    # When the sibling meta JSON has per-feature labels, use them - and
    # draw a vertical separator + colored group banner at each group
    # boundary so a reader can scan the four regions at a glance.
    feature_labels = _load_feature_labels(Path(csv_path), n_features)
    used_semantic_xticks = False
    if feature_labels:
        ax.set_xticks(np.arange(n_features))
        ax.set_xticklabels(
            [e["label"] for e in feature_labels],
            rotation=60, ha="right", fontsize=7,
        )
        # Find group boundaries and draw vertical separator lines.
        groups: List[str] = [e["group"] for e in feature_labels]
        boundaries: List[Tuple[int, int, str]] = []
        start = 0
        for i in range(1, n_features + 1):
            if i == n_features or groups[i] != groups[start]:
                boundaries.append((start, i, groups[start]))
                if i != n_features:
                    ax.axvline(i - 0.5, color="white", linewidth=1.6,
                               alpha=0.85)
                start = i
        # Colored group strip across the top of the axes.
        y_top = ax.get_ylim()[1]
        for (s, e, g) in boundaries:
            color = _GROUP_COLORS.get(g, "#888888")
            ax.add_patch(
                Rectangle(
                    (s - 0.5, y_top - 0.4), (e - s), 0.4,
                    facecolor=color, alpha=0.7, edgecolor="none",
                    clip_on=False, zorder=5,
                )
            )
            ax.text(
                (s + e - 1) / 2.0, y_top - 0.2,
                _GROUP_PRETTY.get(g, g),
                ha="center", va="center",
                color="white", fontsize=7, weight="bold",
                clip_on=False, zorder=6,
            )
        ax.set_xlabel("fingerprint feature")
        used_semantic_xticks = True
    else:
        # Legacy fallback: numeric ticks, thinned out so they stay readable.
        ax.set_xlabel("fingerprint feature index")
        if n_features > 30:
            step = max(1, n_features // 12)
            ax.set_xticks(np.arange(0, n_features, step))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="feature value")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="05_fingerprint_heatmap",
        section="fingerprint",
        title="Interaction fingerprint heatmap",
        description=(
            "Per-pose interaction-distance fingerprint (residue x ligand-atom "
            "bins). Rows are Boltz diffusion-sample poses; bright bands "
            "highlight the family-consensus contact pattern."
        ),
        path=out_path,
        source_files=[Path(csv_path)],
        renderer="matplotlib",
        params={
            "n_rows": int(n_rows),
            "n_rows_total": int(n_rows_full),
            "n_features": int(n_features),
            "sampled": bool(n_rows_full > _MAX_ROWS),
            "semantic_xticks": bool(used_semantic_xticks),
            "style": style,
        },
    )
