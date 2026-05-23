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
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

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
    ax.set_xlabel("fingerprint feature index")
    ax.set_ylabel("pose")
    ax.set_title("Interaction fingerprint matrix")

    # Truncate pose labels so they don't fight the layout.
    short_labels = [
        (pid[:10] + "..." if len(pid) > 13 else pid) for pid in pose_ids
    ]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(short_labels, fontsize=7)

    # Thin out x-ticks so they stay readable.
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
            "style": style,
        },
    )
