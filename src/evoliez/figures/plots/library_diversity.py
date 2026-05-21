"""2D embedding of the focused library, colored by evidence class."""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.style import evidence_color
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

# Eisenberg consensus hydrophobicity (single-letter -> kcal/mol).  Used as a
# scalar feature so PCA has something more than one-hot to chew on.
_HYDROPHOBICITY = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

# Single-letter -> volume (Å^3) from Zamyatnin 1972, used as a second
# physicochemical channel.
_VOLUME = {
    "A": 88.6,  "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1,  "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0,  "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}


def _parse_mutation(s: str) -> Optional[Tuple[str, int, str]]:
    """Parse a "A123K"-style token into (wt, pos, mut)."""
    s = s.strip()
    if len(s) < 3:
        return None
    wt = s[0]
    mut = s[-1]
    try:
        pos = int(s[1:-1])
    except ValueError:
        return None
    return wt, pos, mut


def _row_features(row: Dict[str, str]) -> Optional[List[float]]:
    """Build a small numeric feature vector per candidate row."""
    mutation = row.get("mutation") or row.get("mutations") or ""
    parsed = _parse_mutation(mutation.split(",")[0] if mutation else "")
    if parsed is None:
        # Fall back to a hash-derived deterministic feature so untracked
        # rows still get a stable position in the embedding.
        h = int(hashlib.md5(mutation.encode("utf-8")).hexdigest()[:8], 16)
        return [
            (h & 0xFF) / 255.0,
            ((h >> 8) & 0xFF) / 255.0,
            ((h >> 16) & 0xFF) / 255.0,
            0.0,
        ]
    _, pos, mut = parsed
    hyd = _HYDROPHOBICITY.get(mut.upper(), 0.0)
    vol = _VOLUME.get(mut.upper(), 130.0)
    return [float(pos), hyd, vol, float(ord(mut.upper()) - ord("A"))]


def _final_score(row: Dict[str, str]) -> float:
    for k in ("final_score", "score", "ml_score"):
        v = row.get(k)
        if v in (None, "", "NA"):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _project_2d(matrix: List[List[float]]) -> List[Tuple[float, float]]:
    """PCA via numpy SVD; deterministic ordering, no scikit-learn needed."""
    import numpy as np

    X = np.asarray(matrix, dtype=float)
    if X.size == 0:
        return []
    if X.shape[0] == 1:
        return [(0.0, 0.0)]
    X = X - X.mean(axis=0, keepdims=True)
    # Guard against zero-variance columns
    stds = X.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    X = X / stds
    if X.shape[1] < 2:
        col = X[:, 0]
        return [(float(v), 0.0) for v in col]
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        comps = vt[:2]
        proj = X @ comps.T
        return [(float(p[0]), float(p[1])) for p in proj]
    except np.linalg.LinAlgError:
        return [(float(row[0]), float(row[1])) for row in X]


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Scatter the focused library in 2D, color by evidence, size by score."""
    csv_path = getattr(artifacts, "focused_library_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        _LOGGER.warning("library_diversity: focused_library_csv missing")
        return None

    rows: List[Dict[str, str]] = []
    try:
        with Path(csv_path).open("r", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    except OSError:
        _LOGGER.warning("library_diversity: cannot read %s", csv_path)
        return None
    if not rows:
        _LOGGER.warning("library_diversity: %s empty", csv_path)
        return None

    features: List[List[float]] = []
    keep: List[Dict[str, str]] = []
    for r in rows:
        feat = _row_features(r)
        if feat is None:
            continue
        features.append(feat)
        keep.append(r)
    if not features:
        _LOGGER.warning("library_diversity: no usable features")
        return None

    coords = _project_2d(features)
    if not coords:
        return None

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433
    import numpy as np

    scores = np.asarray([_final_score(r) for r in keep], dtype=float)
    smin, smax = float(scores.min()), float(scores.max())
    if smax > smin:
        sizes = 60.0 + 240.0 * (scores - smin) / (smax - smin)
    else:
        sizes = np.full_like(scores, 90.0)

    colors = [evidence_color((r.get("evidence_class") or "Uncertain")) for r in keep]
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])

    fig, ax = plt.subplots()
    ax.scatter(
        xs,
        ys,
        c=colors,
        s=sizes,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.85,
    )

    # Label top-3 by score
    top_idx = np.argsort(scores)[::-1][:3]
    for i in top_idx:
        cid = keep[i].get("candidate_id") or keep[i].get("mutation") or ""
        ax.annotate(
            str(cid),
            (xs[i], ys[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize="small",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Focused library diversity (n={len(keep)})")
    ax.grid(True, alpha=0.3)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="09_library_diversity",
        section="final_library",
        title="Focused library diversity",
        description=(
            "2D PCA embedding of the focused library; color = evidence class, "
            "marker size = final score, top-3 labelled."
        ),
        path=out_path,
        source_files=[Path(csv_path)],
        renderer="matplotlib",
        params={
            "n_candidates": len(keep),
            "feature_dim": len(features[0]) if features else 0,
            "style": style,
        },
    )
