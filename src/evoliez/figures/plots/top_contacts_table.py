"""Section 5 - Top-K most frequent ligand-residue contact table.

Reads ``ml_datasets/edge_level.csv`` (written by stage s11; rows are
``(residue_index, ligand_atom_id, mean_distance, contact_frequency,
confidence_weighted_score, weak_contact)``).  Sorts by
``contact_frequency`` (descending), keeps the top-K, and renders as a
matplotlib ``Table`` figure (rendered to PNG so it embeds cleanly in
the report alongside the heatmap and the 3D contact-line viewer).

The matplotlib-table-as-PNG approach matches the rest of section 5's
figure cards - we don't want to introduce an HTML table here because
the section already has a separate datatable component for raw rows;
this is meant to be a single eye-catchable summary figure.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)

# Candidate column names in priority order - the schema isn't 100%
# stable across pipeline versions, so we accept a few aliases.
_FREQ_COLS = ("contact_frequency", "frequency", "freq")
_DIST_COLS = ("mean_distance", "distance", "mean_dist")
_RES_COLS = ("residue_index", "residue", "res_idx")
_ATOM_COLS = ("ligand_atom_id", "ligand_atom", "atom_id")


def _pick_col(row_keys, candidates):
    for k in candidates:
        if k in row_keys:
            return k
    return None


def _load_edges(csv_path: Path) -> List[Dict[str, Any]]:
    """Load and normalise rows from ``edge_level.csv``.

    Returns a list of dicts each containing the keys ``residue``,
    ``atom``, ``frequency`` (float, may be NaN), ``mean_distance``
    (float, may be NaN).  Bad rows are skipped silently - the goal is a
    best-effort summary figure.
    """
    rows: List[Dict[str, Any]] = []
    try:
        with Path(csv_path).open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            res_key = _pick_col(fieldnames, _RES_COLS)
            atom_key = _pick_col(fieldnames, _ATOM_COLS)
            freq_key = _pick_col(fieldnames, _FREQ_COLS)
            dist_key = _pick_col(fieldnames, _DIST_COLS)
            if res_key is None or atom_key is None or freq_key is None:
                return []
            for raw in reader:
                try:
                    freq = float(raw.get(freq_key) or "nan")
                except ValueError:
                    continue
                try:
                    mean_d = (
                        float(raw.get(dist_key))
                        if dist_key and raw.get(dist_key) not in (None, "")
                        else float("nan")
                    )
                except ValueError:
                    mean_d = float("nan")
                rows.append(
                    {
                        "residue": str(raw.get(res_key) or "").strip(),
                        "atom": str(raw.get(atom_key) or "").strip(),
                        "frequency": freq,
                        "mean_distance": mean_d,
                    }
                )
    except OSError:
        return []
    return rows


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    top_k: int = 15,
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Render a top-K contact summary table as a PNG figure."""
    ml = getattr(artifacts, "ml_datasets", {}) or {}
    edge_csv = ml.get("edge_level")
    if edge_csv is None or not Path(edge_csv).exists():
        _LOGGER.info("top_contacts_table: edge_level.csv missing")
        return None

    rows = _load_edges(Path(edge_csv))
    if not rows:
        _LOGGER.info("top_contacts_table: no usable rows in %s", edge_csv)
        return None

    # Sort by frequency (descending); NaNs sink to the bottom.
    def _sort_key(r):
        f = r["frequency"]
        # NaN compares unpredictably - replace with -inf for sorting.
        return f if f == f else float("-inf")

    rows.sort(key=_sort_key, reverse=True)
    keep = rows[: max(1, int(top_k))]

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433

    n_rows = len(keep)
    fig_h = min(8.0, max(2.0, 0.32 * (n_rows + 1) + 0.5))
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    ax.set_axis_off()
    ax.set_title(f"Top {n_rows} ligand-residue contacts", pad=12)

    columns = ["residue", "ligand atom", "frequency", "mean distance (Å)"]
    cell_text: List[List[str]] = []
    for r in keep:
        freq = r["frequency"]
        dist = r["mean_distance"]
        freq_s = f"{freq:.3f}" if freq == freq else "-"
        dist_s = f"{dist:.2f}" if dist == dist else "-"
        cell_text.append([r["residue"], r["atom"], freq_s, dist_s])

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)
    # Make the header row stand out without depending on style sheets.
    for col_idx in range(len(columns)):
        header_cell = table[(0, col_idx)]
        header_cell.set_facecolor("#0072B2")
        header_cell.set_text_props(color="white", weight="bold")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="05_top_contacts",
        section="fingerprint",
        title=f"Top {n_rows} ligand-residue contacts",
        description=(
            "Most frequent ligand-residue contacts across the Boltz "
            "pose ensemble, ranked by contact frequency."
        ),
        path=out_path,
        source_files=[Path(edge_csv)],
        renderer="matplotlib",
        params={
            "top_k": int(top_k),
            "n_rows": int(n_rows),
            "n_rows_total": int(len(rows)),
            "style": style,
        },
    )
