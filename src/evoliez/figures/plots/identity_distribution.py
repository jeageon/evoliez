"""Homolog %-identity distribution vs WT."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from evoliez.figures.plots import apply_style_and_get_dpi
from evoliez.figures.types import FigureSpec, ReportArtifacts

_LOGGER = logging.getLogger(__name__)


def _read_fasta_records(path: Path) -> List[Tuple[str, str]]:
    """Minimal FASTA parser (no biopython dependency)."""
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_chunks: List[str] = []
    try:
        text = Path(path).read_text()
    except OSError:
        return []
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_chunks)))
            header = line[1:].split()[0] if line[1:] else ""
            seq_chunks = []
        else:
            seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


def _pct_identity(a: str, b: str) -> Optional[float]:
    """Identity on aligned columns, ignoring positions where both are gaps."""
    if not a or not b:
        return None
    n = min(len(a), len(b))
    matches = 0
    counted = 0
    for i in range(n):
        ca, cb = a[i], b[i]
        if ca == "-" and cb == "-":
            continue
        counted += 1
        if ca == cb and ca != "-":
            matches += 1
    if counted == 0:
        return None
    return 100.0 * matches / counted


def _identities_from_alignment(
    fasta_path: Path,
) -> Tuple[Optional[str], List[float]]:
    records = _read_fasta_records(fasta_path)
    if len(records) < 2:
        return None, []
    wt_header, wt_seq = records[0]
    out: List[float] = []
    for _, seq in records[1:]:
        pid = _pct_identity(wt_seq, seq)
        if pid is not None:
            out.append(pid)
    return wt_header, out


def _identities_from_interaction_model(path: Path) -> List[float]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    homologs = data.get("homologs") if isinstance(data, dict) else None
    if not isinstance(homologs, list):
        return []
    out: List[float] = []
    for h in homologs:
        if not isinstance(h, dict):
            continue
        for key in ("identity", "percent_identity", "pid"):
            v = h.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv <= 1.0:
                fv *= 100.0
            out.append(fv)
            break
    return out


def _load_core_diverse_split(artifacts: ReportArtifacts) -> Optional[float]:
    for attr in ("provenance_json", "state_json"):
        path = getattr(artifacts, attr, None)
        if path is None:
            continue
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for key in (
            "core_diverse_split",
            "identity_split",
            "homolog_identity_split",
        ):
            v = data.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def render(
    artifacts: ReportArtifacts,
    out_path: Path,
    *,
    style: str = "presentation",
    **kwargs: Any,
) -> Optional[FigureSpec]:
    """Histogram (and KDE) of homolog %-identity vs WT."""
    identities: List[float] = []
    sources: List[Path] = []

    fasta = getattr(artifacts, "alignment_fasta", None)
    if fasta is not None and Path(fasta).exists():
        _, identities = _identities_from_alignment(Path(fasta))
        if identities:
            sources.append(Path(fasta))

    if not identities:
        im_path = getattr(artifacts, "interaction_model_json", None)
        if im_path is not None and Path(im_path).exists():
            identities = _identities_from_interaction_model(Path(im_path))
            if identities:
                sources.append(Path(im_path))

    if not identities:
        _LOGGER.warning("identity_distribution: no homolog identities found")
        return None

    dpi = apply_style_and_get_dpi(style)
    from matplotlib import pyplot as plt  # noqa: WPS433
    import numpy as np
    arr = np.asarray(identities, dtype=float)

    fig, ax = plt.subplots()
    bins = max(10, min(40, len(arr) // 5 + 5))
    ax.hist(arr, bins=bins, color="#56B4E9", edgecolor="black", alpha=0.85)

    # Simple Gaussian KDE without scipy: equal-bandwidth kernel sum.
    if arr.size >= 3 and arr.std(ddof=0) > 0:
        # Silverman's rule
        bw = 1.06 * arr.std(ddof=0) * arr.size ** (-1 / 5)
        xs = np.linspace(arr.min() - 5, arr.max() + 5, 200)
        # density per unit, then scale so KDE sits at histogram height
        diff = (xs[:, None] - arr[None, :]) / max(bw, 1e-6)
        kernel = np.exp(-0.5 * diff ** 2) / (np.sqrt(2 * np.pi) * max(bw, 1e-6))
        density = kernel.sum(axis=1) / arr.size
        bin_width = (arr.max() - arr.min()) / max(bins, 1) if arr.max() > arr.min() else 1.0
        ax.plot(xs, density * arr.size * bin_width, color="#D55E00", linewidth=2.0)

    split = _load_core_diverse_split(artifacts)
    if split is not None:
        if split <= 1.0:
            split = split * 100.0
        ax.axvline(
            split,
            color="#009E73",
            linestyle="--",
            linewidth=1.5,
            label=f"core/diverse split = {split:.0f}%",
        )
        ax.legend()

    ax.set_xlabel("% identity vs WT")
    ax.set_ylabel("number of homologs")
    ax.set_title(f"Homolog identity distribution (n={arr.size})")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return FigureSpec(
        figure_id="02_identity_distribution",
        section="msa",
        title="Homolog identity distribution",
        description="Histogram + KDE of homolog percent-identity to the WT.",
        path=out_path,
        source_files=sources,
        renderer="matplotlib",
        params={
            "n_homologs": int(arr.size),
            "mean_identity": float(arr.mean()),
            "min_identity": float(arr.min()),
            "max_identity": float(arr.max()),
            "core_diverse_split": split,
            "style": style,
        },
    )
