"""Matplotlib styling, palettes, and evidence-class colors.

The constants here are imported directly by the plot renderers and by the
HTML templates' inline CSS generators, so the exported names
(:data:`WONG_PALETTE`, :data:`EVIDENCE_COLORS`, :data:`JOURNAL_STYLES`,
:func:`apply_mpl_style`, :func:`evidence_color`, :func:`is_paper_style`)
must stay stable.

Matplotlib is imported lazily inside :func:`apply_mpl_style` so a bare
``import evoliez.figures`` works without it installed - useful on the
laptop dev box where the ``[figures]`` extra hasn't been pulled in.
"""

from __future__ import annotations

from typing import Any, Dict

# Wong palette (Nature Methods standard, colorblind-safe).
WONG_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Evidence class colors (kept consistent across plots, tables, and 3D views).
EVIDENCE_COLORS: Dict[str, str] = {
    "Strong":    "#009E73",  # green
    "Promising": "#56B4E9",  # blue
    "Uncertain": "#E69F00",  # orange
    "Reject":    "#999999",  # grey
}

# Matplotlib rcParams per output style.  Sizes follow common journal /
# presentation / poster targets so the same plotting code yields legible
# output across all three.
JOURNAL_STYLES: Dict[str, Dict[str, Any]] = {
    "paper": {
        "dpi": 600,
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.figsize": (3.5, 2.5),
    },
    "presentation": {
        "dpi": 200,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.figsize": (8, 5),
    },
    "poster": {
        "dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.figsize": (12, 8),
    },
}


def style_dpi(style: str = "presentation") -> int:
    """Return the dpi for the requested journal style. Use this for
    `plt.savefig(..., dpi=style_dpi(style))` since matplotlib's bare
    ``"dpi"`` is not a valid rcParam (the real keys are
    ``"figure.dpi"`` for screen and ``"savefig.dpi"`` for output).
    """
    if style not in JOURNAL_STYLES:
        raise KeyError(
            f"unknown style {style!r}; choose from {sorted(JOURNAL_STYLES)}"
        )
    return int(JOURNAL_STYLES[style]["dpi"])


def apply_mpl_style(style: str = "presentation") -> None:
    """Apply matplotlib rcParams for the requested journal style.

    Lazily imports matplotlib so that importing this module without the
    ``[figures]`` extra installed is harmless.  Raises ``KeyError`` for an
    unknown style name (the CLI validates before calling).

    Translates the synthetic ``"dpi"`` key in :data:`JOURNAL_STYLES` to
    the real matplotlib rcParams ``figure.dpi`` and ``savefig.dpi``;
    setting ``"dpi"`` directly on ``mpl.rcParams`` raises ``KeyError``
    on matplotlib >= 3.x because no such bare key exists.
    """
    if style not in JOURNAL_STYLES:
        raise KeyError(
            f"unknown style {style!r}; choose from {sorted(JOURNAL_STYLES)}"
        )
    import matplotlib as mpl  # noqa: WPS433 (deferred import is intentional)
    from cycler import cycler

    params = dict(JOURNAL_STYLES[style])
    dpi = params.pop("dpi", None)
    if dpi is not None:
        params["figure.dpi"] = dpi
        params["savefig.dpi"] = dpi
    mpl.rcParams.update(params)
    mpl.rcParams["axes.prop_cycle"] = cycler(color=WONG_PALETTE)


def evidence_color(evidence_class: str) -> str:
    """Return the hex color for an evidence class, or grey if unknown."""
    return EVIDENCE_COLORS.get(evidence_class, "#cccccc")


def is_paper_style(style: str) -> bool:
    """True for the print-paper preset (smaller fonts, higher DPI)."""
    return style == "paper"
