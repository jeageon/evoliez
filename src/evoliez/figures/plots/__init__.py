"""2D plot renderers for the HTML report.

Each sub-module exposes a single :func:`render` entry point with the
signature::

    def render(artifacts, out_path: Path, *, style: str = "presentation",
               **kwargs) -> Optional[FigureSpec]

Renderers defer ``matplotlib`` (and ``rdkit`` for ligand depiction) imports
until they're actually called, so importing this package on a barebones
environment is safe.  When the required input data is missing, a renderer
returns ``None`` instead of raising so the report builder can simply skip
that figure and log a warning.

The figures are referenced from the report manifest by their
``figure_id``; the section names line up with the HTML template layout.
"""

from __future__ import annotations

from typing import Any, Dict

# rcParam keys that aren't real matplotlib rcParams - the foundation style
# dict mixes "dpi" (a savefig kwarg, not an rcParam) with real font /
# figsize rcParams.  We strip those before passing the rest to matplotlib
# so a single bad key doesn't break style application across the whole
# plot library.  See evoliez.figures.style.JOURNAL_STYLES.
_NON_RCPARAM_KEYS = {"dpi"}


def apply_style_and_get_dpi(style: str = "presentation") -> int:
    """Apply matplotlib rcParams for ``style`` and return its DPI.

    Wraps :func:`evoliez.figures.style.apply_mpl_style` so the plot
    renderers have a single place to handle the "dpi is not a valid
    rcParam" quirk of the foundation style dict.  Falls back to manually
    applying the remaining keys if the foundation helper rejects ``dpi``.
    """
    from evoliez.figures.style import JOURNAL_STYLES, WONG_PALETTE, apply_mpl_style

    style_spec: Dict[str, Any] = JOURNAL_STYLES.get(style, {})
    dpi = int(style_spec.get("dpi", 200))

    try:
        apply_mpl_style(style)
    except Exception:  # noqa: BLE001 - any rcParam validation error
        # Strip non-rcParam keys and apply the rest by hand so plots
        # still get the journal-appropriate fonts / figsize.
        import matplotlib as mpl
        from cycler import cycler

        safe = {k: v for k, v in style_spec.items() if k not in _NON_RCPARAM_KEYS}
        # Promote the foundation's ``dpi`` to the real rcParam name.
        if "dpi" in style_spec and "figure.dpi" not in safe:
            safe["figure.dpi"] = dpi
        # Apply one key at a time so a single invalid rcParam doesn't
        # take down the rest of the style.
        for k, v in safe.items():
            try:
                mpl.rcParams[k] = v
            except Exception:  # noqa: BLE001
                continue
        try:
            mpl.rcParams["axes.prop_cycle"] = cycler(color=WONG_PALETTE)
        except Exception:  # noqa: BLE001
            pass
    return dpi


# Submodule imports are at the bottom so plot modules can do
# ``from evoliez.figures.plots import apply_style_and_get_dpi`` without
# tripping a circular import during package initialisation.
from evoliez.figures.plots import (  # noqa: E402
    benchmark_recovery,
    conservation,
    evidence_distribution,
    identity_distribution,
    library_diversity,
    ligand_2d,
    md_rmsd,
    mutation_map,
    score_waterfall,
)

__all__ = [
    "benchmark_recovery",
    "conservation",
    "evidence_distribution",
    "identity_distribution",
    "library_diversity",
    "ligand_2d",
    "md_rmsd",
    "mutation_map",
    "score_waterfall",
    "apply_style_and_get_dpi",
]
