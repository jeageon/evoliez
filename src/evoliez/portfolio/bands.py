"""V7 statistical evidence bands (ROADMAP_V7 Phase V7-2 / §4.2).

ROADMAP_V7 retires top-N ranking in favour of *calibrated evidence bands*: an axis score
is only ``strong`` / ``significant`` once it beats a null model at a controlled FDR, and it
must also clear a magnitude (effect-size) gate so a statistically-tight-but-tiny effect is
never oversold. This module is the band-assignment pass over an ``EvidenceLedgerV7`` bundle.

For every axis that has at least one non-missing candidate it:

  1. calibrates each candidate's score against a null — an externally supplied ``AxisNull``
     (any object exposing ``.values`` and ``.source``) when one is passed, otherwise the
     self-population of that axis's non-missing scores — with a one-sided empirical p-value
     oriented by ``HIGHER_IS_WORSE_AXES`` (the uncertainty axis's beneficial tail is the low
     end);
  2. controls the axis family's false-discovery rate with Benjamini-Hochberg q-values
     (``portfolio.stats``);
  3. attaches a robust population effect size, sign-oriented so a beneficial *low*
     uncertainty score reads as a positive effect;
  4. writes ``band`` from the ``(q, |effect|)`` pair.

Invariants (ROADMAP_V7 §2 / §4.2):

  * A DEFERRED / missing axis is NEVER recalibrated — an honest gap stays a gap, it does not
    silently become an ``unresolved`` zero, and a candidate is never sunk by one missing
    axis.
  * ``subset_level`` q-values (computed against a decision-changing subset rather than the
    full population) are flagged on every recalibrated axis and noted on the bundle, so a
    report can label subset-level q distinctly from full-population q (§4.2 + Gate 5).
  * Deterministic: the calibration is a pure, closed-form function of the scores + null +
    thresholds (``seed`` is accepted for API stability / future resampled nulls only).

The per-candidate overall band is delegated to the ledger's own combinator
(``recompute_overall_band``): strongest axis band, promoted to ``consensus`` once enough
axes clear the consensus q-band.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from evoliez.portfolio.ledger import (
    ALL_BANDS, BAND_DEFERRED, BAND_SIGNIFICANT, BAND_STRONG, BAND_UNRESOLVED,
    HIGHER_IS_WORSE_AXES, SEVEN_AXES, EvidenceLedgerV7, LedgerBundle,
)
from evoliez.portfolio.stats import (
    benjamini_hochberg, empirical_p_values, population_effect_size,
    robust_normal_p_values,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; the runtime path is duck-typed.
    from typing import Protocol

    class AxisNullLike(Protocol):
        values: np.ndarray
        source: str


_SUBSET_NOTE = "bands computed at subset level (subset-level q, not full-population)"


@dataclass(frozen=True)
class BandThresholds:
    """FDR + effect-size cut-offs for the band ladder (ROADMAP_V7 §4.2).

    ``strong_q`` / ``significant_q`` are the BH q ceilings for the strong / significant
    bands; ``consensus_q`` is the (looser) q at which an axis counts toward cross-axis
    consensus; ``consensus_min_axes`` axes at that q promote the overall band to
    ``consensus``. ``min_effect_size`` is the per-axis robust-z magnitude an axis must clear
    to be banded at all — a tiny effect never earns a band no matter how small its q."""

    strong_q: float = 0.03
    significant_q: float = 0.05
    consensus_q: float = 0.10
    consensus_min_axes: int = 2
    min_effect_size: float = 0.5


def _null_for(axis: str, nulls: Optional[Dict[str, "AxisNullLike"]],
              scores: List[float]) -> Tuple[np.ndarray, str]:
    """Resolve the null distribution + its provenance label for one axis. An external
    ``AxisNull`` (``.values`` / ``.source``) wins; otherwise the self-population of the
    axis's non-missing scores is the null (labelled ``population_self``)."""
    ext = None if nulls is None else nulls.get(axis)
    if ext is not None:
        values = np.asarray(getattr(ext, "values"), dtype=float).ravel()
        source = str(getattr(ext, "source", "") or "external_null")
        return values, source
    return np.asarray(scores, dtype=float).ravel(), "population_self"


def _band_for(q: float, effect: float, thresholds: BandThresholds) -> str:
    """Band from a ``(q_value, effect_size)`` pair. Both the q ceiling AND the effect-size
    gate must pass; a NaN q (uncalibrated) or a sub-threshold effect -> ``unresolved``."""
    if not np.isfinite(q):
        return BAND_UNRESOLVED
    effect_ok = abs(effect) >= thresholds.min_effect_size
    if q <= thresholds.strong_q and effect_ok:
        return BAND_STRONG
    if q <= thresholds.significant_q and effect_ok:
        return BAND_SIGNIFICANT
    return BAND_UNRESOLVED


def _is_recalibratable(led: EvidenceLedgerV7, axis: str) -> bool:
    """An axis is calibrated only if it carries real evidence — a missing / deferred axis
    is an honest gap and is left untouched (ROADMAP_V7 §2)."""
    ax = led.axes[axis]
    return not (ax.missing or ax.band == BAND_DEFERRED)


def assign_bands(bundle: LedgerBundle, *, thresholds: BandThresholds = BandThresholds(),
                 nulls: Optional[Dict[str, "AxisNullLike"]] = None,
                 subset_level: bool = False, seed: int = 0) -> LedgerBundle:
    """Calibrate every axis of ``bundle`` against its null and write evidence bands in place.

    Returns the SAME (mutated) bundle so callers can chain. Deterministic. ``seed`` is
    accepted for API stability and future resampled nulls; the current calibration is
    closed-form and does not consume it. See the module docstring for the full contract.
    """
    del seed  # closed-form calibration; reserved for future resampled nulls.

    for axis in SEVEN_AXES:
        live = [led for led in bundle.ledgers if _is_recalibratable(led, axis)]
        if not live:
            continue
        scores = [led.axes[axis].score for led in live]
        null, source = _null_for(axis, nulls, scores)
        null = null[np.isfinite(null)]
        if null.size == 0:
            continue

        bundle.axis_null_sizes[axis] = int(null.size)
        # an axis scored on only a SUBSET of the universe (e.g. reaction-geometry over the MD
        # subset) yields subset-level q-values, not full-population ones (ROADMAP_V7 §4.2 / Gate
        # 5). Flag it whenever the caller forces it OR the axis covers fewer than the full set.
        axis_subset = subset_level or (len(live) < len(bundle.ledgers))
        higher_is_worse = axis in HIGHER_IS_WORSE_AXES
        # Robust-Gaussian empirical-null p (analytic tail): the count-based empirical p has a
        # 1/(n+1) floor that makes BH over a same-size family unable to reach significance, so
        # a genuine outlier could never be banded (see stats.robust_normal_p_values). The
        # count-based p is retained per axis (detail["empirical_p_count"]) for transparency.
        pvals = robust_normal_p_values(scores, null, higher_is_worse=higher_is_worse)
        pvals_count = empirical_p_values(scores, null, higher_is_worse=higher_is_worse)
        qvals = benjamini_hochberg(pvals)

        for led, p, pc, q in zip(live, pvals, pvals_count, qvals):
            ax = led.axes[axis]
            effect = population_effect_size(ax.score, null)
            if higher_is_worse:
                # flip so a beneficial (low) uncertainty score reads as a positive effect
                effect = -effect
            ax.empirical_p = float(p)
            ax.q_value = None if not np.isfinite(q) else float(q)
            ax.effect_size = float(effect)
            # STICKY: never downgrade an axis already flagged subset-level by enrichment (an
            # axis computed on the MD subset stays subset-level even if this bundle happens to
            # cover it fully — its q is not a full-population q). Fable review / Gate 5.
            ax.subset_level = ax.subset_level or axis_subset
            ax.null_source = f"{source}+robust_normal"
            ax.detail["empirical_p_count"] = round(float(pc), 6)
            ax.band = _band_for(q, effect, thresholds)

    # per-candidate: consensus set (non-missing axes at q <= consensus_q) + overall band
    for led in bundle.ledgers:
        consensus: List[str] = []
        for axis in SEVEN_AXES:
            if not _is_recalibratable(led, axis):
                continue
            q = led.axes[axis].q_value
            if q is not None and q <= thresholds.consensus_q:
                consensus.append(axis)
        led.consensus_axes = consensus
        led.recompute_overall_band(consensus_min_axes=thresholds.consensus_min_axes)

    if subset_level and _SUBSET_NOTE not in bundle.notes:
        bundle.notes.append(_SUBSET_NOTE)

    return bundle


def band_summary(bundle: LedgerBundle) -> Dict[str, Dict[str, int]]:
    """Band histogram for a report: ``axis -> {band: count}`` for each of the seven axes,
    plus an ``"overall"`` entry counting each candidate's overall band. Every known band is
    present (0-filled) so the shape is stable across runs."""
    summary: Dict[str, Dict[str, int]] = {}
    for axis in SEVEN_AXES:
        counts = {band: 0 for band in ALL_BANDS}
        for led in bundle.ledgers:
            band = led.axes[axis].band
            counts[band] = counts.get(band, 0) + 1
        summary[axis] = counts
    overall = {band: 0 for band in ALL_BANDS}
    for led in bundle.ledgers:
        overall[led.overall_band] = overall.get(led.overall_band, 0) + 1
    summary["overall"] = overall
    return summary
