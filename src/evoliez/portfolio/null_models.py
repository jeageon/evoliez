"""V7 per-axis null models (ROADMAP_V7 Phase V7-2, §4.1).

A score is only *selectable* after it beats a null (ROADMAP_V7 §4). V7 does not share one
global null across the seven axes: each axis has a different notion of "background", so
**each axis owns its own null distribution**. This module assembles those per-axis nulls
from a set of ledgers, so ``portfolio.stats`` can turn a raw axis score into an empirical
p-value / band against the *right* background.

Two null flavours are built here:

  * **population null** (always available) — the pooled empirical distribution of an axis's
    scores over every candidate that actually carries that axis. This is the honest default
    background: "how does this candidate's axis score sit within the whole design set?".
  * **control-derived null** (preferred for the reaction-geometry and ligand/cofactor/metal
    axes when controls are supplied) — the axis scores of the *role-breaking / geometry-weak*
    controls. Testing a candidate against variants engineered to be non-productive is a far
    more principled null than testing it against the whole population, because it directly
    asks "does this candidate beat what a broken active site looks like?". This keeps the
    reaction-geometry axis honest (ROADMAP_V7 §3 Axis 4) without ever equating a low distance
    with productive catalysis.

Design invariants (ROADMAP_V7 §2):

  * An axis with **no** non-missing score across the ledgers is simply omitted from the
    returned dict — it stays DEFERRED, never forced to a fabricated zero-null.
  * Mechanism-generic: nulls are assembled from axis scores and generic control *role* tags,
    never from any enzyme-specific residue list.
  * Deterministic: the null is the exact pooled empirical distribution, so the result is
    reproducible for a fixed ``seed`` (the seed only governs the safety-valve subsample of a
    pathologically large null).

Pure schema + numpy; imports cleanly in the light env (no scipy/rdkit/torch).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from evoliez.portfolio.ledger import (
    AXIS_LIGAND, AXIS_REACTION_GEOMETRY, SEVEN_AXES, EvidenceLedgerV7,
)

# --- null-source vocabulary (the human label an AxisNull carries) --------------------
NULL_POPULATION = "population"                 # pooled scores over all candidates
NULL_POSITION_MATCHED = "position_matched"     # (reserved) same-position substitutions
NULL_CHEMISTRY_MATCHED = "chemistry_matched"   # (reserved) chemistry-matched substitutions
NULL_SCRAMBLED_REACTIVE = "scrambled_reactive"  # (reserved) reactive-atom scramble
NULL_ROLE_SHELL = "role_shell"                 # (reserved) role-shell residues
NULL_CONTROL_DERIVED = "control_derived"       # scores of role-breaking / geometry-weak controls

ALL_NULL_SOURCES = (
    NULL_POPULATION, NULL_POSITION_MATCHED, NULL_CHEMISTRY_MATCHED,
    NULL_SCRAMBLED_REACTIVE, NULL_ROLE_SHELL, NULL_CONTROL_DERIVED,
)

# Axis for which a control-derived null MAY replace the population null: only the
# reaction-geometry access axis, and only from a large UNBIASED negative set (see below).
# The ligand/structural/etc. cheap axes always use the robust population null — an Efron-style
# empirical null estimated from the bulk (median/MAD), which resists the signal tail and is
# unbiased. (A null built from controls that were SELECTED to be weak is biased low, so almost
# every candidate would trivially beat it -> a false-positive explosion; that is not a valid
# null for one-sided FDR banding, so weak/geometry controls are never used as the null.)
_CONTROL_DERIVED_AXES = frozenset({AXIS_REACTION_GEOMETRY})

# Control roles admissible as an UNBIASED null: a random-matched sample (no score selection),
# an explicit known-negative, or a reactive-atom scramble / role-shell perturbation. NOT
# geometry_weak / low_signal / scalar_top — those are selected on their score and would bias
# the null. The control-derived null is used only when it also clears _MIN_CONTROL_NULL.
_NULL_CONTROL_ROLES = frozenset({
    "random_matched", "negative", "scrambled_reactive", "role_shell",
})

# A control-derived null needs enough samples for a stable robust fit; below this it is not a
# credible background and we fall back to the (large) population null.
_MIN_CONTROL_NULL = 30

# Safety valve: a null larger than this is deterministically subsampled (with ``seed``) so a
# huge design set does not blow up per-candidate empirical-p evaluation. Far above any real
# panel, so ordinary runs keep the full empirical distribution.
_MAX_NULL_SIZE = 200_000


class AxisNull(BaseModel):
    """The null distribution for a single evidence axis.

    ``source`` is the human label of *which* background this is (``population`` /
    ``control_derived`` / ...). ``values`` is the empirical null sample and ``n`` its size
    (auto-filled from ``values`` when left at 0)."""

    model_config = {"extra": "forbid"}

    axis: str
    source: str
    values: List[float] = Field(default_factory=list)
    n: int = 0

    @model_validator(mode="after")
    def _check(self) -> "AxisNull":
        if self.axis not in SEVEN_AXES:
            raise ValueError(f"axis must be one of {SEVEN_AXES}, got {self.axis!r}")
        if self.source not in ALL_NULL_SOURCES:
            raise ValueError(
                f"source must be one of {ALL_NULL_SOURCES}, got {self.source!r}")
        if self.n == 0 and self.values:
            self.n = len(self.values)
        return self

    def as_array(self) -> np.ndarray:
        """The null as a finite numpy array (for ``portfolio.stats`` consumption)."""
        a = np.asarray(self.values, dtype=float).ravel()
        return a[np.isfinite(a)]


def null_source_for(axis: str, has_controls: bool) -> str:
    """Which null source axis ``axis`` uses under the current policy (the *intended* source;
    ``build_axis_nulls`` falls back to the population null unless a large, unbiased control set
    exists). Only the reaction-geometry axis can adopt a control-derived null, and only from an
    unbiased negative set of at least ``_MIN_CONTROL_NULL``; every other axis — and all axes
    without such controls — uses the robust population null."""
    if has_controls and axis in _CONTROL_DERIVED_AXES:
        return NULL_CONTROL_DERIVED
    return NULL_POPULATION


def _axis_scores(ledgers: List[EvidenceLedgerV7], axis: str) -> np.ndarray:
    """Finite, non-missing scores for ``axis`` across ``ledgers`` (deferred axes dropped)."""
    vals: List[float] = []
    for led in ledgers:
        ev = led.axes[axis]                 # always present (ledger auto-fills deferred axes)
        if ev.missing:
            continue
        vals.append(float(ev.score))
    a = np.asarray(vals, dtype=float).ravel()
    return a[np.isfinite(a)]


def _null_controls(control_ledgers: List[EvidenceLedgerV7]) -> List[EvidenceLedgerV7]:
    """The subset of controls admissible as an UNBIASED null (random-matched / known-negative /
    scramble / role-shell). Score-selected controls (geometry_weak / low_signal / scalar_top)
    are excluded — using them as the null would bias it low and inflate significance."""
    return [c for c in control_ledgers
            if (c.control_role or "").strip().lower() in _NULL_CONTROL_ROLES]


def _make_null(axis: str, source: str, values: np.ndarray, seed: int) -> AxisNull:
    """Build an ``AxisNull``, applying the deterministic large-null safety-valve subsample."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size > _MAX_NULL_SIZE:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(v.size, size=_MAX_NULL_SIZE, replace=False))
        v = v[idx]
    vals = [float(x) for x in v]
    return AxisNull(axis=axis, source=source, values=vals, n=len(vals))


def build_axis_nulls(
    ledgers: List[EvidenceLedgerV7],
    *,
    mechanism: Optional[Any] = None,
    control_ledgers: Optional[List[EvidenceLedgerV7]] = None,
    seed: int = 0,
) -> Dict[str, AxisNull]:
    """Assemble a per-axis null model over ``ledgers`` (ROADMAP_V7 §4.1).

    For every axis in ``SEVEN_AXES`` that carries at least one non-missing score:

      * default = a **population** null (pooled scores of that axis over all ``ledgers``);
      * for the reaction-geometry and ligand axes, if ``control_ledgers`` are supplied and
        carry a score for that axis, a **control-derived** null is used instead (the axis
        scores of the role-breaking / geometry-weak controls), which is a stricter, more
        principled background than the population.

    The uncertainty axis (scored higher-is-worse) gets an ordinary population null of its
    disagreement scores; the higher-is-worse orientation is applied by the *caller* via
    ``stats.empirical_p_values(higher_is_worse=True)``, not here.

    Axes with no non-missing score are omitted (they stay DEFERRED). Deterministic given
    ``seed``; numpy only. ``mechanism`` is accepted for signature parity across the portfolio
    layer and reserved for mechanism-specific nulls (position-/chemistry-matched); the current
    implementation builds mechanism-agnostic empirical nulls."""
    del mechanism  # reserved; see docstring
    controls = list(control_ledgers) if control_ledgers else []
    null_controls = _null_controls(controls) if controls else []

    out: Dict[str, AxisNull] = {}
    for axis in SEVEN_AXES:
        pop = _axis_scores(ledgers, axis)
        if pop.size == 0:
            continue  # nothing observed on this axis -> deferred, omit

        source = NULL_POPULATION
        values = pop
        # A control-derived null replaces the population null ONLY for the reaction-geometry
        # axis and ONLY from an unbiased negative set large enough for a stable fit; otherwise
        # the robust population null (Efron-style) is the honest, unbiased background.
        if null_controls and axis in _CONTROL_DERIVED_AXES:
            ctrl = _axis_scores(null_controls, axis)
            if ctrl.size >= _MIN_CONTROL_NULL:
                source = NULL_CONTROL_DERIVED
                values = ctrl
        out[axis] = _make_null(axis, source, values, seed)
    return out
