"""V7 mechanism-generic controls injector (ROADMAP_V7 Axis 7 / §8).

Every experimental panel needs its own null model: without controls a portfolio has no
way to tell a real evidence band from a scalar artifact, a weak-geometry design from a
productive one, or a low-confidence false negative from a true miss (ROADMAP_V7 §8). The
v4 ``experimental/controls.py`` hardcoded FDH residue strings (``S340G``, ``Q382R`` …),
which does not survive a target swap and silently invents residues that may not exist for
another enzyme. V7 instead DERIVES the control lanes from the candidate set + mechanism:

  * ``wt_parental``    — the parental/assay baseline (always present).
  * ``scalar_top``     — candidates that would top a naive scalar rank but whose evidence
                         bands do NOT support them (a scalar false-positive probe).
  * ``geometry_weak``  — candidates whose reaction-geometry axis is present but weak, or —
                         when geometry is still deferred — the lowest mechanism-consistency
                         candidates (a negative reaction-geometry probe).
  * ``low_signal``     — the most-uncertain candidates (the model's own false-negative
                         probes / highest uncertainty-axis score).
  * ``random_matched`` — a deterministic seeded draw from the unselected pool (the
                         structural/evolutionary null-matched baseline).

Nothing here invents a residue: every derived control is one of the *given* ledgers /
records, so it works for any mechanism (``mechanism=None`` is fine). Selection is
deterministic under ``seed``. This module never makes an activity claim — controls exist
to bound and calibrate claims, not to assert them.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator

from evoliez.portfolio.ledger import (
    AXIS_MECHANISM, AXIS_PORTFOLIO, AXIS_REACTION_GEOMETRY, AXIS_UNCERTAINTY,
    BAND_CONSENSUS, BAND_CONTROL, BAND_SIGNIFICANT, BAND_STRONG, CHEAP_AXES,
    HIGHER_IS_WORSE_AXES, EvidenceLedgerV7,
)

# --- control vocabulary (shared V7 contract) -----------------------------------------
CTRL_WT = "wt_parental"
CTRL_SCALAR_TOP = "scalar_top"
CTRL_GEOMETRY_WEAK = "geometry_weak"
CTRL_LOW_SIGNAL = "low_signal"
CTRL_POSITIVE = "positive"
CTRL_NEGATIVE = "negative"
CTRL_RANDOM_MATCHED = "random_matched"

CONTROL_TYPES = (
    CTRL_WT, CTRL_SCALAR_TOP, CTRL_GEOMETRY_WEAK, CTRL_LOW_SIGNAL, CTRL_POSITIVE,
    CTRL_NEGATIVE, CTRL_RANDOM_MATCHED,
)

_BAND_SUPPORTED = (BAND_STRONG, BAND_SIGNIFICANT, BAND_CONSENSUS)


class ControlSpec(BaseModel):
    """One derived control lane member (ROADMAP_V7 §8). ``control_type`` names the null it
    provides; ``reason`` is human-readable and stays claim-safe (never asserts activity)."""

    model_config = {"extra": "forbid"}

    variant_id: str
    mutation: str
    control_type: str
    reason: str
    provenance: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ControlSpec":
        if self.control_type not in CONTROL_TYPES:
            raise ValueError(
                f"control_type must be one of {CONTROL_TYPES}, got {self.control_type!r}")
        return self


# --- naive-scalar + band helpers -----------------------------------------------------
def _naive_scalar(led: EvidenceLedgerV7) -> float:
    """The scalar a naive ranker would compute: mean of the cheap-axis scores, with
    higher-is-worse axes (uncertainty) negated so the aggregate is coherently
    higher-is-better. Missing axes are skipped; no cheap evidence at all -> -inf."""
    vals: List[float] = []
    for ax in CHEAP_AXES:
        ev = led.axes[ax]
        if ev.missing:
            continue
        s = -ev.score if ax in HIGHER_IS_WORSE_AXES else ev.score
        vals.append(s)
    return sum(vals) / len(vals) if vals else float("-inf")


def _band_supported(led: EvidenceLedgerV7) -> bool:
    """True when the ledger's evidence bands already justify it (so it is NOT a
    scalar-only false positive)."""
    return led.overall_band in _BAND_SUPPORTED or bool(led.significant_axes())


def _mutation_for(led: EvidenceLedgerV7, records_by_id: Dict[str, dict]) -> str:
    """Mutation label for a ledger, falling back to a matching record's ``mutation`` when
    the ledger carries none (never invents a residue — only reads what was given)."""
    if led.mutation:
        return led.mutation
    rec = records_by_id.get(led.variant_id)
    if rec:
        return str(rec.get("mutation", "") or "")
    return ""


def _mechanism_tag(mechanism) -> Optional[str]:
    """A generic provenance tag for the mechanism, if one was passed. Reads only a
    ``mechanism_spec_id`` attribute — never an enzyme-specific field."""
    if mechanism is None:
        return None
    mid = getattr(mechanism, "mechanism_spec_id", None)
    return f"mechanism:{mid}" if mid else "mechanism:present"


def derive_controls(
    ledgers: List[EvidenceLedgerV7],
    records: Optional[List[dict]] = None,
    *,
    mechanism=None,
    wt_id: str = "WT",
    wt_mutation: str = "WT",
    n_scalar_top: int = 2,
    n_geometry_weak: int = 2,
    n_low_signal: int = 2,
    n_random_matched: int = 2,
    seed: int = 0,
) -> List[ControlSpec]:
    """Derive the control panel generically from the candidate ledgers + mechanism.

    Always emits one ``wt_parental``; then, bounded by availability and de-duplicated by
    ``variant_id`` (first-assigned control type wins), the scalar-false-positive,
    negative-geometry, low-confidence, and random-matched probes. Selects only from the
    given ledgers/records — never invents a residue. Deterministic under ``seed``.
    """
    records_by_id: Dict[str, dict] = {}
    for rec in (records or []):
        vid = rec.get("variant_id")
        if vid is not None and vid not in records_by_id:
            records_by_id[str(vid)] = rec

    mech_tag = _mechanism_tag(mechanism)

    def _prov(*tags: str) -> List[str]:
        base = ["v7_controls.derive_controls", *tags]
        if mech_tag:
            base.append(mech_tag)
        return base

    out: List[ControlSpec] = []
    claimed: set = set()

    # 1) wt_parental — always present.
    out.append(ControlSpec(
        variant_id=wt_id, mutation=wt_mutation, control_type=CTRL_WT,
        reason="assay baseline / parental reference",
        provenance=_prov("baseline")))
    claimed.add(wt_id)

    # candidate pool = every non-WT ledger not already a control in the input.
    pool = [led for led in ledgers if led.variant_id != wt_id and not led.is_control]
    by_id = {led.variant_id: led for led in pool}

    def _take(candidate_ids, control_type: str, reason: str, prov_tag: str, n: int) -> None:
        if n <= 0:
            return
        picked = 0
        for vid in candidate_ids:
            if picked >= n:
                break
            if vid in claimed:
                continue
            led = by_id[vid]
            out.append(ControlSpec(
                variant_id=vid, mutation=_mutation_for(led, records_by_id),
                control_type=control_type, reason=reason,
                provenance=_prov(prov_tag)))
            claimed.add(vid)
            picked += 1

    # 2) scalar_top — top naive scalar but NOT band-supported (scalar false positive).
    scalar_order = sorted(
        (led.variant_id for led in pool if not _band_supported(led)),
        key=lambda v: (-_naive_scalar(by_id[v]), v))
    _take(scalar_order, CTRL_SCALAR_TOP,
          "tops a naive scalar rank but no axis reaches a significant band "
          "— scalar false-positive probe", "scalar_rank_top", n_scalar_top)

    # 3) geometry_weak — reaction-geometry present-but-weak, else lowest mechanism consistency.
    geom_present = any(not led.axes[AXIS_REACTION_GEOMETRY].missing for led in pool)
    if geom_present:
        geom_order = sorted(
            (led.variant_id for led in pool
             if not led.axes[AXIS_REACTION_GEOMETRY].missing
             and not led.axes[AXIS_REACTION_GEOMETRY].is_significant()),
            key=lambda v: (by_id[v].axes[AXIS_REACTION_GEOMETRY].score, v))
        _take(geom_order, CTRL_GEOMETRY_WEAK,
              "reaction-geometry axis present but weak/unresolved "
              "— negative reaction-geometry probe", "geometry_weak", n_geometry_weak)
    else:
        mech_order = sorted(
            (led.variant_id for led in pool),
            key=lambda v: (by_id[v].axes[AXIS_MECHANISM].score, v))
        _take(mech_order, CTRL_GEOMETRY_WEAK,
              "lowest mechanism-consistency (reaction-geometry deferred) "
              "— negative-geometry surrogate probe", "geometry_weak_surrogate",
              n_geometry_weak)

    # 4) low_signal — highest uncertainty-axis score (model most unsure).
    unc_order = sorted(
        (led.variant_id for led in pool),
        key=lambda v: (-by_id[v].axes[AXIS_UNCERTAINTY].score, v))
    _take(unc_order, CTRL_LOW_SIGNAL,
          "highest uncertainty-axis score (model most unsure) "
          "— low-confidence false-negative probe", "high_uncertainty", n_low_signal)

    # 5) random_matched — deterministic seeded draw from what is left.
    if n_random_matched > 0:
        remaining = sorted(vid for vid in by_id if vid not in claimed)
        if remaining:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(remaining))
            chosen = [remaining[int(i)] for i in perm[:n_random_matched]]
            _take(chosen, CTRL_RANDOM_MATCHED,
                  "seeded random draw from unselected candidates "
                  "— structural/evolutionary null-matched baseline", "random_matched_seeded",
                  n_random_matched)

    return out


def _stamp_portfolio_axis(led: EvidenceLedgerV7, control_type: str, reason: str) -> None:
    """Record control membership on the ledger's portfolio-calibration axis (Axis 7): the
    axis is no longer a deferred gap, it is populated by experimental design."""
    ev = led.axes[AXIS_PORTFOLIO]
    ev.missing = False
    ev.band = BAND_CONTROL
    ev.provenance = ["v7_controls.inject", f"control_role:{control_type}", reason]


def inject_controls(ledgers: List[EvidenceLedgerV7], controls: List[ControlSpec]) -> None:
    """Mark the ledgers that are controls in place, and add a synthetic WT ledger if the
    WT control has no ledger of its own. Mutates ``ledgers``; returns ``None``.

    For every ledger whose ``variant_id`` matches a control: sets ``is_control=True``,
    ``control_role`` to the control type, stamps the portfolio axis, and recomputes the
    overall band (a control's overall band becomes ``control``)."""
    ctrl_by_id: Dict[str, ControlSpec] = {}
    for c in controls:
        ctrl_by_id.setdefault(c.variant_id, c)

    have_ids = {led.variant_id for led in ledgers}

    for led in ledgers:
        c = ctrl_by_id.get(led.variant_id)
        if c is None:
            continue
        led.is_control = True
        led.control_role = c.control_type
        _stamp_portfolio_axis(led, c.control_type, c.reason)
        led.recompute_overall_band()

    # add a synthetic WT ledger if the WT control was not among the given ledgers.
    for c in controls:
        if c.control_type != CTRL_WT:
            continue
        if c.variant_id in have_ids:
            continue
        wt = EvidenceLedgerV7(
            variant_id=c.variant_id, mutation=c.mutation, generator="v7_controls.inject",
            is_control=True, control_role=CTRL_WT)
        _stamp_portfolio_axis(wt, CTRL_WT, c.reason)
        wt.recompute_overall_band()
        ledgers.append(wt)
        have_ids.add(c.variant_id)
