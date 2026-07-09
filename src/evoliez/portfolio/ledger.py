"""V7 seven-axis evidence ledger — the data contract (ROADMAP_V7 Phase V7-0).

V7 does not collapse a candidate into one scalar. It carries, per candidate, a
**seven-axis evidence ledger** where every axis owns its own score, effect size,
empirical p / q-value, statistical band, confidence, provenance, and a claim ceiling
(ROADMAP_V7 §3). The seven axes:

  1. structural_viability
  2. evolutionary_tolerance
  3. ligand_cofactor_metal_accommodation
  4. reaction_geometry_access          (distance AND angle observables kept separate)
  5. mechanism_hypothesis_consistency  (role tags / deconvolution / active-site design)
  6. uncertainty_learning_value        (score is "higher = more uncertain")
  7. portfolio_calibration_controls    (lane membership / control role — panel-level)

Design invariants (ROADMAP_V7 §2):

  * Missing evidence is DEFERRED / UNSUPPORTED, never a silent zero (``AxisEvidenceV7.
    missing`` + ``band == "deferred"``). A high-cost axis that has not run yet is deferred,
    so a candidate is never rejected because one expensive axis is absent.
  * Bands replace top-N. A score becomes selectable only after it is calibrated against a
    null model (``band`` is set by ``portfolio.stats``), never by raw rank.
  * Claims stay bounded. ``claim_ceiling`` is per-axis and pre-wet-lab it can only ever be
    ``L0_hypothesis`` or ``L1_screening``; ClaimGuard (``portfolio.claims``) reads the
    ledger and blocks any activity/kcat/validated-lead language.

Pure schema (pydantic v2, ``extra='forbid'``); imports in the light env — no numpy/rdkit
needed to construct or serialize a ledger.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from evoliez.mechanism.vocab import (
    CONFIDENCE_LABELS, HYPOTHESIS_GRADE, MODERATE_SCREENING, STRONG_SCREENING,
    UNCALIBRATED,
)

# --- the seven axes ------------------------------------------------------------------
AXIS_STRUCTURAL = "structural_viability"
AXIS_EVOLUTIONARY = "evolutionary_tolerance"
AXIS_LIGAND = "ligand_cofactor_metal_accommodation"
AXIS_REACTION_GEOMETRY = "reaction_geometry_access"
AXIS_MECHANISM = "mechanism_hypothesis_consistency"
AXIS_UNCERTAINTY = "uncertainty_learning_value"
AXIS_PORTFOLIO = "portfolio_calibration_controls"

SEVEN_AXES = (
    AXIS_STRUCTURAL, AXIS_EVOLUTIONARY, AXIS_LIGAND, AXIS_REACTION_GEOMETRY,
    AXIS_MECHANISM, AXIS_UNCERTAINTY, AXIS_PORTFOLIO,
)

# The uncertainty axis is scored "higher = worse" (more disagreement). All other axes are
# "higher = better". The stats layer needs this to orient the one-sided null test.
HIGHER_IS_WORSE_AXES = frozenset({AXIS_UNCERTAINTY})

# Axes that carry a cheap all-variant score (Tier-0). The portfolio + reaction-geometry
# axes are panel-level / expensive and are DEFERRED until the builder / GPU tiers fill them.
CHEAP_AXES = (AXIS_STRUCTURAL, AXIS_EVOLUTIONARY, AXIS_LIGAND, AXIS_MECHANISM,
              AXIS_UNCERTAINTY)

# --- statistical bands (ROADMAP_V7 §4.2) ---------------------------------------------
BAND_STRONG = "strong"            # q <= strong_q AND effect size passes
BAND_SIGNIFICANT = "significant"  # q <= significant_q AND effect size passes
BAND_CONSENSUS = "consensus"      # q <= consensus_q in >= consensus_min_axes axes
BAND_EXPLORATORY = "exploratory"  # not significant, but high learning value / required
BAND_CONTROL = "control"          # selected by experimental design, not significance
BAND_UNRESOLVED = "unresolved"    # scored, but does not reach any band
BAND_DEFERRED = "deferred"        # evidence not computed yet (high-cost axis pending)

ALL_BANDS = (BAND_STRONG, BAND_SIGNIFICANT, BAND_CONSENSUS, BAND_EXPLORATORY,
             BAND_CONTROL, BAND_UNRESOLVED, BAND_DEFERRED)

# strongest -> weakest, for aggregating an overall band across axes
_BAND_RANK = {BAND_STRONG: 0, BAND_SIGNIFICANT: 1, BAND_CONSENSUS: 2,
              BAND_EXPLORATORY: 3, BAND_CONTROL: 3, BAND_UNRESOLVED: 4,
              BAND_DEFERRED: 5}


def band_rank(band: str) -> int:
    """Position of a band on the strong->weak scale (unknown -> weakest)."""
    return _BAND_RANK.get(band, max(_BAND_RANK.values()))


def strongest_band(*bands: str) -> str:
    """The strongest band among the arguments (the overall-band combinator). No args or
    all-unknown -> ``unresolved``."""
    known = [b for b in bands if b in _BAND_RANK]
    if not known:
        return BAND_UNRESOLVED
    return min(known, key=band_rank)


# --- multi-fidelity tiers (ROADMAP_V7 §5) --------------------------------------------
TIER_CHEAP = "tier0_cheap"            # all-variant cheap ledger (CPU / light GPU)
TIER_GPU_BROAD = "tier1_gpu_broad"    # broad GPU validation (Boltz / short MD)
TIER_FOCUSED = "tier2_focused_md"     # focused Amber explicit MD / retention / short PMF
TIER_EXPENSIVE = "tier3_reaction_core"  # PMF / MBAR-RBFE / QM-MM-lite

ALL_TIERS = (TIER_CHEAP, TIER_GPU_BROAD, TIER_FOCUSED, TIER_EXPENSIVE)

# --- per-axis claim ceiling (ROADMAP_V7 §3 + §11) ------------------------------------
# What a single axis's evidence can support ON ITS OWN, before wet-lab. Nothing above
# screening is reachable pre-wet-lab; ``higher_only_after_wetlab`` is the explicit gate.
CLAIM_L0_HYPOTHESIS = "L0_hypothesis"
CLAIM_L1_SCREENING = "L1_screening"
CLAIM_WETLAB_REQUIRED = "higher_only_after_wetlab"
CLAIM_CEILINGS = (CLAIM_L0_HYPOTHESIS, CLAIM_L1_SCREENING, CLAIM_WETLAB_REQUIRED)

# map a V7 axis claim ceiling onto the shared vocab.CLAIM_LADDER so ClaimGuard reasons
# over ONE vocabulary (the ledger never invents a competing claim scale).
CLAIM_CEILING_TO_STRENGTH = {
    CLAIM_L0_HYPOTHESIS: HYPOTHESIS_GRADE,
    CLAIM_L1_SCREENING: MODERATE_SCREENING,
    CLAIM_WETLAB_REQUIRED: STRONG_SCREENING,
}


def ceiling_to_claim_strength(ceiling: str) -> str:
    """vocab.CLAIM_LADDER strength for a V7 axis claim ceiling (unknown -> uncalibrated)."""
    return CLAIM_CEILING_TO_STRENGTH.get(ceiling, UNCALIBRATED)


# --- portfolio lanes (ROADMAP_V7 §7 / §8) --------------------------------------------
LANE_STRONG = "strong_significant"
LANE_CONSENSUS = "cross_axis_consensus"
LANE_DECONVOLUTION = "deconvolution"
LANE_SINGLE_SITE = "single_site_probe"
LANE_CONTROL = "control"
LANE_UNCERTAINTY = "uncertainty_probe"
LANE_WT = "wt_parental"

ALL_LANES = (LANE_STRONG, LANE_CONSENSUS, LANE_DECONVOLUTION, LANE_SINGLE_SITE,
             LANE_CONTROL, LANE_UNCERTAINTY, LANE_WT)


class _Base(BaseModel):
    model_config = {"extra": "forbid"}


class AxisEvidenceV7(_Base):
    """One evidence axis for one candidate (ROADMAP_V7 §3 axis schema).

    ``score`` is the raw axis measurement (``[0,1]``, higher-is-better except the
    uncertainty axis). ``effect_size`` / ``empirical_p`` / ``q_value`` / ``band`` are filled
    by ``portfolio.stats`` after calibration against the axis null model; until then they
    are ``None`` / ``unresolved``. ``missing=True`` (band ``deferred``) means the evidence
    was NOT computed — an honest gap, never a zero that would silently sink the candidate.
    ``detail`` keeps sub-observables that must stay separate (e.g. reaction geometry keeps
    ``distance`` and ``angle`` apart — a short distance is never equated with a productive
    in-line NAC unless the angle also supports it, ROADMAP_V7 §3 Axis 4).
    """

    axis: str
    score: float = 0.0
    effect_size: Optional[float] = None
    empirical_p: Optional[float] = None
    q_value: Optional[float] = None
    band: str = BAND_UNRESOLVED
    confidence: str = "low"
    provenance: List[str] = Field(default_factory=list)
    claim_ceiling: str = CLAIM_L0_HYPOTHESIS
    tier: str = TIER_CHEAP
    missing: bool = False
    subset_level: bool = False          # True => q-value is subset-level, not full-population
    detail: Dict[str, float] = Field(default_factory=dict)
    null_source: str = ""               # which null model calibrated this axis

    @model_validator(mode="after")
    def _check(self) -> "AxisEvidenceV7":
        if self.axis not in SEVEN_AXES:
            raise ValueError(f"axis must be one of {SEVEN_AXES}, got {self.axis!r}")
        if self.band not in ALL_BANDS:
            raise ValueError(f"band must be one of {ALL_BANDS}, got {self.band!r}")
        if self.confidence not in CONFIDENCE_LABELS:
            raise ValueError(
                f"confidence must be one of {CONFIDENCE_LABELS}, got {self.confidence!r}")
        if self.claim_ceiling not in CLAIM_CEILINGS:
            raise ValueError(
                f"claim_ceiling must be one of {CLAIM_CEILINGS}, got {self.claim_ceiling!r}")
        if self.tier not in ALL_TIERS:
            raise ValueError(f"tier must be one of {ALL_TIERS}, got {self.tier!r}")
        return self

    def is_significant(self) -> bool:
        return self.band in (BAND_STRONG, BAND_SIGNIFICANT)


def _deferred_axis(axis: str) -> AxisEvidenceV7:
    """A high-cost axis that has not been computed yet — an explicit gap, not a zero."""
    return AxisEvidenceV7(axis=axis, score=0.0, band=BAND_DEFERRED, missing=True,
                          provenance=["deferred_pending_evidence"])


class EvidenceLedgerV7(_Base):
    """The full seven-axis ledger for one candidate. All seven axes are ALWAYS present;
    an unfilled axis is a ``_deferred_axis`` (missing=True), never absent — so a downstream
    consumer can rely on ``ledger.axes[AXIS_X]`` existing for every X in ``SEVEN_AXES``."""

    variant_id: str
    mutation: str = ""
    generator: str = ""
    axes: Dict[str, AxisEvidenceV7] = Field(default_factory=dict)
    lane: Optional[str] = None
    reason_to_test: str = ""
    overall_band: str = BAND_UNRESOLVED
    consensus_axes: List[str] = Field(default_factory=list)  # axes at q <= consensus_q
    is_control: bool = False
    control_role: Optional[str] = None
    deconvolution_of: Optional[str] = None   # parent multipoint variant, if this is a probe
    tier_reached: str = TIER_CHEAP           # highest fidelity tier this candidate reached
    tier_rationale: str = ""                 # why the allocator ran (or did not run) tiers

    @model_validator(mode="after")
    def _fill_axes(self) -> "EvidenceLedgerV7":
        for ax in SEVEN_AXES:
            got = self.axes.get(ax)
            if got is None:
                self.axes[ax] = _deferred_axis(ax)
            elif got.axis != ax:
                raise ValueError(f"axes[{ax!r}].axis == {got.axis!r} (key/axis mismatch)")
        extra = set(self.axes) - set(SEVEN_AXES)
        if extra:
            raise ValueError(f"unknown axis key(s) {sorted(extra)}")
        return self

    def axis(self, name: str) -> AxisEvidenceV7:
        return self.axes[name]

    def significant_axes(self) -> List[str]:
        """Axes whose band is strong/significant (the multi-axis support count)."""
        return [ax for ax in SEVEN_AXES if self.axes[ax].is_significant()]

    def recompute_overall_band(self, *, consensus_min_axes: int = 2) -> str:
        """Overall band = strongest axis band, promoted to consensus when >= N axes sit in
        the consensus q-band. Controls keep their control band. Sets + returns the field."""
        if self.is_control:
            self.overall_band = BAND_CONTROL
            return self.overall_band
        per_axis = strongest_band(*(self.axes[ax].band for ax in SEVEN_AXES))
        if len(self.consensus_axes) >= consensus_min_axes and band_rank(per_axis) > band_rank(
                BAND_CONSENSUS):
            per_axis = BAND_CONSENSUS
        self.overall_band = per_axis
        return self.overall_band


class LedgerBundle(_Base):
    """A run's full ledger set + the run-level metadata a report / allocator needs. This is
    the on-disk V7 artifact (``reports/provenance/v7_evidence_ledger.json``)."""

    run_id: str = ""
    target_id: str = ""
    mechanism_class: str = ""
    ledgers: List[EvidenceLedgerV7] = Field(default_factory=list)
    n_candidates: int = 0
    axis_null_sizes: Dict[str, int] = Field(default_factory=dict)   # per-axis null n
    notes: List[str] = Field(default_factory=list)

    def by_id(self) -> Dict[str, EvidenceLedgerV7]:
        return {led.variant_id: led for led in self.ledgers}
