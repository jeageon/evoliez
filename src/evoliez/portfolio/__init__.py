"""EvoLiEZ V7 — mechanism-ranked portfolio engine (ROADMAP_V7).

A GPU-first, multi-fidelity, seven-axis evidence-band portfolio engine: cheap evidence for
every variant, statistically calibrated evidence bands instead of top-N ranking, expensive
Amber/PMF/QM-MM tiers reserved for decision-changing subsets, and a claim-safe sub-50
experimental panel with controls, deconvolution, and uncertainty probes.

The keystone contract lives in ``ledger`` (the seven-axis ``EvidenceLedgerV7``) and ``stats``
(pure-numpy null-model calibration). Higher layers (cheap_axes, null_models, bands,
allocator, builder, report, calibration) build on that contract.
"""
from __future__ import annotations

from evoliez.portfolio.ledger import (  # noqa: F401
    ALL_BANDS, ALL_LANES, ALL_TIERS, AXIS_EVOLUTIONARY, AXIS_LIGAND, AXIS_MECHANISM,
    AXIS_PORTFOLIO, AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL, AXIS_UNCERTAINTY,
    BAND_CONSENSUS, BAND_CONTROL, BAND_DEFERRED, BAND_EXPLORATORY, BAND_SIGNIFICANT,
    BAND_STRONG, BAND_UNRESOLVED, CHEAP_AXES, CLAIM_L0_HYPOTHESIS, CLAIM_L1_SCREENING,
    CLAIM_WETLAB_REQUIRED, HIGHER_IS_WORSE_AXES, LANE_CONSENSUS, LANE_CONTROL,
    LANE_DECONVOLUTION, LANE_SINGLE_SITE, LANE_STRONG, LANE_UNCERTAINTY, LANE_WT,
    SEVEN_AXES, TIER_CHEAP, TIER_EXPENSIVE, TIER_FOCUSED, TIER_GPU_BROAD, AxisEvidenceV7,
    EvidenceLedgerV7, LedgerBundle, ceiling_to_claim_strength, strongest_band,
)

__all__ = [
    "EvidenceLedgerV7", "AxisEvidenceV7", "LedgerBundle", "SEVEN_AXES", "ALL_BANDS",
    "ALL_LANES", "ALL_TIERS", "CHEAP_AXES", "HIGHER_IS_WORSE_AXES",
]
