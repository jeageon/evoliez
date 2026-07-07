"""Benchmark Stage-1 registry + claim integration (ROADMAP_V3 D9 / V3-8).

Stage 1 = FDH + TEM-1 β-lactamase + glycosidase. Each benchmark carries a
``BenchmarkCard`` that pins what its label MEANS (antibiotic fitness != kcat) and which
claims are allowed/prohibited, so cross-benchmark claims can't drift. Metrics stay
known-active retention / known-inactive rejection / top-k enrichment / FN rate — never a
single AUC.

The REAL TEM-1 / glycosidase end-to-end runs still require benchmark-specific curation
(templates already exist; references, controls, and label adapters do not) — this module
is the card registry + the claim hookup, not the curated data. ``benchmark_claim_guard``
turns a card's ``prohibited_claims`` into extra linter phrases so e.g. "direct kcat
prediction" is enforced for a fitness-labelled benchmark even if wet-lab exists.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .cards import BenchmarkCard

# --- Stage-1 benchmark cards ---------------------------------------------------------
FDH = BenchmarkCard(
    target_id="fdh_nadp",
    mechanism_template="hydride_transfer",
    label_primary="NADPH_formation_rate",
    label_secondary="NADP_specificity_ratio",
    is_direct_kcat=True,                # a direct kinetic endpoint is available in principle
    quantitative=True,
    assay_context="A340_NADPH_absorbance",
    reference_tiers=["B", "C"],
    substrate_or_analog="formate",
    wt_available=True,
    allowed_claims=["known-active retention", "top-k enrichment against activity"],
    prohibited_claims=[],
)

TEM1 = BenchmarkCard(
    target_id="TEM1_beta_lactamase",
    mechanism_template="nucleophilic_acyl_substitution",
    label_primary="antibiotic_fitness",
    label_secondary="known_activity_or_resistance",
    is_direct_kcat=False,               # fitness is NOT a kinetic constant
    replicate_available=True,
    quantitative=True,
    assay_context="E_coli_growth_under_antibiotic",
    reference_tiers=["A", "B"],
    substrate_or_analog="beta_lactam_or_inhibitor",
    wt_available=True,
    known_active_available=True,
    known_inactive_available=True,
    allowed_claims=["known-active retention", "known-inactive rejection",
                    "top-k enrichment against fitness"],
    prohibited_claims=["direct kcat prediction"],
)

GLYCOSIDASE = BenchmarkCard(
    target_id="glycosidase",
    mechanism_template="glycosidic_bond_cleavage",
    label_primary="catalytic_activity_DMS",
    is_direct_kcat=False,
    quantitative=True,
    assay_context="microfluidic_DMS",
    reference_tiers=["B", "C"],
    substrate_or_analog="glycoside_substrate",
    wt_available=True,
    allowed_claims=["non-dehydrogenase mechanism generalization", "top-k enrichment"],
    prohibited_claims=["direct kcat prediction"],
)

# P450 BM3 is declared but kept as a ReactionState STRESS TEST, not a strong mechanistic
# claim (redox/intermediate state is hard) — included so the template is exercised.
P450_BM3 = BenchmarkCard(
    target_id="P450_BM3",
    mechanism_template="metal_cofactor_redox",
    label_primary="variant_activity_database",
    is_direct_kcat=False,
    reference_tiers=["C", "D"],
    substrate_or_analog="fatty_acid_or_analog",
    wt_available=True,
    allowed_claims=["reaction-state stress test only"],
    prohibited_claims=["direct kcat prediction", "validated catalytic state",
                       "mechanistic activity prediction"],
)

STAGE1 = {c.target_id: c for c in (FDH, TEM1, GLYCOSIDASE)}
ALL_BENCHMARKS = {c.target_id: c for c in (FDH, TEM1, GLYCOSIDASE, P450_BM3)}


def get_benchmark(target_id: str) -> Optional[BenchmarkCard]:
    return ALL_BENCHMARKS.get(target_id)


def benchmark_extra_forbidden(card: BenchmarkCard) -> List[str]:
    """Phrases to add to the ClaimGuard linter for this benchmark. A fitness/DMS label is
    never a kinetic constant, so 'kcat' claims are forbidden regardless of wet-lab."""
    extra = list(card.prohibited_claims)
    if not card.is_direct_kcat:
        extra += ["kcat prediction", "predicts kcat", "kinetic constant"]
    return extra


def benchmark_unlocks_kinetics(card: BenchmarkCard) -> bool:
    """Whether replicated wet-lab on THIS benchmark may unlock kinetic-parameter claims.
    Only a direct-kcat assay can — a fitness/resistance/DMS proxy never does."""
    return bool(card.is_direct_kcat)
