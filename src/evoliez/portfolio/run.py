"""V7 portfolio orchestrator (ROADMAP_V7 — ties the phases together).

Reads a COMPLETED (or partially complete) pipeline run's on-disk provenance, builds the
all-candidate cheap seven-axis ledger, calibrates statistical bands against per-axis null
models, derives mechanism-generic controls, plans the multi-fidelity compute allocation
(subset-only expensive tiers), assembles the <50-variant mechanism-ranked portfolio, and
writes the claim-safe report + machine-readable ledger/plan artifacts.

This is deliberately a POST-RUN step (a CLI command, ``evoliez portfolio``), not a pipeline
stage: V7's cheap ledger + bands operate on the full candidate universe that s07-s09 already
produced, and the allocator's expensive tiers are then run on a subset. It reuses the run's
existing expensive evidence (s10 MD / NAC / PMF) where present and marks the rest deferred.

Pure-python + numpy; imports in the light env (no rdkit/openmm/torch).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from evoliez.portfolio import bands as _bands
from evoliez.portfolio import builder as _builder
from evoliez.portfolio import cheap_axes as _cheap
from evoliez.portfolio import claims as _claims
from evoliez.portfolio import controls as _controls
from evoliez.portfolio import null_models as _nulls
from evoliez.portfolio.allocator import TierBudget, allocate_tiers
from evoliez.portfolio.ledger import TIER_CHEAP, LedgerBundle

# provenance files in order of INCREASING evidence fidelity. The first file that carries a
# candidate seeds its identity + cheap features; each later (higher-fidelity) file OVERWRITES
# the evidence fields it provides, so the s10 MD frame wins over the s09 docking frame for a
# candidate that reached MD (e.g. the MD-frame catalytic_distance_mean, not the docking one).
_MERGE_FILES = (
    "generated_candidates.json",       # full s07 universe (widest) — identity + cheap feats
    "reranked_candidates.json",        # + s08 cheap features
    "validated_candidates.json",       # + s09 non-MD (real docking / stability)
    "md_candidates.json",              # + s10 MD/NAC — authoritative for the reaction-geometry frame
)
# keys that identify a candidate: kept from the FIRST (widest) file, never overwritten.
_IDENTITY_KEYS = frozenset(
    {"candidate_id", "variant_id", "id", "mutation_string", "mutation", "generator"})


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _rows(obj) -> List[dict]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        # some artifacts are {id: record}
        return [v for v in obj.values() if isinstance(v, dict)]
    return []


def _cid(rec: dict) -> Optional[str]:
    return rec.get("candidate_id") or rec.get("variant_id") or rec.get("id")


def load_run_records(prov_dir: Path, *, md_dir: Optional[Path] = None) -> List[dict]:
    """Merge the run's provenance rows into ONE record per candidate. Identity fields come from
    the widest (first) file; every other (evidence) field is taken from the HIGHEST-fidelity
    source that provides a non-null value, so a candidate that reached MD carries its MD-frame
    geometry (nac / MD-frame catalytic_distance_mean) rather than the earlier docking frame.

    The per-candidate ``md/<candidate_id>/analysis.json`` files (the authoritative s10 MD/NAC
    results — ``md_candidates.json`` is only a run-level summary) are merged last when
    ``md_dir`` is given, and ``nac_delta_vs_wt`` + a ``stability_score`` proxy are derived so
    the enrichment can lift them onto the reaction-geometry / structural axes."""
    prov_dir = Path(prov_dir)
    merged: Dict[str, dict] = {}
    order: List[str] = []
    for name in _MERGE_FILES:
        for rec in _rows(_load_json(prov_dir / name)):
            cid = _cid(rec)
            if not cid:
                continue
            if cid not in merged:
                merged[cid] = dict(rec)
                order.append(cid)
                continue
            _merge_evidence(merged[cid], rec)

    if md_dir is not None:
        _merge_md_analyses(merged, order, Path(md_dir), prov_dir)
    return [merged[c] for c in order]


def _merge_evidence(dst: dict, rec: dict) -> None:
    """Merge ``rec`` into ``dst``: identity keys keep the first (widest) value; every other
    (evidence) field is overwritten by a non-null higher-fidelity value."""
    for k, v in rec.items():
        if k in _IDENTITY_KEYS:
            dst.setdefault(k, v)
        elif v is not None:
            dst[k] = v


def _merge_md_analyses(merged: Dict[str, dict], order: List[str], md_dir: Path,
                       prov_dir: Path) -> None:
    """Fold each ``md/<candidate_id>/analysis.json`` (authoritative per-candidate MD results)
    into its record, and derive the two fields the EvidenceCard mapper needs but s10 leaves
    implicit: ``nac_delta_vs_wt`` (occupancy minus the WT reference) and a ``stability_score``
    proxy from ``md_lite_score`` (the MD stability composite)."""
    if not md_dir.is_dir():
        return
    summary = _load_json(prov_dir / "md_candidates.json") or {}
    wt_nac = summary.get("wt_nac_occupancy") if isinstance(summary, dict) else None
    for sub in sorted(md_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        rec = _load_json(sub / "analysis.json")
        if not isinstance(rec, dict):
            continue
        cid = sub.name
        # derive the fields the EvidenceCard mapper reads (s10 records NAC nested in `nac`)
        occ = rec.get("nac_occupancy")
        if occ is None and isinstance(rec.get("nac"), dict):
            occ = rec["nac"].get("nac_occupancy")
        if isinstance(occ, (int, float)):
            rec.setdefault("nac_occupancy", occ)
            if isinstance(wt_nac, (int, float)):
                rec.setdefault("nac_delta_vs_wt", round(float(occ) - float(wt_nac), 4))
        mls = rec.get("md_lite_score")
        if isinstance(mls, (int, float)):
            rec.setdefault("stability_score", float(mls))   # MD stability -> structural prior
        if cid not in merged:
            merged[cid] = {"candidate_id": cid}
            order.append(cid)
        _merge_evidence(merged[cid], rec)


class PortfolioParams:
    """Knobs for one portfolio build (mirrors config.PortfolioConfig; kept as a plain object
    so run.py has no hard dependency on the Config schema)."""

    def __init__(self, *, panel_size: int = 48, strong_q: float = 0.03,
                 significant_q: float = 0.05, consensus_q: float = 0.10,
                 min_effect_size: float = 0.5, consensus_min_axes: int = 2,
                 tier1_max: int = 80, tier2_max: int = 40, tier3_max: int = 12,
                 gpu_pool: tuple = (), subset_level: bool = False, seed: int = 0):
        self.panel_size = panel_size
        self.strong_q = strong_q
        self.significant_q = significant_q
        self.consensus_q = consensus_q
        self.min_effect_size = min_effect_size
        self.consensus_min_axes = consensus_min_axes
        self.tier1_max = tier1_max
        self.tier2_max = tier2_max
        self.tier3_max = tier3_max
        self.gpu_pool = tuple(gpu_pool)
        self.subset_level = subset_level
        self.seed = seed


def build_portfolio_for_run(run_dir, *, params: Optional[PortfolioParams] = None,
                            mechanism=None, target_id: str = "", mechanism_class: str = "",
                            strict: Optional[bool] = None) -> dict:
    """Full V7 pipeline over a finished run directory. Returns a dict of written artifact
    paths + in-memory objects (bundle, portfolio, tier_plan) for callers/tests."""
    run_dir = Path(run_dir)
    params = params or PortfolioParams()
    prov = run_dir / "reports" / "provenance"
    out_dir = run_dir / "reports"

    records = load_run_records(prov, md_dir=run_dir / "md")
    if not records:
        raise FileNotFoundError(
            f"no candidate provenance under {prov} (expected one of {_MERGE_FILES})")

    # V7-1: all-candidate cheap ledger, then V7-4/5 enrichment of the SUBSET that reached the
    # expensive tiers (real s09 docking/stability + s10 MD/NAC) — this fills the deferred
    # reaction-geometry axis for the MD subset (honestly subset-level).
    ledgers = _cheap.build_cheap_ledgers(records, mechanism=mechanism, seed=params.seed)
    n_enriched = _cheap.enrich_expensive_axes(records, ledgers, mechanism=mechanism)
    bundle = LedgerBundle(run_id=run_dir.name, target_id=target_id,
                          mechanism_class=mechanism_class, ledgers=ledgers,
                          n_candidates=len(ledgers))
    if n_enriched:
        bundle.notes.append(
            f"{n_enriched} candidate(s) enriched with expensive-tier evidence "
            "(reaction-geometry / real structural-ligand); those axes are subset-level")

    # mechanism-generic controls (Axis 7) BEFORE bands so a control-derived null is available
    ctrl = _controls.derive_controls(ledgers, records, mechanism=mechanism, seed=params.seed)
    _controls.inject_controls(bundle.ledgers, ctrl)
    control_ids = {c.variant_id for c in ctrl}
    control_ledgers = [led for led in bundle.ledgers if led.variant_id in control_ids]

    # V7-2: per-axis null models + statistical bands
    nulls = _nulls.build_axis_nulls(bundle.ledgers, mechanism=mechanism,
                                    control_ledgers=control_ledgers, seed=params.seed)
    thresholds = _bands.BandThresholds(
        strong_q=params.strong_q, significant_q=params.significant_q,
        consensus_q=params.consensus_q, consensus_min_axes=params.consensus_min_axes,
        min_effect_size=params.min_effect_size)
    _bands.assign_bands(bundle, thresholds=thresholds, nulls=nulls,
                        subset_level=params.subset_level, seed=params.seed)

    # V7-3: multi-fidelity compute allocation (subset-only expensive tiers)
    budget = TierBudget(tier1_gpu_broad_max=params.tier1_max,
                        tier2_focused_md_max=params.tier2_max,
                        tier3_reaction_core_max=params.tier3_max,
                        gpu_pool=params.gpu_pool, consensus_q=params.consensus_q)
    tier_plan = allocate_tiers(bundle, budget=budget)
    _stamp_tiers(bundle, tier_plan)

    # V7-6: mechanism-ranked portfolio + claim-safe report
    portfolio = _builder.build_portfolio(bundle, panel_size=params.panel_size,
                                         controls=ctrl, seed=params.seed)
    prov_card = _claims.ledger_claim_provenance(bundle)
    written = _report(portfolio, bundle, tier_plan, out_dir, prov_card, strict, run_dir.name)
    return {
        "bundle": bundle, "portfolio": portfolio, "tier_plan": tier_plan,
        "n_candidates": len(ledgers), "artifacts": written,
    }


def _stamp_tiers(bundle: LedgerBundle, tier_plan) -> None:
    """Record the allocator's RECOMMENDATION (what tier to run next) without clobbering the
    ACTUAL tier a candidate already reached — ``enrich_expensive_axes`` set ``tier_reached``
    from real evidence, so only the still-cheap candidates get the recommendation stamped."""
    by_id = bundle.by_id()
    for a in tier_plan.assignments:
        led = by_id.get(a.variant_id)
        if led is not None and led.tier_reached == TIER_CHEAP:
            led.tier_rationale = f"recommended: {a.tier} — {a.reason}"


def _report(portfolio, bundle, tier_plan, out_dir: Path, prov_card, strict, run_id: str) -> dict:
    from evoliez.portfolio import report as _rep
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = _rep.write_portfolio_report(portfolio, bundle, out_dir,
                                          claim_provenance=prov_card, strict=strict,
                                          run_id=run_id)
    # the tier plan (allocator justification, Gate 3) is a first-class artifact
    plan_path = out_dir / "provenance" / "v7_tier_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(tier_plan.model_dump(), indent=2))
    written = dict(written)
    written["tier_plan"] = str(plan_path)
    return written
