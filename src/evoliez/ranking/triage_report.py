"""v3 triage report — the integration that ties the deltas together (ROADMAP_V3 §7).

Given a finished run's provenance directory, for every candidate it:
  * builds an ``EvidenceCard`` (D4) from the record,
  * assembles a ``triage_recommendation`` (why selected / what's risky / next validation),
  * caps the whole report's claims with a ``ClaimGuard`` verdict (D3),
and renders text that is GUARANTEED claim-clean (``assert_report_clean`` runs on the
output, so a leak is a hard failure, not a warning).

Reads ``md_candidates.json`` (+ validated/generated for context) via the same join as
``ml.learnability``. Pure-stdlib + the v3 modules; runnable as a CLI on a real run:

    python -m evoliez.ranking.triage_report <run>/reports/provenance
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

from evoliez.mechanism.cards import ReferenceConfidenceCard, SimulationSetupCard
from evoliez.ranking import claim_guard as cg
from evoliez.ranking.evidence_card import build_evidence_card


def _load(prov_dir: str, name: str) -> Dict[str, dict]:
    path = os.path.join(prov_dir, name)
    if not os.path.exists(path):
        return {}
    data = json.load(open(path))
    recs = data if isinstance(data, list) else (data.get("candidates") or data.get("records") or [])
    return {r["candidate_id"]: r for r in recs if isinstance(r, dict) and r.get("candidate_id")}


def _recommend(card) -> dict:
    """Derive a triage recommendation from the evidence card axes (claim-safe language)."""
    sv = card.structural_viability
    rg = card.reaction_geometry_accommodation
    unc = card.uncertainty
    reasons, risks = [], []
    if sv.score >= 0.6:
        reasons.append("high structural viability")
    if card.ligand_cofactor_competence.score >= 0.6:
        reasons.append("ligand/cofactor anchor contacts retained")
    if rg.score >= 0.5 and rg.confidence not in ("low",):
        reasons.append("reaction-geometry accommodation supported")
    if rg.confidence in ("low", "low_to_medium"):
        risks.append("reaction-geometry confidence low")
    if unc.score >= 0.6:
        risks.append("high alternative-pose uncertainty")
    if not reasons:
        reasons.append("screening-level evidence only")

    # recommendation tier (NO activity language)
    if sv.score >= 0.6 and rg.score >= 0.4 and unc.score < 0.6:
        rec = "test_in_round1"
    elif sv.score >= 0.5:
        rec = "secondary_candidate"
    else:
        rec = "deprioritized"
    return {
        "variant_id": card.variant_id, "recommendation": rec,
        "primary_reason": reasons, "primary_risk": risks or ["none flagged"],
        "next_validation": ["wet-lab activity assay", "specificity counter-screen"],
    }


def build_triage(
    prov_dir: str, *, reference_card: Optional[ReferenceConfidenceCard] = None,
    sim_card: Optional[SimulationSetupCard] = None,
    claim_provenance=None, wt_geometry_sparse: bool = False,
    ensemble_geometry_variance: float = 0.0,
) -> dict:
    md = _load(prov_dir, "md_candidates.json")
    validated = _load(prov_dir, "validated_candidates.json")
    records = md or validated
    verdict = cg.evaluate(claim_provenance)

    cards, recs = [], []
    for cid, rec in records.items():
        merged = {**validated.get(cid, {}), **rec}
        card = build_evidence_card(
            merged, reference_card=reference_card, sim_card=sim_card,
            wt_geometry_sparse=wt_geometry_sparse,
            ensemble_geometry_variance=ensemble_geometry_variance)
        cards.append(card)
        recs.append(_recommend(card))

    return {
        "n_candidates": len(records),
        "claim_ceiling": verdict.claim_ceiling,
        "claim_flags": sorted(verdict.flags),
        "allowed_claim_categories": verdict.allow(),
        "fail_safe": verdict.fail_safe,
        "evidence_cards": [c.model_dump() for c in cards],
        "recommendations": recs,
    }


def render_triage_text(triage: dict, claim_provenance=None) -> str:
    """Render a claim-clean markdown summary. Asserts the output is clean for the given
    provenance — a prohibited phrase here is a HARD failure."""
    lines = ["# Triage recommendation (screening evidence only)", ""]
    lines.append(f"- candidates evaluated: {triage['n_candidates']}")
    lines.append(f"- claim ceiling: {triage['claim_ceiling']}")
    if triage["claim_flags"]:
        lines.append(f"- flags: {', '.join(triage['claim_flags'])}")
    lines.append("- This workflow does not directly predict kcat or kcat/KM; it "
                 "prioritizes variants for validation.")
    lines.append("")
    for r in triage["recommendations"]:
        lines.append(
            f"- **{r['variant_id']}** → {r['recommendation']}. "
            f"Reasons: {', '.join(r['primary_reason'])}. "
            f"Risks: {', '.join(r['primary_risk'])}. "
            f"Next: {', '.join(r['next_validation'])}.")
    text = "\n".join(lines)
    cg.assert_report_clean(text, claim_provenance)   # guarantee: no leaked over-claim
    return text


def write_triage_artifacts(prov_dir: str, *, claim_provenance=None,
                           out_dir: Optional[str] = None) -> Dict[str, str]:
    """Build the triage + render text, and WRITE both as run artifacts
    (``triage_v3.json`` + ``triage_v3.md``) into ``out_dir`` (defaults to the reports
    dir = the parent of the provenance dir). Returns the written paths."""
    prov = claim_provenance if claim_provenance is not None else cg.ClaimProvenance()
    triage = build_triage(prov_dir, claim_provenance=prov)
    text = render_triage_text(triage, claim_provenance=prov)   # asserts claim-clean
    out_dir = out_dir or os.path.dirname(os.path.normpath(prov_dir))  # .../reports
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "triage_v3.json")
    md_path = os.path.join(out_dir, "triage_v3.md")
    with open(json_path, "w") as fh:
        json.dump(triage, fh, indent=2, default=str)
    with open(md_path, "w") as fh:
        fh.write(text + "\n")
    return {"json": json_path, "md": md_path}


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m evoliez.ranking.triage_report <provenance_dir> [--write]")
        return 2
    prov_dir = argv[0]
    # conservative default provenance: no wet-lab, short MD, no controls -> floor verdict
    prov = cg.ClaimProvenance()
    triage = build_triage(prov_dir, claim_provenance=prov)
    text = render_triage_text(triage, claim_provenance=prov)
    print(text)
    print(f"\n[triage] {triage['n_candidates']} candidates, "
          f"ceiling={triage['claim_ceiling']}, fail_safe={triage['fail_safe']}")
    if "--write" in argv[1:]:
        paths = write_triage_artifacts(prov_dir, claim_provenance=prov)
        print(f"[triage] wrote {paths['md']} + {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
