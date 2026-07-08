#!/usr/bin/env python
"""V6-4 — CAR candidate portfolio (claim-safe mechanism-probe panel).

The V6-3 PMF is NON-CONVERGED (well-overlapped but half-drift 1.4 kcal/mol) and the
in-line attack angle is never productive even at near-attack distance. Ranking on a
non-converged PMF is PROHIBITED (ROADMAP_V6 §6), and the banked CAR V5 sign-offs
forbid a ranked/activity plate. So this emits an HONEST EXPLORATORY panel — a
claim-safe L0 portfolio for structural/binding + reaction-geometry-EVIDENCE +
uncertainty exploration — explicitly NOT an activity ranking and NOT a validated
lead. Every card is L0 (born uncalibrated) and the whole output is ClaimGuard-gated.

Composes existing infra: build_evidence_card_v4, write_candidate_cards, claim_guard.
Draws the variant set from configs/car_v5_focused_candidates.csv and attaches the
V6-2 MD (wt / G430R;S433F;G407K / P438N were run) + V6-3 PMF diagnosis as evidence.

Outputs: outputs/car_v6/{mechanism_probe_plate.csv, evidence_cards.json},
docs/car_v6/car_candidate_portfolio.md
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from evoliez.ranking import claim_guard as cg  # noqa: E402
from evoliez.ranking.claim_guard import ClaimProvenance  # noqa: E402
from evoliez.ranking.evidence_card_v4 import build_evidence_card_v4  # noqa: E402
from evoliez.reports.v4_candidate_card import write_candidate_cards  # noqa: E402

# claim-safe role/lane per v5 category (NO "lead"/"strong"/activity words)
_CATEGORY_ROLE = {
    "control": ("control", "baseline_control"),
    "lead": ("tier_A_catalytic_hypothesis", "multipoint_mechanism_hypothesis"),
    "deconvolution": ("mechanism_probe", "deconvolution_probe"),
    "clean_single": ("mechanism_probe", "single_site_probe"),
    "integrated": ("mechanism_probe", "multipoint_hypothesis"),
    "scalar_control": ("control", "objective_conflict_control"),
}

# The V6-3 diagnosis, applied as panel-wide mechanism context. The in-line angle
# (not the distance) is the unoccupied coordinate; mechanism variants HYPOTHESISE an
# anchoring change toward an in-line approach — to be TESTED, never predicted.
_ANGLE_CONTEXT = ("V6-3 PMF: near-attack distance is accessible under bias but the "
                  "in-line O_nuc-Palpha-O_leaving angle is never productive "
                  "(0/1827 frames); PMF non-converged -> ranking prohibited")

# Per-variant claim-safe test purpose + risk + information-gain (mechanism framing).
# why_include is a TEST PURPOSE / HYPOTHESIS, never an activity claim.
_META = {
    "WT": dict(why="baseline: fixes the Mg placement + the O_nuc->Palpha reference "
                   "geometry every variant is compared against; plate-QC replicate anchor",
               risk="none (reference)", gain="reference", prio="control"),
    "G430R;S433F;G407K": dict(
        why="hypothesis: the combined substitutions re-anchor the 3-HP carboxylate "
            "toward an in-line alpha-P approach (the angle V6-3 found unoccupied); "
            "tests whether the multipoint change shifts the reaction GEOMETRY. Anchors "
            "the deconvolution series",
        risk="multipoint epistasis; possible fold destabilisation", gain="high", prio="A"),
    "G430R": dict(why="deconvolution: isolates the Arg430 carboxylate-anchor contribution",
                  risk="single may not reproduce the multipoint geometry", gain="high", prio="A"),
    "S433F": dict(why="deconvolution: isolates the Phe433 pocket-packing contribution",
                  risk="hydrophobic swap may mis-pack", gain="medium", prio="B"),
    "G407K": dict(why="deconvolution: isolates the Lys407 charge/anchor contribution",
                  risk="Lys flexibility; ambiguous anchoring", gain="medium", prio="B"),
    "G430R;S433F": dict(why="deconvolution: Arg+Phe pair without the Lys",
                        risk="pairwise epistasis", gain="medium", prio="B"),
    "G430R;G407K": dict(why="deconvolution: Arg+Lys pair without the Phe",
                        risk="two anchors may clash", gain="medium", prio="B"),
    "S433F;G407K": dict(why="deconvolution: Phe+Lys pair without the Arg anchor",
                        risk="may lack the primary carboxylate anchor", gain="medium", prio="C"),
    "P438N": dict(why="single-site probe at the cleanest design position P438 "
                      "(low structural risk); V6-2 MD run — ligand retained, stable",
                  risk="low (clean single)", gain="medium", prio="B"),
    "P438R": dict(why="single-site probe: P438 Arg (alternative carboxylate anchor near the pocket)",
                  risk="low-medium", gain="medium", prio="B"),
    "P438K": dict(why="single-site probe: P438 Lys charge variant",
                  risk="low-medium", gain="low", prio="C"),
    "P438Q": dict(why="single-site probe: P438 Gln polar variant",
                  risk="low", gain="low", prio="C"),
    "P438S": dict(why="single-site probe: P438 Ser small-polar variant",
                  risk="low", gain="low", prio="C"),
    "P438T": dict(why="single-site probe: P438 Thr small-polar variant",
                  risk="low", gain="low", prio="C"),
    "Y264F;T265S;G274A": dict(why="alternative multipoint geometry hypothesis (A3 loop)",
                              risk="multipoint epistasis", gain="medium", prio="B"),
    "P438R;G407H": dict(why="alternative multipoint hypothesis (pocket + His anchor)",
                        risk="multipoint epistasis", gain="medium", prio="B"),
    "T265S;G274A;A275V": dict(why="binding/stability control: MD-lite favourable but "
                                  "reaction-geometry-neutral — separates binding from geometry",
                              risk="none (control)", gain="control", prio="control"),
    "G430H;S433I": dict(why="objective-conflict control: pipeline scalar-ranked high with "
                            "reaction-geometry evidence ~0 — probes scalar-vs-geometry divergence",
                        risk="none (control)", gain="control", prio="control"),
    "G430H;P438L;A275I": dict(why="objective-conflict control: second scalar-top, geometry ~0",
                              risk="none (control)", gain="control", prio="control"),
}


def load_md_evidence():
    """Map the 3 V6-2 MD-run systems onto their panel mutations."""
    p = REPO / "reports/provenance/amber_gpu_jobs.jsonl"
    id_to_mut = {"wt": "WT", "mut_00000": "G430R;S433F;G407K", "mut_00001": "P438N"}
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            mut = id_to_mut.get(r["job_id"])
            if mut:
                a = r.get("analysis", {})
                out[mut] = {
                    "md_status": r["status"],
                    "ligand_retention": a.get("ligand_retention"),
                    "ligand_rmsd_final_A": a.get("ligand_rmsd_final_A"),
                    "energy_drift": a.get("energy_drift"),
                }
    return out


def load_pmf_context():
    p = REPO / "reports/provenance/amber_pmf_jobs.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if rows:
            r = rows[-1]
            return {"verdict": r["verdict"], "classification": r["classification"],
                    "angle_occ": r["qc"].get("near_attack_angle_occupancy"),
                    "overlap_min": r["qc"].get("overlap", {}).get("min_overlap")}
    return {}


def accommodation_for(mut, category, md):
    """Claim-safe axis scores. reaction_geometry stays LOW for ALL (no converged
    geometry evidence — the v5 table shows nac_status=none everywhere and V6-3 is
    non-converged); differentiation is by TEST PURPOSE, not score. Structural
    viability uses V6-2 MD where available. pose_uncertainty is HIGH."""
    md_run = md.get(mut)
    struct = 0.75 if (md_run and md_run.get("ligand_retention") == 1.0) else 0.5
    return {
        "structural_viability": struct,
        "evolutionary_tolerance": 0.4,
        "ligand_cofactor_competence": 0.5 if md_run else 0.4,
        "substrate_positioning": 0.2,
        "reaction_geometry_accommodation": 0.15,   # LOW for all — no converged geometry
        "reference_state_accommodation": 0.4,
        "pose_uncertainty": 0.75,                   # per-mutant Boltz noise + PMF non-converged
        "experimental_calibration": 0.0,            # -> L0
    }


def main() -> int:
    variants = list(csv.DictReader(
        (REPO / "configs/car_v5_focused_candidates.csv").read_text().splitlines()))
    md = load_md_evidence()
    pmf = load_pmf_context()

    cards, plate_rows = [], []
    prov = ClaimProvenance()   # default: no wetlab, short MD, no controls -> strict
    wells = [f"{r}{c}" for r in "ABCD" for c in range(1, 7)]   # 24-well
    for i, v in enumerate(variants):
        mut = v["mutation_string"]
        category = v["category"]
        role, lane = _CATEGORY_ROLE.get(category, ("mechanism_probe", category))
        meta = _META.get(mut, dict(why=v["rationale"], risk="unspecified",
                                   gain="medium", prio="C"))
        cid = f"car_v6_{i:02d}"
        md_run = md.get(mut)
        ev_source = ["car_v5_focused_design", "v6_3_pmf_diagnostic"]
        if md_run:
            ev_source.append("v6_2_amber_md")
        seed = SimpleNamespace(candidate_id=cid, mutation=mut, role=role,
                               evidence_source=ev_source,
                               claim_level="L0_uncalibrated")
        card = build_evidence_card_v4(seed, accommodation_for(mut, category, md))
        # attach mechanism context + MD evidence to the geometry axis provenance
        card.reaction_geometry_accommodation.provenance.update({
            "panel_context": _ANGLE_CONTEXT, "pmf_verdict": pmf.get("verdict"),
            "pmf_classification": pmf.get("classification")})
        if md_run:
            card.structural_viability.provenance.update(md_run)
        cards.append(card)

        why = meta["why"]
        cg.assert_clean(why)      # every rationale string is ClaimGuard-clean
        plate_rows.append({
            "plate": "CAR_V6_probe_plate", "well": wells[i] if i < len(wells) else f"X{i}",
            "variant_id": cid, "candidate_id": cid, "mutation": mut, "lane": lane,
            "all_lanes": lane, "why_include": why, "risk": meta["risk"],
            "expected_information_gain": meta["gain"], "assay_priority": meta["prio"],
            "codon_design_note": "site-saturation not required (defined substitution)",
        })

    out_dir = REPO / "outputs/car_v6"
    out_dir.mkdir(parents=True, exist_ok=True)
    # evidence_cards.json (V4, claim-level-gated) via the canonical writer
    write_candidate_cards(cards, out_dir / "_cards.md", out_dir / "evidence_cards.json")
    (out_dir / "_cards.md").unlink(missing_ok=True)

    # mechanism-probe plate CSV (library_plan schema). NOT named "wetlab_plate":
    # a file literally called that reads as a validated experimental recommendation,
    # which this panel is not (reviewer claim-safety note). A leading caveat comment
    # states the claim boundary in the artifact itself.
    fields = ["plate", "well", "variant_id", "candidate_id", "mutation", "lane",
              "all_lanes", "why_include", "risk", "expected_information_gain",
              "assay_priority", "codon_design_note"]
    caveat = ("# hypothesis-grade mechanism-probe panel; screening-level "
              "reaction-geometry evidence only. Assay deconvolution variants + "
              "controls together. Purpose is mechanism hypothesis-probing, not lead "
              "validation. Fulfils the ROADMAP_V6 V6-4 plate deliverable "
              "(renamed from wetlab_plate_24.csv for claim-safety). "
              "See docs/car_v6/car_candidate_portfolio.md")
    plate_path = out_dir / "mechanism_probe_plate.csv"
    with plate_path.open("w", newline="") as fh:
        fh.write(caveat + "\n")
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(plate_rows)

    # final ClaimGuard gate on the CSV content as a whole (caveat line included)
    cg.assert_report_clean(plate_path.read_text(), prov)

    n_A = sum(1 for r in plate_rows if r["assay_priority"] == "A")
    n_ctrl = sum(1 for r in plate_rows if r["assay_priority"] == "control")
    print(f"portfolio: {len(plate_rows)} variants "
          f"({n_A} tier-A hypotheses, {n_ctrl} controls, "
          f"{len(plate_rows)-n_A-n_ctrl} probes)")
    print(f"cards: {len(cards)} (all L0_uncalibrated); "
          f"PMF verdict={pmf.get('verdict')}; ClaimGuard: clean")
    print(f"wrote {out_dir}/mechanism_probe_plate.csv + evidence_cards.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
