#!/usr/bin/env python3
"""Integrated final ranking for THIS CAR run: merge the two axes the pipeline
produced for the current 16 candidates —
  * BINDING / stability  = s10 MD-lite (per-candidate analysis.json, md_lite_score)
                           + MM-GBSA endpoint (binding_dg, top-3, confounded -> shown not ranked)
  * CATALYTIC geometry   = static near-attack accommodation (nac_accommodation_run.csv):
                           does the mutant pocket orient 3-HP's carboxylate in-line to the
                           ATP alpha-P (not flipped) and anchor it, vs WT which FLIPS it.
The dynamic MD-NAC is 0 for all candidates (needs Mg2+ preorganization + an O->P angle
metric — documented), so the catalytic signal here is the static pass.
Writes final_ranking_integrated_run.csv sorted by a combined score."""
import csv
import glob
import json
import os

RUN = "runs/srcar_3hp_full"
NACCSV = f"{RUN}/reports/nac/nac_accommodation_run.csv"

nac = {r["candidate_id"]: r for r in csv.DictReader(open(NACCSV))}


def md_metrics(cid):
    aj = f"{RUN}/md/{cid}/analysis.json"
    if not os.path.exists(aj):
        return None, None, None
    d = json.load(open(aj))
    bdg = d.get("binding_dg")
    if isinstance(bdg, dict):
        bdg = bdg.get("gbsa")
    return d.get("md_lite_score"), d.get("md_lite_status"), bdg


rows = []
for cid, r in nac.items():
    mdl, mds, gbsa = md_metrics(cid)
    acc = float(r["accommodation_score"])
    notflip = int(r["not_flipped"])
    mdl_v = mdl if isinstance(mdl, (int, float)) else 0.0
    rows.append({
        "candidate_id": cid, "mutation": r["mutation"],
        "md_lite_score": round(mdl_v, 3), "md_lite_status": mds,
        "gbsa_kcal": gbsa,
        "accommodation_score": acc, "not_flipped": notflip,
        "carboxylate_anchor": r["carboxylate_anchor"],
        "delta_anchor_vs_wt": r["delta_anchor_vs_wt"],
        "clashes": r["clashes"], "attack_angle": r["attack_angle"],
        # combined: catalytic accommodation (primary) + binding stability (secondary)
        "combined_score": round(acc + 2.0 * mdl_v, 3),
        "catalytically_productive": bool(notflip) and float(r["delta_anchor_vs_wt"]) >= 0,
    })

rows.sort(key=lambda x: -x["combined_score"])
out = f"{RUN}/reports/nac/final_ranking_integrated_run.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'mutation':34s} {'mdlite':>6s} {'gbsa':>7s} {'accom':>6s} {'flip?':>5s} {'dAnc':>5s} {'comb':>6s} {'prod':>5s}")
for r in rows:
    flip = "keep" if r["not_flipped"] else "FLIP"
    g = f"{r['gbsa_kcal']:.1f}" if isinstance(r["gbsa_kcal"], (int, float)) else "-"
    print(f"{r['mutation'][:34]:34s} {r['md_lite_score']:>6.3f} {g:>7s} "
          f"{r['accommodation_score']:>6.2f} {flip:>5s} {str(r['delta_anchor_vs_wt']):>5s} "
          f"{r['combined_score']:>6.2f} {str(r['catalytically_productive']):>5s}")
print(f"\nwrote {out}  ({len(rows)} candidates)")
