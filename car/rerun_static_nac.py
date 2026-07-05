#!/usr/bin/env python3
"""Static near-attack (NAC) accommodation ranking for THIS run's 16 candidates,
computed on the relaxed Boltz mutant complexes (the mutated sidechain relaxes into
place, so the pocket's real effect on the near-attack 3-HP shows up; the anchored
WT-backbone structures leave the sidechain unrelaxed and don't discriminate).

For each: place the 3-HP carboxylate O in-line to the ATP alpha-P (car_nac) and
score whether the mutant pocket ACCOMMODATES it: wrong_pose (3-HP flipped, hydroxyl
toward P instead of the carboxylate = unproductive), carboxylate anchoring contacts,
steric clashes, attack angle. WT is the baseline. Writes nac_accommodation_run.csv."""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import car_nac  # noqa: E402

RUN = "runs/srcar_3hp_full"


def mutstr(c):
    ms = c.get("mutations") or c.get("mutation_string") or ""
    if isinstance(ms, list):
        out = []
        for m in ms:
            if isinstance(m, dict):
                out.append(f"{m.get('wt','')}{m.get('position','')}{m.get('mut','')}")
            else:
                out.append(str(m))
        return ";".join(p for p in out if p)
    return str(ms)


prov = json.load(open(f"{RUN}/reports/provenance/generated_candidates.json"))
cands = prov if isinstance(prov, list) else prov.get("candidates", prov.get("generated", []))
id2mut = {c.get("candidate_id"): mutstr(c) for c in cands}


def boltz_pdb(cid):
    hits = glob.glob(f"{RUN}/complexes/mutant_boltz/**/{cid}_boltz_input_model_0.pdb",
                     recursive=True)
    return hits[0] if hits else None


wt_pdb = f"{RUN}/complexes/boltz/boltz_results_wt_boltz_input/predictions/wt_boltz_input/wt_boltz_input_model_0.pdb"
wt = car_nac.analyze(wt_pdb, "WT")
wt_anchor = wt["carboxylate_anchor"] if wt else 0

rows = []
for d in sorted(glob.glob(f"{RUN}/md/mut_*/")):
    cid = os.path.basename(d.rstrip("/"))
    p = boltz_pdb(cid)
    if not p:
        continue
    s = car_nac.analyze(p, cid)
    if not s:
        continue
    productive = (s["wrong_pose"] == 0)
    rows.append({
        "candidate_id": cid, "mutation": id2mut.get(cid, cid),
        "d_Oatt_aP": s["d_OattP"], "attack_angle": s["attack_angle"],
        "carboxylate_anchor": s["carboxylate_anchor"],
        "delta_anchor_vs_wt": s["carboxylate_anchor"] - wt_anchor,
        "clashes": s["clashes"], "wrong_pose": s["wrong_pose"],
        "not_flipped": int(productive),
        # rank: not-flipped first, then more anchoring, then fewer clashes
        "accommodation_score": round(3 * productive + s["carboxylate_anchor"]
                                     - 0.3 * s["clashes"], 2),
    })

rows.sort(key=lambda r: -r["accommodation_score"])
out = f"{RUN}/reports/nac/nac_accommodation_run.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nWT baseline: anchor={wt_anchor} wrong_pose={wt['wrong_pose']} "
      f"(WT flips 3-HP = unproductive)" if wt else "WT: analyze failed")
print(f"{'mutation':34s} {'notflip':>7s} {'anchor':>6s} {'dAnc':>5s} {'clash':>5s} {'angle':>6s} {'score':>6s}")
for r in rows:
    print(f"{r['mutation'][:34]:34s} {r['not_flipped']:>7d} {r['carboxylate_anchor']:>6d} "
          f"{r['delta_anchor_vs_wt']:>+5d} {r['clashes']:>5d} {str(r['attack_angle']):>6s} "
          f"{r['accommodation_score']:>6.2f}")
print(f"\nwrote {out}  ({len(rows)} candidates)")
