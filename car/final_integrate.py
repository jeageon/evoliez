#!/usr/bin/env python3
"""Final integrated ranking: merge s10 binding/stability (md_lite, MM-GBSA) with the
CAR near-attack catalytic-geometry pass (Δcarboxylate-anchor, no-flip). Two axes,
one honest ranking. Writes final_ranking_integrated.csv + prints the top."""
import json, glob, csv, sys

RUN = "/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_full"
mc = json.load(open(f"{RUN}/reports/provenance/md_candidates.json"))["candidates"]
bybind = {}
for x in mc:
    bybind[x.get("mutation_string")] = (x.get("md_lite_score"), (x.get("binding_dg") or {}).get("gbsa"))

nac = {}
with open(f"{RUN}/reports/nac/nac_accommodation.csv") as fh:
    for r in csv.DictReader(fh):
        nac[r["mutation"]] = (int(r["delta_anchor_vs_wt"]), int(r["wrong_pose"]), int(r["clashes_min"]))

rows = []
for ms, (dl, gb) in bybind.items():
    if ms not in nac or dl is None: continue
    danch, wrong, clash = nac[ms]
    # catalytic productivity: must not flip 3-HP, reward carboxylate anchoring near alpha-P
    cat_ok = (wrong == 0 and danch >= 0)
    cat = danch - 3 * wrong - 0.15 * clash
    combined = (1.0 * cat) + (2.5 * dl)          # catalytic geometry + binding stability
    rows.append((ms, round(dl, 3), gb, danch, wrong, clash, round(combined, 3), cat_ok))

rows.sort(key=lambda r: (-int(r[7]), -r[6]))     # productive first, then combined score
print("=== FINAL INTEGRATED RANKING (binding + catalytic geometry) ===")
print("%-24s md_lite gbsa   Δanch flip clash  score  productive" % "mutation")
for r in rows[:15]:
    print("%-24s %6s %6s  %+4d   %d   %3d  %6s   %s" %
          (r[0], r[1], r[2] if r[2] is not None else "-", r[3], r[4], r[5], r[6], "YES" if r[7] else "no"))
with open(f"{RUN}/reports/nac/final_ranking_integrated.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["mutation","md_lite","gbsa","delta_anchor","wrong_pose","clashes","combined_score","catalytically_productive"])
    w.writerows(rows)
print("\nproductive (no-flip, anchoring gained):", sum(1 for r in rows if r[7]), "of", len(rows))
print("saved: final_ranking_integrated.csv")
