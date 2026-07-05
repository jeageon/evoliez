import json, glob, os
RUN = "/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_full"

mc = json.load(open(f"{RUN}/reports/provenance/md_candidates.json"))
print("=== MD candidates:", len(mc), "===")
print("keys:", list(mc[0].keys()) if mc else [])

def score(x):
    return x.get("md_lite_score") or x.get("final_score") or x.get("md_score") or 0
top = sorted(mc, key=score, reverse=True)[:10]
for x in top:
    ms = x.get("mutation_string") or x.get("candidate_id")
    bg = (x.get("binding_dg") or {}).get("gbsa")
    nac = x.get("nac_occupancy", x.get("nac_status", "n/a"))
    print("  %-24s md_lite=%s gbsa=%s nac=%s" % (ms, x.get("md_lite_score"), bg, nac))

print("\n=== evidence cards (v3) ===")
f = glob.glob(f"{RUN}/**/evidence_cards.json", recursive=True)
if f:
    e = json.load(open(f[0]))
    e = e if isinstance(e, list) else e.get("cards", [])
    print("cards:", len(e))
    c = e[0]
    for k, v in c.items():
        print("  %s: %s" % (k, str(v)[:120]))

print("\n=== final_report.md (head) ===")
fr = f"{RUN}/reports/final_report.md"
if os.path.exists(fr):
    print("".join(open(fr).readlines()[:40]))
