#!/usr/bin/env python3
"""List the current run's 16 MD candidates -> mutation strings, so the catalytic
ranking (static NAC) can be checked for relevance and the README lists real muts."""
import glob
import json
import os

RUN = "runs/srcar_3hp_full"

prov = json.load(open(f"{RUN}/reports/provenance/generated_candidates.json"))
cands = prov if isinstance(prov, list) else prov.get("candidates", prov.get("generated", []))


def mutstr(c):
    ms = c.get("mutations") or c.get("mutation_string") or ""
    if isinstance(ms, list):
        # list of dicts or strings
        parts = []
        for m in ms:
            if isinstance(m, dict):
                parts.append(f"{m.get('wt','')}{m.get('position','')}{m.get('mut','')}")
            else:
                parts.append(str(m))
        return ";".join(p for p in parts if p)
    return str(ms)


id2mut = {c.get("candidate_id"): mutstr(c) for c in cands}

md_ids = sorted(os.path.basename(d.rstrip("/"))
                for d in glob.glob(f"{RUN}/md/mut_*/"))
print(f"MD candidates: {len(md_ids)}")
for cid in md_ids:
    print(f"  {cid}\t{id2mut.get(cid, '(mutation not found in provenance)')}")
