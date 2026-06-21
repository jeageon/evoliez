#!/usr/bin/env python
"""Anti-s06 verification of s08b real mutant Boltz Δ.

The s08b analog of the s06 corruption would be: mutant folds == WT, so the real
Boltz Δ is identically 0 (no signal). REAL data: each of the 40 mutant folds has
Boltz confidence metrics DISTINCT from WT, giving a distributed d_ligand_iptm /
d_confidence that carries the per-mutant signal s09 feeds into MD selection.
Reads the Boltz confidence jsons directly (scalar metrics — no frame alignment).
"""
import glob
import json
import statistics
import sys
from pathlib import Path

RD = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track"
wtc = glob.glob(str(Path(RD) / "complexes/boltz/**/confidence_wt_boltz_input_model_0.json"),
                recursive=True)[0]
wt = json.load(open(wtc))
print(f"WT: conf={wt['confidence_score']:.3f} iptm={wt['iptm']:.3f} "
      f"ligand_iptm={wt['ligand_iptm']:.3f} plddt={wt['complex_plddt']:.3f}")
muts = sorted(glob.glob(str(
    Path(RD) / "complexes/mutant_boltz/_batch_out_gpu*/boltz_results_*/"
    "predictions/*/confidence_*_model_0.json")))
print(f"{len(muts)} mutant folds\n")

d_lig, d_conf, rows = [], [], []
for m in muts:
    c = json.load(open(m))
    cid = Path(m).parent.name.replace("_boltz_input", "")
    dl = c["ligand_iptm"] - wt["ligand_iptm"]
    dc = c["confidence_score"] - wt["confidence_score"]
    d_lig.append(dl)
    d_conf.append(dc)
    rows.append((cid, c["ligand_iptm"], dl, dc))

for cid, li, dl, dc in sorted(rows, key=lambda r: r[2]):
    print(f"  {cid}: ligand_iptm={li:.3f} (Δ{dl:+.3f})  confΔ{dc:+.3f}")


def st(xs):
    return (f"n={len(xs)} min={min(xs):+.3f} max={max(xs):+.3f} "
            f"mean={statistics.mean(xs):+.3f} std={statistics.pstdev(xs):.3f}")


print(f"\n[s08b] d_ligand_iptm: {st(d_lig)}")
print(f"[s08b] d_confidence:  {st(d_conf)}")
identical = sum(1 for x in d_lig if abs(x) < 0.005)
print(f"[s08b] mutants ~identical to WT (|Δligand_iptm|<0.005): {identical}/{len(d_lig)}")
real = (len(d_lig) >= 20 and statistics.pstdev(d_lig) > 0.02
        and identical < len(d_lig) // 4)
print("VERDICT:", "REAL — mutant folds carry distinct, distributed Δ vs WT (real signal)"
      if real else "SUSPICIOUS — folds too close to WT (possible non-signal)")
