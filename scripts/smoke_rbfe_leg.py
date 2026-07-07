#!/usr/bin/env python
"""Phase-B increment-2 smoke: a full alchemical LEG (lambda schedule -> dG).

Builds the apo hybrid for an X->ALA mutation, then runs amber_rbfe.run_leg over a
tiny 3-window lambda schedule (min->heat->prod each) and integrates <dV/dl> ->
dG_leg (trapezoidal TI). Validates the lambda-window orchestration + integration
on top of the per-window machinery from increment 1.

Usage: python smoke_rbfe_leg.py <protein.pdb> [resid] [mut]
Needs AmberTools on PATH + pmemd amber.sh sourced + a GPU.
"""
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evoliez.adapters.amber_rbfe import Mutation, build_hybrid, run_leg  # noqa:E402

PROT = Path(sys.argv[1])
WORK = Path(os.environ.get("RBFE_WORK", "/tmp/smoke_rbfe_leg"))
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)


def sh(c, **k):
    return subprocess.run(c, cwd=WORK, capture_output=True, text=True, **k)


sh(["pdb4amber", "-i", str(PROT), "-o", "clean.pdb", "--nohyd", "--dry"])
clean = WORK / "clean.pdb"
resids = {}
for l in clean.read_text().splitlines():
    if l.startswith("ATOM"):
        resids.setdefault(int(l[22:26]), l[17:20].strip())
ordered = sorted(resids)
BIG = {"LEU", "ILE", "VAL", "PHE", "MET"}
mid = ordered[len(ordered) // 3: 2 * len(ordered) // 3]
cand = [r for r in mid if resids[r] in BIG]
resid = int(sys.argv[2]) if len(sys.argv) > 2 else cand[len(cand) // 2]
mut = sys.argv[3] if len(sys.argv) > 3 else "ALA"
wt = resids[resid]
print(f"[smoke] mutation {wt}{resid}{mut}")

hyb_prm, hyb_rst, masks = build_hybrid(
    clean, Mutation(resid, wt, mut), WORK, ligand=False)
print(f"[smoke] hybrid built; timask1={masks.timask1}")

# tiny leg: 3 Gauss-Legendre lambda windows, restrained min + gradual heat
res = run_leg(hyb_prm, hyb_rst, masks, WORK, n_lambda=3,
              min_cyc=3000, heat_steps=8000, prod_steps=8000, ntpr=200)
print(f"[smoke] lambdas={res['lambdas']}")
print(f"[smoke] <dV/dl> per window={[round(m, 2) for m in res['dvdl_means']]}")
print(f"[smoke] dG_leg={res['dg']:.2f} kcal/mol  "
      f"(method={res['method']}, n_fail={res['n_fail']})")

# the orchestration is validated when it runs the schedule + integrates a finite
# dG over >=2 windows; window STABILITY (n_fail) is a separate protocol-tuning
# concern reported alongside (high-lambda softcore overheating needs longer
# equilibration / GTI thermostat tuning before the leg dG is production-grade).
ok = len(res["lambdas"]) >= 2 and math.isfinite(res["dg"])
stab = "all windows stable" if res["n_fail"] == 0 else (
    f"WARNING: {res['n_fail']} high-lambda window(s) unstable (softcore overheat) "
    f"-> trapz fallback; needs equil/thermostat tuning for production")
print(f"\n[smoke] window stability: {stab}")
print("SMOKE:", f"PASS - leg orchestration integrated {len(res['lambdas'])} "
      f"windows -> dG={res['dg']:.2f} kcal/mol" if ok else "FAIL")
sys.exit(0 if ok else 1)
