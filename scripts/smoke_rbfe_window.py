#!/usr/bin/env python
"""Phase-B increment-1 smoke: one full mutation-TI window, end to end.

Builds the APO hybrid topology for a real X->ALA mutation via
amber_rbfe.build_hybrid (solvate WT -> swap residue keeping waters ->
softcore_setup -> hybrid + masks), then runs min -> heat -> TI production at
lambda=0.5 on pmemd.cuda and checks dV/dl is produced. This validates the
mutation softcore-TI integration (hybrid topology + unique-atom masks + GPU TI
engine together) -- the last de-risk before the lambda/leg orchestration.

Usage: python smoke_rbfe_window.py <protein.pdb> [resid] [mut_resname]
Needs AmberTools on PATH + pmemd amber.sh sourced + a GPU.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evoliez.adapters.amber_rbfe import (   # noqa: E402
    Mutation, build_hybrid, mdin_ti_min, mdin_ti_heat, mdin_ti_prod)

PROT = Path(sys.argv[1])
WORK = Path(os.environ.get("RBFE_WORK", "/tmp/smoke_rbfe"))
PMEMD = os.environ.get(
    "PMEMD", "/mnt/data/jglee/pmemd24_src/build/src/pmemd/src/pmemd.cuda_SPFP")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)


def sh(cmd, **k):
    return subprocess.run(cmd, cwd=WORK, capture_output=True, text=True, **k)


# clean
sh(["pdb4amber", "-i", str(PROT), "-o", "clean.pdb", "--nohyd", "--dry"])
clean = WORK / "clean.pdb"
lines = clean.read_text().splitlines()

# pick a sizeable hydrophobic sidechain away from the termini, mutate -> ALA
# (alanine scan; hydrophobic => no charge change => cleanest first TI smoke)
BIG = {"LEU", "ILE", "VAL", "PHE", "MET"}
resids = {}
for l in lines:
    if l.startswith("ATOM"):
        resids.setdefault(int(l[22:26]), l[17:20].strip())
ordered = sorted(resids)
if len(sys.argv) > 2:
    resid = int(sys.argv[2])
    mut = sys.argv[3] if len(sys.argv) > 3 else "ALA"
else:
    mid = ordered[len(ordered) // 3: 2 * len(ordered) // 3]   # buried core, stable
    cand = [ri for ri in mid if resids[ri] in BIG]
    if not cand:
        sys.exit("FAIL: no sizeable sidechain to mutate")
    resid, mut = cand[len(cand) // 2], "ALA"
wt = resids[resid]
print(f"[smoke] mutation {wt}{resid}{mut}  (protein {len(ordered)} residues)")

# build the apo hybrid
hyb_prm, hyb_rst, masks = build_hybrid(
    clean, Mutation(resid, wt, mut), WORK, ligand=False)
print(f"[smoke] hybrid={hyb_prm.name}  timask1={masks.timask1}  "
      f"timask2={masks.timask2}")
print(f"[smoke] noshakemask={masks.noshakemask}")

CLAM = 0.5


def run(tag, mdin, cin, rout, oout=None, ref=None):
    (WORK / f"{tag}.in").write_text(mdin)
    cmd = [PMEMD, "-O", "-i", f"{tag}.in", "-o", oout or f"{tag}.out",
           "-p", hyb_prm.name, "-c", cin, "-r", rout]
    if ref:                       # ntr=1 (restrained min/heat) needs a reference
        cmd += ["-ref", ref]
    r = sh(cmd, timeout=1200)
    ok = (WORK / rout).exists()
    print(f"  [{tag}] exit={r.returncode}  {rout}={ok}")
    if not ok:
        print("    stderr:", (r.stderr or "")[-300:])
        for ln in (WORK / (oout or f"{tag}.out")).read_text().splitlines() \
                if (WORK / (oout or f"{tag}.out")).exists() else []:
            if "ERROR" in ln or "Terminat" in ln:
                print("    out:", ln.strip())
    return ok


ok = run("ti_min", mdin_ti_min(masks, CLAM, maxcyc=2000), hyb_rst.name,
         "min.rst", ref=hyb_rst.name)
ok = ok and run("ti_heat", mdin_ti_heat(masks, CLAM, nsteps=5000),
                "min.rst", "heat.rst", ref=hyb_rst.name)
ok = ok and run("ti_prod", mdin_ti_prod(masks, CLAM, nsteps=5000, ntpr=200),
                "heat.rst", "prod.rst", "prod.out")

dvdl = []
if (WORK / "prod.out").exists():
    for line in (WORK / "prod.out").read_text().splitlines():
        m = re.search(r"DV/DL\s*=\s*(-?\d+\.\d+)", line)
        if m:
            dvdl.append(float(m.group(1)))
print(f"[smoke] DV/DL samples: {dvdl[:5]} (n={len(dvdl)})")

good = ok and len(dvdl) > 0
print("\nSMOKE:", "PASS - mutation softcore-TI window produced dV/dl"
      if good else "FAIL (see above)")
sys.exit(0 if good else 1)
