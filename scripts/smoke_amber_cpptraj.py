#!/usr/bin/env python
"""Increment 3 of the s10 Amber (tier-3) backend: cpptraj analysis smoke.

Runs cpptraj on the increment-2 trajectory (complex.prmtop + prod.nc) to compute
backbone RMSD, ligand RMSD, a ligand-residue distance, and h-bonds, then PARSES
the .dat columns - the analysis mechanics that feed the Amber-backend MDResult
(ligand_rmsd_series / pocket_rmsd_series / key_distances / hbond_occupancy).
Values here reflect the deliberately-minimal smoke MD; the point is the parsing
chain. Run with AmberTools on PATH, AFTER the param + pmemd smokes.
"""
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/tmp/smoke_amber")
prm, traj = WORK / "complex.prmtop", WORK / "prod.nc"
if not (prm.exists() and traj.exists()):
    sys.exit("FAIL: run smoke_amber_param.py + smoke_amber_pmemd.py first")
print("cpptraj:", shutil.which("cpptraj") or "MISSING")

cpptraj_in = """parm complex.prmtop
trajin prod.nc
rms bb @CA,C,N first mass out bbrms.dat
rms lig :LIG&!@H= first nofit out ligrms.dat
distance lig_res1 :LIG :1 out dist.dat
hbond HB out hbond.dat
run
quit
"""
(WORK / "cpptraj.in").write_text(cpptraj_in)
r = subprocess.run(["cpptraj", "-i", "cpptraj.in"], cwd=WORK,
                   capture_output=True, text=True)
print("cpptraj exit:", r.returncode)
if r.returncode != 0:
    print("STDERR:", (r.stderr or "")[-1200:])
    print("STDOUT:", (r.stdout or "")[-800:])


def parse_dat(name):
    """cpptraj .dat -> list of float value-columns per frame (col 0 = frame#)."""
    p = WORK / name
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        rows.append([float(x) for x in parts])
    return rows


print("\n=== parsed outputs ===")
all_ok = True
for name in ("bbrms.dat", "ligrms.dat", "dist.dat", "hbond.dat"):
    rows = parse_dat(name)
    if rows is None:
        print(f"  {name}: ABSENT")
        all_ok = False
        continue
    series = [r[1] for r in rows] if rows and len(rows[0]) > 1 else []
    print(f"  {name}: {len(rows)} frames | series[:3]={[round(s,3) for s in series[:3]]}"
          f" .. last={round(series[-1],3) if series else None}")

print("\nSMOKE:", "PASS - cpptraj parsed RMSD/distance/hbond series"
      if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
