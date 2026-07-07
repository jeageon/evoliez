#!/usr/bin/env python
"""Phase-B increment 1: pmemd.cuda softcore-TI ENGINE smoke (EXPLICIT).

Confirms the server's pmemd.cuda_SPFP runs softcore alchemical TI and emits
DV/DL. pmemd.cuda TI requires EXPLICIT solvent (CUDA TI is unsupported with
implicit GB) and softcore requires ntf=1 + noshakemask on the perturbed region.
Runs one ligand-annihilation TI window (timask1/scmask1=:LIG) at lambda=0.5 from
the explicit equilibrated restart produced by
`verify_amber_engine.py VERIFY_SOLVENT=explicit` (/tmp/verify_amber). Needs the
pmemd amber.sh sourced + a GPU.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("TI_WORK", "/tmp/verify_amber"))
PMEMD = os.environ.get(
    "PMEMD", "/mnt/data/jglee/pmemd24_src/build/src/pmemd/src/pmemd.cuda_SPFP")
prm = WORK / "complex.prmtop"
if not prm.exists():
    sys.exit("FAIL: need an EXPLICIT complex.prmtop - run "
             "verify_amber_engine.py with VERIFY_SOLVENT=explicit first")
start = "equil.rst" if (WORK / "equil.rst").exists() else "complex.inpcrd"
print(f"pmemd: {PMEMD}\nWORK:  {WORK}   start: {start}")

(WORK / "ti.in").write_text(
    "softcore TI ligand annihilation, clambda=0.5 (explicit NVT)\n&cntrl\n"
    " imin=0, nstlim=500, dt=0.001, irest=1, ntx=5,\n"
    " ntb=1, cut=10.0, iwrap=1,\n"
    " ntc=2, ntf=1, noshakemask=':LIG',\n"
    " ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,\n"
    " icfe=1, ifsc=1, clambda=0.5,\n"
    " timask1=':LIG', timask2='',\n"
    " scmask1=':LIG', scmask2='',\n"
    " ntpr=50,\n/\n")
r = subprocess.run([PMEMD, "-O", "-i", "ti.in", "-o", "ti.out", "-p",
                    "complex.prmtop", "-c", start, "-r", "ti.rst"],
                   cwd=WORK, capture_output=True, text=True, timeout=600)
print("TI exit:", r.returncode)
if r.returncode != 0:
    print("STDERR:", (r.stderr or "")[-400:])
    if (WORK / "ti.out").exists():
        print("--- ti.out errors ---")
        for ln in (WORK / "ti.out").read_text().splitlines():
            if "ERROR" in ln or "Terminat" in ln:
                print(ln)

dvdl = []
out = WORK / "ti.out"
if out.exists():
    for line in out.read_text().splitlines():
        m = re.search(r"DV/DL\s*=\s*(-?\d+\.\d+)", line)
        if m:
            dvdl.append(float(m.group(1)))
print(f"DV/DL samples: {dvdl[:5]} (n={len(dvdl)})")

ok = r.returncode == 0 and len(dvdl) > 0
print("\nSMOKE:", "PASS - pmemd.cuda softcore TI (explicit) works, DV/DL produced"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
