#!/usr/bin/env python
"""Increment 1 of the s10 Amber (tier-3) backend: tleap parameterization smoke.

Builds a protein + small-ligand (ethanol) complex and runs the Amber
small-molecule param chain to produce an Amber topology:
  protein : pdb4amber  -> ff14SB (tleap loadpdb)
  ligand  : antechamber (AM1-BCC, gaff2) -> parmchk2 (frcmod)
  combine : tleap -> complex.prmtop + complex.inpcrd   (implicit-ready, mbondi3)

Run with a python that has rdkit (evoliez-cu124) AND AmberTools on PATH:
  PATH=<AmberTools25/bin>:$PATH python smoke_amber_param.py <protein.pdb>
"""
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

PROT = Path(sys.argv[1])
LIG_SMILES = sys.argv[2] if len(sys.argv) > 2 else "CCO"
LIG_CHARGE = sys.argv[3] if len(sys.argv) > 3 else "0"
WORK = Path("/tmp/smoke_amber")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True)
    tag = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(f"  -> {tag}")
    if r.returncode != 0:
        print("  STDOUT:", (r.stdout or "")[-1200:])
        print("  STDERR:", (r.stderr or "")[-1200:])
    return r.returncode == 0


print("=== tool availability ===")
for t in ("pdb4amber", "antechamber", "parmchk2", "sqm", "tleap"):
    print(f"  {t}: {shutil.which(t) or 'MISSING'}")


def _xyz(line):
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


# --- protein (ATOM only) + dense-CA ligand placement (reuse verify recipe) ---
prot = [l for l in PROT.read_text().splitlines() if l.startswith("ATOM")]
if not prot:
    sys.exit(f"FAIL: no ATOM records in {PROT}")
pcoords = np.array([_xyz(l) for l in prot])
centroid = pcoords.mean(0)
ca = np.array([_xyz(l) for l in prot if l[12:16].strip() == "CA"])
_d2 = ((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1)
anchor = ca[int(np.argmax((_d2 <= 8.0 ** 2).sum(1)))]
outv = anchor - centroid
outv = outv / (np.linalg.norm(outv) + 1e-9)
(WORK / "protein_atom.pdb").write_text("\n".join(prot) + "\nEND\n")

m = Chem.AddHs(Chem.MolFromSmiles(LIG_SMILES))
AllChem.EmbedMolecule(m, randomSeed=7)
AllChem.MMFFOptimizeMolecule(m)
conf = m.GetConformer()
lpos = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
lpos -= lpos.mean(0)
center = anchor.copy()
for _ in range(60):
    if np.sqrt((((lpos + center)[:, None, :] - pcoords[None, :, :]) ** 2)
               .sum(-1)).min() >= 2.6:
        break
    center = center + outv * 0.3
for i in range(m.GetNumAtoms()):
    conf.SetAtomPosition(i, Point3D(*(lpos[i] + center)))
with Chem.SDWriter(str(WORK / "ligand.sdf")) as w:
    w.write(m)
print(f"\n[setup] protein {len(prot)} atoms | ligand {LIG_SMILES} "
      f"({m.GetNumAtoms()} atoms, charge {LIG_CHARGE}) placed near a dense CA")

# --- Amber param chain ------------------------------------------------------
steps_ok = True
steps_ok &= run(["pdb4amber", "-i", "protein_atom.pdb", "-o", "protein_clean.pdb",
                 "--nohyd", "--dry"])
steps_ok &= run(["antechamber", "-i", "ligand.sdf", "-fi", "sdf",
                 "-o", "ligand.mol2", "-fo", "mol2", "-c", "bcc",
                 "-nc", LIG_CHARGE, "-at", "gaff2", "-rn", "LIG"])
steps_ok &= run(["parmchk2", "-i", "ligand.mol2", "-f", "mol2",
                 "-o", "ligand.frcmod"])

tleap_in = """source leaprc.protein.ff14SB
source leaprc.gaff2
loadamberparams ligand.frcmod
LIG = loadmol2 ligand.mol2
prot = loadpdb protein_clean.pdb
comp = combine {prot LIG}
set default PBRadii mbondi3
saveamberparm comp complex.prmtop complex.inpcrd
savepdb comp complex.pdb
quit
"""
(WORK / "tleap.in").write_text(tleap_in)
steps_ok &= run(["tleap", "-s", "-f", "tleap.in"])

# --- verdict ----------------------------------------------------------------
prm = WORK / "complex.prmtop"
crd = WORK / "complex.inpcrd"
ok = (steps_ok and prm.exists() and prm.stat().st_size > 0 and crd.exists())
print(f"\nprmtop: {prm.exists()} "
      f"({prm.stat().st_size if prm.exists() else 0} B) | inpcrd: {crd.exists()}")
if (WORK / "leap.log").exists():
    tail = (WORK / "leap.log").read_text().splitlines()[-12:]
    print("--- leap.log tail ---")
    print("\n".join(tail))
print("\nSMOKE:", "PASS - Amber topology built" if ok else "FAIL")
sys.exit(0 if ok else 1)
