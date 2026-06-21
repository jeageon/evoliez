#!/usr/bin/env python
"""End-to-end test of the Amber tier-3 backend (run_md_amber).

Builds a protein + ethanol complex (same recipe as verify_s10_md_real) and runs
run_md_amber -> MDResult, checking status + populated geometry. Env:
  source <pmemd build>/pathscripts/amber.sh         # pmemd.cuda LD_LIBRARY_PATH
  export AMBERHOME=<AmberTools25>; PATH=$AMBERHOME/bin:$PATH   # tleap/antechamber
  export EVOLIEZ_PMEMD_CUDA=<.../pmemd.cuda_SPFP>; CUDA_VISIBLE_DEVICES=<gpu>
  python verify_amber_engine.py <protein.pdb>
"""
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from evoliez.types import Complex, ProteinStructure, Ligand
from evoliez.config import MDConfig
from evoliez.adapters.amber_engine import run_md_amber
from evoliez.adapters.openmm_engine import _pdb_one_letter_seq

PROT = Path(sys.argv[1])
LIG_SMILES = sys.argv[2] if len(sys.argv) > 2 else "CCO"
WORK = Path("/tmp/verify_amber")
WORK.mkdir(parents=True, exist_ok=True)


def _xyz(l):
    return np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])


# --- protein + ethanol complex (dense-CA placement; verify recipe) ----------
prot = [l for l in PROT.read_text().splitlines() if l.startswith("ATOM")]
pc = np.array([_xyz(l) for l in prot])
cen = pc.mean(0)
ca = np.array([_xyz(l) for l in prot if l[12:16].strip() == "CA"])
_d2 = ((ca[:, None, :] - ca[None, :, :]) ** 2).sum(-1)
anchor = ca[int(np.argmax((_d2 <= 8.0 ** 2).sum(1)))]
outv = anchor - cen
outv = outv / (np.linalg.norm(outv) + 1e-9)

m = Chem.AddHs(Chem.MolFromSmiles(LIG_SMILES))
AllChem.EmbedMolecule(m, randomSeed=7)
AllChem.MMFFOptimizeMolecule(m)
m = Chem.RemoveHs(m)
conf = m.GetConformer()
lp = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
lp -= lp.mean(0)
ctr = anchor.copy()
for _ in range(60):
    if np.sqrt((((lp + ctr)[:, None, :] - pc[None, :, :]) ** 2).sum(-1)).min() >= 2.6:
        break
    ctr = ctr + outv * 0.3
for i in range(m.GetNumAtoms()):
    conf.SetAtomPosition(i, Point3D(*(lp[i] + ctr)))

maxser = max(int(l[6:11]) for l in prot)
lig_lines = []
for ln in Chem.MolToPDBBlock(m, flavor=0).splitlines():
    if ln.startswith(("HETATM", "ATOM")):
        ln = "HETATM" + ln[6:17] + "LIG" + ln[20:]
        ln = ln[:6] + f"{int(ln[6:11]) + maxser:5d}" + ln[11:]
        lig_lines.append(ln)
    elif ln.startswith("CONECT"):
        b = ln[6:].rstrip()
        ns = [int(b[i:i + 5]) + maxser for i in range(0, len(b), 5) if b[i:i + 5].strip()]
        lig_lines.append("CONECT" + "".join(f"{n:5d}" for n in ns))
cpdb = WORK / "complex.pdb"
cpdb.write_text("\n".join(prot + lig_lines + ["END"]) + "\n")

cas = [(int(l[22:26]), _xyz(l)) for l in prot if l[12:16].strip() == "CA"]
cas.sort(key=lambda rc: float(((rc[1] - ctr) ** 2).sum()))
cat = sorted({rc[0] for rc in cas[:3]})
seq = _pdb_one_letter_seq(cpdb)
cx = Complex(structure=ProteinStructure(sequence=seq, pdb_path=str(cpdb)),
             ligand=Ligand(id="LIG", smiles=LIG_SMILES, formal_charge=0))
print(f"[setup] protein {len(prot)} atoms | ligand {LIG_SMILES} | catalytic {cat}")

# --- run the Amber tier-3 backend -------------------------------------------
import os
cfg = MDConfig(enabled=True, engine="amber", protocol_level=1,
               solvent=os.environ.get("VERIFY_SOLVENT", "implicit"),
               production_ns=0.002, equilibration_ps=2.0)
print(f"[setup] solvent={cfg.solvent}")
r = run_md_amber(cx, "verify_amber", cfg, WORK,
                 catalytic_positions=cat, dry_run=False)

print("\n=== MDResult ===")
print("status            :", r.status)
print("solvent_mode      :", r.solvent_mode)
print("simulation_time_ns:", r.simulation_time_ns)
print("ligand_rmsd       :", r.ligand_rmsd_series[:3], "... last",
      r.ligand_rmsd_series[-1] if r.ligand_rmsd_series else None)
print("pocket_rmsd       :", r.pocket_rmsd_series[:3])
print("key_distances     :", {k: v[:3] for k, v in r.key_distances.items()})
print("hbond_occupancy   :", r.hbond_occupancy)
print("energy_drift      :", r.energy_drift)
print("failure_reason    :", r.failure_reason)

ok = (r.status in ("ok", "unstable")
      and len(r.ligand_rmsd_series) > 1 and bool(r.key_distances))
print("\nVERIFY:", "PASS - Amber tier-3 MDResult populated" if ok else "FAIL")
sys.exit(0 if ok else 1)
