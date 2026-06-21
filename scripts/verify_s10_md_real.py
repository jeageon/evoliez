#!/usr/bin/env python
"""Server-only REAL OpenMM MD verification for the s10 changes:
  #1  solvent honesty  (solvent='explicit' -> warn + report 'implicit')
  #2  real-trajectory geometry (key_distances / contact_occupancy / hbond)

Builds a complex from a REAL full-atom protein PDB (argv[1]) + a small
GAFF-parameterizable ligand (phenol: aromatic ring -> contacts, OH -> exercises
the H-bond proxy) placed in contact with the nearest surface residues, then runs
the REAL OpenMM path (`_run_real`) at tiny step counts and inspects the result.

CPU is forced (CUDA_VISIBLE_DEVICES="" + apply_gpu_selection no-op) because the
box's conda OpenMM CUDA build hits CUDA_ERROR_UNSUPPORTED_PTX_VERSION; the
analysis code under test is platform-independent.

Usage:  python verify_s10_md_real.py /path/to/protein_model_0.pdb
"""
import os
# Default forces CPU (the conda OpenMM CUDA build is PTX-broken on the 12.4
# driver). Set VERIFY_FORCE_CPU=0 (+ CUDA_VISIBLE_DEVICES=<gpu>) to let OpenMM
# auto-select a GPU platform (it picks OpenCL, since CUDA fails to load).
if os.environ.get("VERIFY_FORCE_CPU", "1") != "0":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENMM_CPU_THREADS", "4") # modest on the shared box

import sys
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("verify_s10")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import evoliez.adapters.openmm_engine as E
E.apply_gpu_selection = lambda *a, **k: None      # never pick a GPU (force CPU)

from evoliez.types import Complex, ProteinStructure, Ligand
from evoliez.config import MDConfig
from evoliez.adapters.openmm_engine import _run_real, _pdb_one_letter_seq

PROT = Path(sys.argv[1])
WORK = Path("/tmp/verify_s10_md")
WORK.mkdir(parents=True, exist_ok=True)
# argv[2] overrides the ligand SMILES. Default ethanol: non-aromatic (no
# AssignBondOrdersFromTemplate symmetry ambiguity) + an O for the H-bond proxy.
LIG_SMILES = sys.argv[2] if len(sys.argv) > 2 else "CCO"


def _xyz(line):
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


# --- 1. protein heavy-atom ATOM records only (drop HETATM / CONECT / TER) ----
prot = [l for l in PROT.read_text().splitlines() if l.startswith("ATOM")]
if not prot:
    sys.exit(f"FAIL: no ATOM records in {PROT}")
pcoords = np.array([_xyz(l) for l in prot])
centroid = pcoords.mean(0)

# --- 2. anchor = the CA with the MOST CA neighbours within 8A, so the 8A
#         pocket (pkt_idx) is populated and contact_occupancy is exercised -----
ca_xyz = np.array([_xyz(l) for l in prot if l[12:16].strip() == "CA"])
_d2 = ((ca_xyz[:, None, :] - ca_xyz[None, :, :]) ** 2).sum(-1)
anchor = ca_xyz[int(np.argmax((_d2 <= 8.0 ** 2).sum(1)))]
outv = anchor - centroid
outv = outv / (np.linalg.norm(outv) + 1e-9)

# --- 3. RDKit-embed ligand, centre it, then step OUTWARD from the anchor CA
#         only until the hard clash clears (stays close to the CA cluster) -----
m = Chem.AddHs(Chem.MolFromSmiles(LIG_SMILES))
AllChem.EmbedMolecule(m, randomSeed=7)
AllChem.MMFFOptimizeMolecule(m)
m = Chem.RemoveHs(m)                              # heavy atoms only in the PDB
conf = m.GetConformer()
lpos = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
lpos -= lpos.mean(0)                              # ligand centroid at origin
center = anchor.copy()
for _ in range(60):
    dmin = np.sqrt((((lpos + center)[:, None, :] - pcoords[None, :, :]) ** 2)
                   .sum(-1)).min()
    if dmin >= 2.6:                               # clear the hard clash, stay close
        break
    center = center + outv * 0.3
placed = lpos + center
dmin = float(np.sqrt(((placed[:, None, :] - pcoords[None, :, :]) ** 2)
                     .sum(-1)).min())
for i in range(m.GetNumAtoms()):
    conf.SetAtomPosition(i, Point3D(*placed[i]))

# --- 4. ligand -> HETATM + CONECT lines (resName LIG, serials offset) --------
maxser = max(int(l[6:11]) for l in prot)
lig_lines = []
for ln in Chem.MolToPDBBlock(m, flavor=0).splitlines():
    if ln.startswith(("HETATM", "ATOM")):
        ln = "HETATM" + ln[6:17] + "LIG" + ln[20:]        # force HETATM + LIG
        ser = int(ln[6:11]) + maxser
        ln = ln[:6] + f"{ser:5d}" + ln[11:]
        lig_lines.append(ln)
    elif ln.startswith("CONECT"):
        body = ln[6:].rstrip()
        nums = [int(body[i:i + 5]) + maxser
                for i in range(0, len(body), 5) if body[i:i + 5].strip()]
        lig_lines.append("CONECT" + "".join(f"{n:5d}" for n in nums))

if not any(l.startswith("CONECT") for l in lig_lines):
    sys.exit("FAIL: RDKit wrote no CONECT for the ligand (reader needs them)")

complex_pdb = WORK / "complex.pdb"
complex_pdb.write_text("\n".join(prot + lig_lines + ["END"]) + "\n")
print(f"[setup] protein_atoms={len(prot)}  ligand_heavy={m.GetNumAtoms()}  "
      f"ligand_min_dist_to_protein={dmin:.2f}A  pdb={complex_pdb}")

# --- 5. catalytic positions = 3 residues with CA nearest the ligand centre ---
cas = [(int(l[22:26]), _xyz(l)) for l in prot if l[12:16].strip() == "CA"]
cas.sort(key=lambda rc: float(((rc[1] - center) ** 2).sum()))
cat_positions = sorted({rc[0] for rc in cas[:3]})
print(f"[setup] catalytic_positions (nearest CAs) = {cat_positions}")

# --- 6. cx; sequence MUST equal the PDB seq so the mutant guard passes -------
seq = _pdb_one_letter_seq(complex_pdb)
cx = Complex(structure=ProteinStructure(sequence=seq, pdb_path=str(complex_pdb)),
             ligand=Ligand(id="LIG", smiles=LIG_SMILES))
cache = WORK / "_ligcache"


def run(solvent):
    cfg = MDConfig(enabled=True, engine="openmm", protocol_level=1,
                   solvent=solvent, minimize_steps=300, equilibration_ps=1.0,
                   restrained_md_ps=1.0, production_ns=0.001, replicas=1)
    try:
        return _run_real(cx, f"verify_{solvent}", cfg, WORK,
                         catalytic_positions=cat_positions, dry_run=False,
                         ligand_cache_dir=cache)
    except Exception as exc:                       # show the real cause
        import traceback
        traceback.print_exc()
        return None


print("\n=== RUN 1: solvent=implicit ===")
r1 = run("implicit")
if r1 is not None:
    print("status            :", r1.status)
    print("solvent_mode      :", r1.solvent_mode)
    print("key_distances     :", {k: v[:3] for k, v in r1.key_distances.items()})
    print("  n_cat_series    :", len(r1.key_distances))
    print("contact_occupancy :", dict(list(r1.contact_occupancy.items())[:6]),
          f"... (n={len(r1.contact_occupancy)})")
    print("hbond_occupancy   :", r1.hbond_occupancy)
    print("ligand_rmsd[:3]   :", r1.ligand_rmsd_series[:3])
    print("failure_reason    :", r1.failure_reason)

print("\n=== RUN 2: solvent=explicit (must WARN + report implicit) ===")
r2 = run("explicit")
if r2 is not None:
    print("status            :", r2.status)
    print("solvent_mode      :", r2.solvent_mode, "(expected: implicit)")

# --- 7. verdict --------------------------------------------------------------
fails = []
if r1 is None:
    fails.append("RUN1 raised (see traceback)")
elif r1.status not in ("ok", "unstable"):
    fails.append(f"RUN1 did not run: status={r1.status} reason={r1.failure_reason}")
else:
    if not r1.key_distances:
        fails.append("key_distances EMPTY (#2 regressed)")
    if not r1.contact_occupancy:
        fails.append("contact_occupancy EMPTY (#2 regressed)")
    if not (0.0 <= float(r1.hbond_occupancy) <= 1.0):
        fails.append("hbond_occupancy out of [0,1]")
    alld = [d for s in r1.key_distances.values() for d in s]
    if alld and not (1.0 <= min(alld) <= 40.0):
        fails.append(f"catalytic distance implausible: min={min(alld)}A")
if r2 is None:
    fails.append("RUN2 raised (see traceback)")
elif r2.solvent_mode != "implicit":
    fails.append(f"#1 honesty FAILED: explicit reported as {r2.solvent_mode}")

print("\n========================================")
if fails:
    print("VERIFY: FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("VERIFY: PASS - real OpenMM run populated geometry (#2) + solvent "
      "honesty holds (#1)")
