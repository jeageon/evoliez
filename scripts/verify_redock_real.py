#!/usr/bin/env python
"""Anti-s06 deep verification of s09 redock poses.

The s06 silent corruption: a failed atom-relabel OVERWROTE pose coords with the
REFERENCE, so RMSD-to-reference == 0 for ~all poses = a zero-signal free pass that
looked fine on the surface. REAL paper-grade redock data must show, recomputed
FROM the stored pose coordinates via the s05-report's proven, no-superposition
symmetry_corrected_rmsd:
  - non-zero, DISTRIBUTED RMSD-to-reference (a real spread, NOT a spike at ~0),
  - gnina's selected representative mode is NOT always rank-1 (real selection),
  - poses DIFFER between candidates (not all collapsed onto the one reference),
  - the reference ligand is the full cofactor (not empty/degenerate).

Usage: verify_redock_real.py [run_dir] [sample_N]
"""
import glob
import os
import random
import statistics
import sys
from pathlib import Path

RD = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
sys.path.insert(0, "/mnt/data/jglee/EvoLiEZ/src")
from evoliez.adapters.gnina import _mode_molblocks  # production counts-line-anchored splitter
from evoliez.config import load_config
from evoliez.features.ligand import rmsd_template, symmetry_corrected_rmsd

cfg = load_config("/mnt/data/jglee/EvoLiEZ/configs/target.local.5track.yaml")
smiles = cfg.input.ligand.value
tmpl = rmsd_template(smiles)
print(f"ligand SMILES len={len(smiles)} ; template heavy atoms="
      f"{tmpl.GetNumAtoms() if tmpl else None}")
random.seed(0)


def split_sdf(text):
    return [p.strip() + "\n" for p in text.split("$$$$") if p.strip()]


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n=0 (no RMSD computed!)"
    return (f"n={len(xs)} min={min(xs):.2f} max={max(xs):.2f} "
            f"mean={statistics.mean(xs):.2f} std={statistics.pstdev(xs):.2f} "
            f"near-zero(<0.05A)={sum(1 for x in xs if x < 0.05)}")


G = Path(RD) / "validation/redock/gnina"

# --- reference sanity: full cofactor + per-candidate vs shared? ---
refs = sorted(glob.glob(str(G / "*_ref_lig.pdb")))
print(f"\n[REF] {len(refs)} gnina ref_lig.pdb files")
shared_ref = None
if refs:
    shared_ref = Path(refs[0]).read_text()
    na = sum(1 for ln in shared_ref.splitlines()
             if ln.startswith(("ATOM", "HETATM")))
    same = (len(refs) >= 2 and shared_ref == Path(refs[1]).read_text())
    print(f"[REF] atoms in {os.path.basename(refs[0])}: {na} ; "
          f"identical to 2nd ref: {same}")

# --- GNINA: recompute selected-mode RMSD from STORED pose coords ---
gn = sorted(glob.glob(str(G / "*_gnina_out.sdf")))
print(f"\n[GNINA] {len(gn)} completed candidates; sampling {min(N, len(gn))}")
sel_rmsds, sel_modes = [], []
for f in random.sample(gn, min(N, len(gn))):
    cid = os.path.basename(f).replace("_gnina_out.sdf", "")
    ref = G / f"{cid}_ref_lig.pdb"
    if not ref.exists():
        continue
    rt = ref.read_text()
    modes = _mode_molblocks(Path(f))  # blank-title-safe (the same parser s09 uses)
    rs = [(i, symmetry_corrected_rmsd(m, rt, smiles, pose_fmt="sdf", template=tmpl))
          for i, m in enumerate(modes)]
    rs = [(i, r) for i, r in rs if r is not None]
    if not rs:
        print(f"  {cid}: NO RMSD ({len(modes)} modes parsed)")
        continue
    bi, br = min(rs, key=lambda t: t[1])
    sel_rmsds.append(br)
    sel_modes.append(bi + 1)
    print(f"  {cid}: {len(rs)} modes, selected mode#{bi + 1} RMSD={br:.2f}A, "
          f"spread=[{min(r for _, r in rs):.2f}..{max(r for _, r in rs):.2f}]")
print(f"[GNINA] selected-mode RMSD: {stats(sel_rmsds)}")
if sel_modes:
    print(f"[GNINA] selected mode# set={sorted(set(sel_modes))} ; "
          f"mode#1 chosen {sel_modes.count(1)}/{len(sel_modes)}")

# --- DIFFDOCK: recompute rank1 RMSD vs the reference ---
dd = sorted(glob.glob(str(Path(RD) / "validation/redock/dd_gpu*/mut_*/rank1.sdf")))
print(f"\n[DIFFDOCK] {len(dd)} rank1 poses; sampling {min(N, len(dd))}")
dd_rmsds = []
for f in random.sample(dd, min(N, len(dd))):
    cid = Path(f).parent.name
    own = G / f"{cid}_ref_lig.pdb"
    rt = own.read_text() if own.exists() else shared_ref
    if rt is None:
        continue
    r = symmetry_corrected_rmsd(Path(f).read_text(), rt, smiles,
                                pose_fmt="sdf", template=tmpl)
    dd_rmsds.append(r)
    print(f"  {cid}: rank1 RMSD={r if r is None else round(r, 2)}A")
print(f"[DIFFDOCK] rank1 RMSD: {stats(dd_rmsds)}")

# --- VERDICT: s06-class corruption signature ---
print("\n=== s06-class corruption check (RMSD recomputed from stored coords) ===")
gz = sum(1 for x in sel_rmsds if x < 0.05)
ddv = [x for x in dd_rmsds if x is not None]
dz = sum(1 for x in ddv if x < 0.05)
print(f"gnina selected near-zero: {gz}/{len(sel_rmsds)} ; "
      f"diffdock near-zero: {dz}/{len(ddv)}")
real = (
    len(sel_rmsds) >= 5 and statistics.pstdev(sel_rmsds) > 0.2 and gz == 0
    and (not sel_modes or sel_modes.count(1) < len(sel_modes))
    and len(ddv) >= 5 and statistics.pstdev(ddv) > 0.2 and dz == 0
)
print("VERDICT:", "REAL — distributed, non-zero, varied modes; NOT s06-class"
      if real else "SUSPICIOUS — inspect (possible s06-class signature)")
