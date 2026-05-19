#!/usr/bin/env python
"""Validate the REAL OpenMM + GAFF MD-lite path on an already-produced
full-atom Boltz structure - WITHOUT re-running the ~20-min Boltz step.

The `md` smoke only MDs mutant candidates, and we have no per-mutant
predicted structures yet, so every mutant honestly becomes
skipped_no_mutant_structure. This drives run_md(real) on the WT Boltz
complex (the #2-passing path: PDB sequence == intended sequence), which
actually exercises OpenMM + antechamber/parmchk2 GAFF params + the
CONECT'd ligand. ~1-2 min, no GPU-Boltz.

  python scripts/check_real_md.py <real_boltz_model.pdb> [config.yaml]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from evoliez.adapters.boltz import _parse_real_structure
from evoliez.adapters.openmm_engine import run_md
from evoliez.config import Backend, load_config
from evoliez.features.ligand import parse_ligand
from evoliez.md.analysis import analyse


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    pdb = Path(sys.argv[1])
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "configs/smoke.yaml"
    if not pdb.exists():
        print(f"no such PDB: {pdb}")
        return 2

    cfg = load_config(cfg_path)
    ic = cfg.input
    seq = (ic.target_sequence or "").strip().upper()
    if not seq and ic.target_fasta and Path(ic.target_fasta).exists():
        seq = "".join(
            l.strip() for l in Path(ic.target_fasta).read_text().splitlines()
            if l and not l.startswith(">")
        ).upper()
    ligand = parse_ligand(ic.ligand)

    # Build the WT complex straight from the real Boltz output, then point
    # its structure at that same full-atom PDB so the MD adapter uses it.
    cx = _parse_real_structure(pdb, seq, ligand)
    cx.structure.pdb_path = str(pdb)

    work = Path(tempfile.mkdtemp(prefix="evz_mdwt_"))
    print(f">> real OpenMM MD-lite on WT: {pdb}\n>> workdir: {work}")
    res = run_md(
        cx, "wt_ref", cfg.validation.md, work,
        instability=0.0, catalytic_positions=[], backend=Backend.real,
    )
    metrics = analyse(res, cfg.scoring)
    print(">> status     :", res.status)
    print(">> failure    :", res.failure_reason or "(none)")
    print(">> md passed  :", metrics.passed)
    print(">> analysis   :", json.dumps(
        {k: getattr(metrics, k) for k in (
            "pocket_rmsd_mean", "ligand_rmsd_mean", "energy_drift",
            "catalytic_distance_mean", "md_lite_score", "passed")},
        indent=2))
    ok = res.status not in ("failed",) and not str(
        res.status).startswith("skipped_no")
    print(">> REAL OpenMM MD path:",
          "VALIDATED" if ok else f"NOT validated ({res.status})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
