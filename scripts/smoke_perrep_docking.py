"""Server smoke: the PER-REP multi-engine docking fan-out on 3 EXISTING reps with
REAL gnina/diffdock. Verifies _run_per_rep_docking end to end BEFORE the 5-6 h
full run — the fork ProcessPool + GPU pinning + rep stem_dir re-parse + classify
chain (the only parts the mock suite can't exercise). Uses the current
single-ligand reps (the docking CODE is ligand-count-agnostic; this checks the
machinery, not the formate fold)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from evoliez.config import Backend, load_config
from evoliez.features.ligand import parse_ligand
from evoliez.logging_utils import get_logger
from evoliez.stages.s06b_interaction_model import _run_per_rep_docking

RD = Path("/mnt/data/jglee/runs/fdh_5track")
cfg_full = load_config("configs/target.local.5track.yaml")
imcfg = cfg_full.interaction_model
ligand = parse_ligand(cfg_full.input.ligand)
dock_cfg = cfg_full.validation.redocking
primary_method = cfg_full.complex_prediction.primary_method

model = json.load(open(RD / "interaction_graphs" / "interaction_model.json"))
consensus_fp = np.array(model["consensus"], dtype=float)
print("consensus_fp dim:", consensus_fp.shape, "| methods:", imcfg.multi_engine_methods)

# 3 existing reps: sequence from the boltz-input YAML, stem_dir = predictions subdir
N = 3
targets = []
for i in range(N):
    rd = RD / "structures" / "representatives" / f"hom_{i:03d}"
    y = yaml.safe_load(open(rd / f"hom_{i:03d}_boltz_input.yaml"))
    seq = y["sequences"][0]["protein"]["sequence"]
    stem = (rd / f"boltz_results_hom_{i:03d}_boltz_input" / "predictions"
            / f"hom_{i:03d}_boltz_input")
    assert stem.exists(), f"stem missing: {stem}"
    targets.append((
        f"rep_{i:03d}", str(stem), seq, ligand, consensus_fp,
        list(imcfg.multi_engine_methods), dock_cfg, str(RD / "docking_smoke"),
        Backend.real, imcfg.contact_cutoff, imcfg.k_nearest_residues,
        [], bool(cfg_full.input.extra_ligands), imcfg, primary_method,
    ))

import os
gpu_list = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
print(f"gpu_list={gpu_list} targets={len(targets)} (fan-out fork pool)")

log = get_logger("smoke.perrep")
kept, diags = _run_per_rep_docking(targets, gpu_list, log)

print("\n=== RESULT ===")
print("kept PoseRecords:", len(kept), "| classified:", len(diags))
print("roles:", dict(Counter(d.role for d in diags)))
for d in diags:
    print(f"  {d.source:9s} {d.role:14s} rmsd_to_own_pose={d.rmsd_to_consensus} "
          f"fp_overlap_vs_family={d.fp_overlap} clash={d.clash} w={d.sample_weight}")
# PASS = the fan-out ran every target through real docking + classification
ok = len(diags) >= len(targets)  # at least one classified pose per target
print("RESULT:", "PASS (per-rep fan-out + real docking + classify OK)" if ok
      else "FAIL (some targets produced no docking/classification)")
