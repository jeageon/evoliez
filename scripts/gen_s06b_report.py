"""Generate the s06b interaction-model HTML report for a finished run, from its
persisted artifacts (standalone — does not need the in-memory pipeline)."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from evoliez.config import load_config
from evoliez.io.interaction_model_report import (
    compute_interaction_stats, find_wt_complex_pdb, write_interaction_report)

RD = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track")
CFG = sys.argv[2] if len(sys.argv) > 2 else "configs/target.local.5track.yaml"

cfg = load_config(CFG)
meta = json.load(open(RD / "_state.json"))["meta"]["interaction_model"]
artifacts = json.load(open(RD / "interaction_graphs/s06b_artifacts.json"))
model = json.load(open(RD / "interaction_graphs/interaction_model.json"))
wt_pdb = find_wt_complex_pdb(RD / "complexes")
print("wt_pdb:", wt_pdb)

stats = compute_interaction_stats(meta, artifacts, model, wt_pdb)
h = cfg.interaction_model
conditions = [
    ("representatives", "%d (auto, by MSA cluster coverage)" % meta["representatives"]),
    ("Boltz samples / representative", str(h.poses_per_homolog)),
    ("rep MSA source", h.rep_msa),
    ("ensemble poses", str(meta["poses_total"])),
    ("interaction model", meta["model_kind"]),
    ("validation", "subfamily-holdout AUROC"),
    ("evoliez version", "0.1.0"),
]
p = RD / "reports" / "interaction_model_report.html"
write_interaction_report(p, target_id=cfg.input.target_id, stats=stats,
                         conditions=conditions,
                         generated=datetime.now().strftime("%Y-%m-%d %H:%M"))
html = open(p).read()
print("WROTE %d bytes | leftover-tokens=%s | 3Dmol=%s | charts=%s"
      % (os.path.getsize(p), "%%" in html, "3Dmol" in html, "Chart" in html))
