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

# Multi-engine docking audit (per-rep multi-engine augmentation). Optional:
# absent when the run had multi_engine off -> that report section is skipped.
me_path = RD / "interaction_graphs" / "multi_engine_docking.json"
me_audit = json.load(open(me_path)) if me_path.exists() else None
print("multi_engine audit:",
      ("%d rows" % len(me_audit.get("rows", []))) if me_audit else "absent")

stats = compute_interaction_stats(meta, artifacts, model, wt_pdb,
                                  multi_engine_audit=me_audit,
                                  run_dir=RD)
# Validation label: prefer the canonical full leave-one-subfamily-out CV (read
# from reports/cv_subfamily_auroc.json by compute_interaction_stats); fall back to
# the single subfamily-holdout when no CV json is present.
_cvm = stats["meta"]
if _cvm.get("cv_auroc_mean") is not None:
    validation_label = (
        "%d-fold leave-one-subfamily-out CV AUROC %.4f ± %.4f (single holdout "
        "%.4f as one example fold)"
        % (_cvm.get("cv_n_folds") or 0, _cvm["cv_auroc_mean"],
           _cvm.get("cv_auroc_std") or 0.0,
           _cvm.get("subfamily_holdout_auroc") or 0.0))
else:
    validation_label = "subfamily-holdout AUROC"
h = cfg.interaction_model
conditions = [
    ("representatives", "%d (auto, by MSA cluster coverage)" % meta["representatives"]),
    ("Boltz samples / representative", str(h.poses_per_homolog)),
    ("rep MSA source", h.rep_msa),
    ("ensemble poses", str(meta["poses_total"])),
    ("interaction model", meta["model_kind"]),
    ("validation", validation_label),
    ("evoliez version", "0.1.0"),
]
# Provenance stamp: standalone reconstruction (cfg + run dir only, no live
# RunContext). Best-effort per field; also drops RUN_INFO.json next to the run.
from types import SimpleNamespace

from evoliez.io._provenance import provenance_fields, provenance_html, write_run_info

_pfields = provenance_fields(SimpleNamespace(config=cfg, root=RD))
write_run_info(RD, _pfields)

p = RD / "reports" / "interaction_model_report.html"
write_interaction_report(p, target_id=cfg.input.target_id, stats=stats,
                         conditions=conditions,
                         generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                         provenance=provenance_html(_pfields))
html = open(p).read()
print("WROTE %d bytes | leftover-tokens=%s | 3Dmol=%s | charts=%s"
      % (os.path.getsize(p), "%%" in html, "3Dmol" in html, "Chart" in html))
