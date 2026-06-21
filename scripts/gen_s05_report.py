"""Regenerate the s05 reference-docking HTML report for a FINISHED run, from its
persisted artifacts — WITHOUT re-running s05 (no GINA/DiffDock, no GPU).

Mirrors the report-build block of ``DockingStage._write_report``
(``src/evoliez/stages/s05_docking.py``): it calls ``compute_docking_stats`` +
``write_docking_report`` exactly as the stage does, but reconstructs every input
from ``RUN_DIR`` instead of the in-memory pipeline context.

Everything the report needs is on disk and reconstructable standalone:

* DB scores / candidate_ids  -> ``<RUN_DIR>/evoliez.sqlite`` ``docking_pose``
  (``compute_docking_stats`` discovers every candidate here, including the
  multi-ligand context variants ``'<base>__<cofactor>'``).
* docked poses / receptor / reference ligand PDB
                              -> ``<RUN_DIR>/docking/{gnina,diffdock}/...``
  (``compute_docking_stats`` reads ``<cand>_ref_lig.pdb`` directly off disk, so
  the s04 reference pose/atoms come from the artifacts, NOT from ctx).
* methods / target_id / ligand id / ligand SMILES / backend  -> the config.

The config is needed for the SMILES template (ligand normalization), the method
list, the target/ligand labels, and the s05 backend shown in the conditions
table. Pass it explicitly, or drop a copy at ``<RUN_DIR>/config.yaml`` and it is
picked up automatically. NOTHING about the docking itself is re-run.

Usage:
    python scripts/gen_s05_report.py [RUN_DIR] [CONFIG]

    RUN_DIR  default /mnt/data/jglee/runs/fdh_5track
    CONFIG   default <RUN_DIR>/config.yaml if present,
             else configs/target.local.5track.yaml
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from evoliez.config import load_config
from evoliez.io.docking_report import compute_docking_stats, write_docking_report
from evoliez.io.paths import ProjectPaths

RD = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track")

# Config: explicit arg > a config dropped in the run dir > the 5-track default.
if len(sys.argv) > 2:
    CFG = sys.argv[2]
elif (RD / "config.yaml").exists():
    CFG = str(RD / "config.yaml")
else:
    CFG = "configs/target.local.5track.yaml"

cfg = load_config(CFG)
paths = ProjectPaths(RD)

db_path = paths.db_path
if not db_path.exists():
    sys.exit("ERROR: %s not found — is RUN_DIR a finished run dir?" % db_path)
if not paths.docking.exists():
    sys.exit("ERROR: %s not found — no s05 docking artifacts to report on."
             % paths.docking)

red = cfg.validation.redocking
li = cfg.input.ligand
ligand_smiles = li.value if li.type == "smiles" else None

# Same call the stage makes; candidate_ids (base + '<base>__<cofactor>' variants)
# are auto-discovered from docking_pose, so the multi-ligand section appears
# whenever the run docked the design ligand with co-modelled cofactor context.
stats = compute_docking_stats(
    str(paths.root), str(db_path), red.methods, ligand_smiles=ligand_smiles)

if not stats["scores"]:
    sys.exit("ERROR: no docking_pose rows for the primary candidate %r in %s — "
             "nothing to report (did s05 run?)." % (stats["candidate"], db_path))

cand_ids = [c["candidate"] for c in stats.get("candidate_summaries", [])]
ml = stats.get("multi_ligand")
print("candidates:", ", ".join(cand_ids) or "(primary only)")
print("multi-ligand:",
      ("per-ligand reference docking, %d row(s), ligands: %s"
       % (len(ml.get("rows", [])), ", ".join(map(str, ml.get("ligands", []))))) if ml else "none")

# Provenance stamp: this script reconstructs only cfg + the run dir (no live
# RunContext), so hand provenance_fields a minimal shim exposing .config + .root.
# Every field is best-effort, so a partial reconstruction still renders + writes
# RUN_INFO.json without crashing.
from types import SimpleNamespace

from evoliez.io._provenance import provenance_fields, provenance_html, write_run_info

_pfields = provenance_fields(SimpleNamespace(config=cfg, root=RD))
write_run_info(RD, _pfields)
_prov = provenance_html(_pfields)

out = paths.reports / "docking_report.html"
write_docking_report(
    out,
    target_id=cfg.input.target_id, stats=stats,
    ligand_name=li.id,
    generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    provenance=_prov,
    conditions=[
        ("methods", ", ".join(red.methods)),
        ("poses / method", red.poses_per_candidate),
        ("reference", "s04 Boltz WT pose"),
        ("backend", cfg.backend_for("s05_docking").value),
    ],
)
html = out.read_text()
print("WROTE %d bytes -> %s | leftover-tokens=%s | 3Dmol=%s | charts=%s | "
      "multi-ligand-section=%s"
      % (os.path.getsize(out), out, "%%" in html, "3Dmol" in html,
         "Chart" in html, "Per-ligand reference docking" in html))
