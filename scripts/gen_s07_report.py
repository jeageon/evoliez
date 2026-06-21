"""Regenerate the s07 mutation-library HTML report for a FINISHED run, from its
persisted artifacts — WITHOUT re-running s07 (no LigandMPNN, no GPU).

Mirrors what the stage would render, but reconstructs every input from
``RUN_DIR`` instead of the in-memory pipeline context. Everything the report
needs is on disk and reconstructable standalone:

* the generated library + budget tiers
      -> ``<RUN_DIR>/reports/provenance/generated_candidates{,_single,
         _multipoint,_risky}.json`` (the FULL per-candidate provenance the s07
         stage writes: generator, n_mutations, the rationale features, the
         structural distances + safety flags, and the budget tiers).
* the design context (designable / catalytic positions, the residue↔ligand
  contact list, the per-position MSA conservation / occupancy)
      -> ``<RUN_DIR>/interaction_graphs/graph_features.json`` (s06 writes the
         contacts + per-residue feature table + designable_positions here),
         with ``catalytic_positions`` / ``designable_positions`` cross-checked
         against ``<RUN_DIR>/_state.json`` ``meta``.
* the WT complex PDB (3D viewer)
      -> the Boltz predictions tree under ``<RUN_DIR>/complexes`` (same locator
         as s04 / s06b).
* target id + ligand ids + the s07 backend / config knobs  -> the config.

NOTHING about generation is re-run. A missing artifact degrades gracefully (the
relevant section explains the omission) so the script never crashes on a partial
run dir.

Usage:
    python scripts/gen_s07_report.py [RUN_DIR] [CONFIG]

    RUN_DIR  default /mnt/data/jglee/runs/fdh_5track
    CONFIG   default <RUN_DIR>/config.yaml if present,
             else configs/target.local.5track.yaml
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from evoliez.config import load_config
from evoliez.io.s07_report import (
    compute_mutation_stats, find_wt_complex_pdb, load_generated_provenance,
    write_mutation_report)

RD = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track")

# Config: explicit arg > a config dropped in the run dir > the 5-track default.
if len(sys.argv) > 2:
    CFG = sys.argv[2]
elif (RD / "config.yaml").exists():
    CFG = str(RD / "config.yaml")
else:
    CFG = "configs/target.local.5track.yaml"

cfg = load_config(CFG)

# ---- s07 provenance (the generated library + budget tiers) ----------------- #
prov = load_generated_provenance(RD)
if not prov["generated"]:
    sys.exit(
        "ERROR: %s/reports/provenance/generated_candidates.json not found or "
        "empty — did s07 run for this RUN_DIR?" % RD)
print("candidates:", len(prov["generated"]),
      "| tiers:", {t: len(r) for t, r in prov["tiers"].items()} or "none")

# ---- design context (positions + contacts + per-position MSA features) ----- #
# s06 persists everything we need to graph_features.json; cross-check the
# position sets against _state.json meta (the authoritative persisted copy).
gf_path = RD / "interaction_graphs" / "graph_features.json"
graph_features = json.loads(gf_path.read_text()) if gf_path.exists() else {}

meta = {}
sp = RD / "_state.json"
if sp.exists():
    try:
        meta = json.loads(sp.read_text()).get("meta", {}) or {}
    except (json.JSONDecodeError, OSError):
        meta = {}

designable = (meta.get("designable_positions")
              or graph_features.get("designable_positions") or [])
catalytic = meta.get("catalytic_positions", []) or []
# fixed = the s01 fixed set (locked + over-conserved); the report subtracts the
# catalytic overlap so the two classes don't double-count. The standalone copy
# lives in meta when present; else fall back to the catalytic set only.
fixed = meta.get("fixed_positions", []) or list(catalytic)

contacts = graph_features.get("contacts", []) or []

# Per-position MSA features for the design-space map (conservation / occupancy):
# rebuild lightweight shims from the s06 per-residue feature table, keyed the way
# the report expects (target_position / conservation_score / gap_frequency / wt).
position_features = []
for r in (graph_features.get("residues", []) or []):
    position_features.append(SimpleNamespace(
        target_position=r.get("residue_index"),
        conservation_score=r.get("conservation"),
        gap_frequency=r.get("gap_frequency"),
        wt=r.get("aa", ""),
    ))

# ---- WT complex PDB (3D viewer) ------------------------------------------- #
wt_pdb = find_wt_complex_pdb(RD / "complexes")
print("wt_pdb:", wt_pdb or "absent (3D view will be omitted)")

# ---- ligand ids (generic, from the config) -------------------------------- #
ci = cfg.input
ligand_ids = []
if getattr(ci, "ligand", None) is not None:
    ligand_ids.append(getattr(ci.ligand, "id", "ligand"))
for el in (getattr(ci, "extra_ligands", None) or []):
    ligand_ids.append(getattr(el, "id", "ligand"))

stats = compute_mutation_stats(
    prov["generated"],
    tier_rows=prov["tiers"],
    designable_positions=designable,
    catalytic_positions=catalytic,
    fixed_positions=fixed,
    contacts=contacts,
    position_features=position_features,
    wt_pdb_path=wt_pdb,
    ligand_ids=ligand_ids,
)

mg = cfg.mutation_generation
conditions = [
    ("generators", ", ".join(mg.methods)),
    ("max candidates", str(mg.max_candidates)),
    ("multipoint", "order ≤ %d" % mg.multipoint_order if mg.multipoint else "off"),
    ("budget tiers (single/multi/risky top-N)",
     "%d / %d / %d" % (mg.tier_single_top, mg.tier_multipoint_top,
                       mg.tier_risky_top)),
    ("risk filter", "drop" if mg.risk_filter else "flag-only"),
    ("backend", cfg.backend_for("s07_mutation_gen").value),
    ("evoliez version", "0.1.0"),
]

# Provenance stamp: this script reconstructs only cfg + the run dir (no live
# RunContext), so hand provenance_fields a minimal shim exposing .config + .root.
# Every field is best-effort, so a partial reconstruction still renders + writes
# RUN_INFO.json without crashing.
from evoliez.io._provenance import provenance_fields, provenance_html, write_run_info

_pfields = provenance_fields(SimpleNamespace(config=cfg, root=RD))
write_run_info(RD, _pfields)

out = RD / "reports" / "s07_mutation_report.html"
write_mutation_report(
    out, target_id=cfg.input.target_id, stats=stats,
    conditions=conditions,
    generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    provenance=provenance_html(_pfields))
html = out.read_text()
print("WROTE %d bytes -> %s | leftover-tokens=%s | 3Dmol=%s | charts=%s | "
      "design-space=%s | provenance-table=%s"
      % (os.path.getsize(out), out, "%%" in html, "3Dmol" in html,
         "Chart" in html, "Design space" in html, "Per-mutation provenance" in html))
