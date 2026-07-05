#!/usr/bin/env python
"""E3 — Mg-retention diagnostic (reviewer Day 2). Reads the focused-run trajectories and separates
WHY the co-substrate diffused: Mg-bridge collapse (metal/FF problem) vs substrate pose diffusion
(anchoring/solvent problem) vs whole-cluster relaxation (needs explicit solvent / active-state anchor).

Per system (WT + each candidate) it traces, over the saved frames:
  d(Mg .. O_nuc)      Mg to the 3-HP carboxylate nucleophile O   (nac.atoms.donor_heavy)
  d(Mg .. O_leaving)  Mg to the ATP phosphate leaving O          (nac.atoms.leaving)
  d(O_nuc .. Palpha)  the in-line attack distance                (donor_heavy .. acceptor)
and derives Mg-bridge occupancy (Mg within COORD of BOTH anions) + the frame each event first fails,
so the diffusion mode is classified deterministically.

Read-only, CPU-only (mdtraj). Run on the server where the trajectories + mdtraj live:
  PYTHONPATH=src python scripts/e3_mg_retention.py runs/srcar_3hp_v5
Writes docs/car_v5/e3_mg_retention_table.csv + prints a summary + a mode classification.
"""
import csv
import glob
import json
import os
import sys

COORD = 0.28   # nm; Mg--O coordination cutoff (~2.0-2.2 A ideal; 2.8 A generous for "still bridging")
ESCAPE = 0.40  # nm; O_nuc--Palpha beyond 4.0 A = left the near-attack window


def _mut_map(run):
    m = {}
    p = os.path.join(run, "reports/provenance/generated_candidates.csv")
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                m[r["candidate_id"]] = r["mutation_string"]
    return m


def _first_false(mask):
    """frame index where a per-frame boolean first becomes False, or -1 if always True."""
    for i, v in enumerate(mask):
        if not v:
            return i
    return -1


def analyze(md_dir):
    import mdtraj as md
    import numpy as np
    cid = os.path.basename(md_dir.rstrip("/"))
    apath = os.path.join(md_dir, "analysis.json")
    if not os.path.exists(apath):
        return None
    aj = json.load(open(apath))
    atoms = ((aj.get("nac") or {}).get("atoms")) or {}
    donor, acc, leav = atoms.get("donor_heavy"), atoms.get("acceptor"), atoms.get("leaving")
    if donor is None or acc is None or leav is None:
        return {"cid": cid, "error": "no nac.atoms indices"}
    pdb = glob.glob(os.path.join(md_dir, "*_minimized.pdb"))
    dcd = glob.glob(os.path.join(md_dir, "*.dcd"))
    if not pdb or not dcd:
        return {"cid": cid, "error": "missing pdb/dcd"}
    t = md.load(dcd[0], top=pdb[0])
    mg = t.topology.select("resname MG")
    if len(mg) == 0:
        return {"cid": cid, "error": "no MG in topology", "n_frames": t.n_frames}
    mg = int(mg[0])
    d_mg_nuc = md.compute_distances(t, [[mg, donor]])[:, 0]
    d_mg_leav = md.compute_distances(t, [[mg, leav]])[:, 0]
    d_nuc_pa = md.compute_distances(t, [[donor, acc]])[:, 0]
    bridged = (d_mg_nuc <= COORD) & (d_mg_leav <= COORD)
    in_window = d_nuc_pa <= ESCAPE
    n = t.n_frames
    return {
        "cid": cid, "n_frames": n,
        "mg_bridge_occ": float(bridged.mean()),
        "mg_nuc_occ": float((d_mg_nuc <= COORD).mean()),
        "mg_leav_occ": float((d_mg_leav <= COORD).mean()),
        "attack_window_occ": float(in_window.mean()),
        "first_bridge_loss": _first_false(bridged),
        "first_escape": _first_false(in_window),
        "d_nuc_pa_init": float(d_nuc_pa[0] * 10), "d_nuc_pa_final": float(d_nuc_pa[-1] * 10),
        "d_mg_nuc_final": float(d_mg_nuc[-1] * 10), "d_mg_leav_final": float(d_mg_leav[-1] * 10),
    }


def classify(r):
    """Deterministic diffusion-mode label from the traces."""
    if r.get("error"):
        return r["error"]
    bl, esc = r["first_bridge_loss"], r["first_escape"]
    if r["mg_bridge_occ"] >= 0.8 and r["attack_window_occ"] >= 0.8:
        return "held (productive retained)"
    if bl >= 0 and (esc < 0 or bl < esc):
        return "mg_bridge_lost_first (metal/FF retention)"
    if esc >= 0 and (bl < 0 or esc <= bl):
        return "substrate_left_first (pose/solvent anchoring)"
    return "cluster_relaxed (needs explicit solvent / active-state anchor)"


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/srcar_3hp_v5"
    mm = _mut_map(run)
    dirs = sorted(glob.glob(os.path.join(run, "md", "*")))
    dirs = [d for d in dirs if os.path.isdir(d) and not d.endswith("_ligand_ff_cache")]
    rows = []
    for d in dirs:
        r = analyze(d)
        if r is None:
            continue
        cid = r["cid"]
        r["mutation"] = "WT" if cid == "_wt_reference" else mm.get(cid, cid)
        r["mode"] = classify(r)
        rows.append(r)
    rows.sort(key=lambda r: (r["mutation"] != "WT", -(r.get("mg_bridge_occ") or 0)))
    cols = ["mutation", "mg_bridge_occ", "mg_nuc_occ", "mg_leav_occ", "attack_window_occ",
            "first_bridge_loss", "first_escape", "d_nuc_pa_init", "d_nuc_pa_final", "n_frames", "mode"]
    out = os.path.join("docs/car_v5/e3_mg_retention_table.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("%-24s %7s %7s %7s %8s %6s %6s  %s" % (
        "mutation", "brdgOcc", "mgNucO", "mgLeaO", "winOcc", "bLoss", "esc", "mode"))
    print("-" * 100)
    for r in rows:
        if r.get("error"):
            print("%-24s  ERROR: %s" % (r["mutation"][:24], r["error"]))
            continue
        print("%-24s %7.2f %7.2f %7.2f %8.2f %6d %6d  %s" % (
            r["mutation"][:24], r["mg_bridge_occ"], r["mg_nuc_occ"], r["mg_leav_occ"],
            r["attack_window_occ"], r["first_bridge_loss"], r["first_escape"], r["mode"]))
    from collections import Counter
    modes = Counter(r.get("mode", "err") for r in rows)
    print("\nmode counts:", dict(modes))
    print("wrote", out)


if __name__ == "__main__":
    main()
