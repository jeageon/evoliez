#!/usr/bin/env python
"""E2 accommodation readout (reviewer Day 4) — score each system by how well it OCCUPIES the
crystal-grounded productive window during the E1 explicit-solvent trajectory. The window is defined
by the 5MST active site (O_nuc->Palpha ~2.8-3.0 A, in-line) and the mechanism config (distance_max
3.6 A, angle_min 150 deg): productive_occupancy = fraction of frames with distance <= 3.6 A AND
angle >= 150 deg. This turns the single-pose E1 metric (which gave discriminates=false) into an
ensemble-occupancy comparison against a REAL active-state target.

Read-only, CPU (mdtraj), reuses the existing E1 dcds — no new MD. Run on the server:
  PYTHONPATH=src python scripts/e2_accommodation.py runs/srcar_3hp_v5_e1
"""
import csv
import glob
import json
import os
import sys

DIST_MAX = 3.6    # A  productive reactive window (mechanism config)
ANGLE_MIN = 150.0  # deg in-line


def _mutmap(run):
    m = {}
    p = os.path.join(run, "reports/provenance/generated_candidates.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            m[r["candidate_id"]] = r["mutation_string"]
    return m


def score(md_dir):
    import mdtraj as md
    import numpy as np
    cid = os.path.basename(md_dir.rstrip("/"))
    aj = os.path.join(md_dir, "analysis.json")
    if not os.path.exists(aj):
        return None
    atoms = ((json.load(open(aj)).get("nac") or {}).get("atoms")) or {}
    donor, acc, leav = atoms.get("donor_heavy"), atoms.get("acceptor"), atoms.get("leaving")
    if donor is None or acc is None:
        return {"cid": cid, "error": "no nac atoms"}
    pdb = glob.glob(os.path.join(md_dir, "*_minimized.pdb"))
    dcd = glob.glob(os.path.join(md_dir, "*.dcd"))
    if not pdb or not dcd:
        return {"cid": cid, "error": "no pdb/dcd"}
    t = md.load(dcd[0], top=pdb[0])
    d = md.compute_distances(t, [[donor, acc]])[:, 0] * 10.0   # nm->A
    ang = None
    if leav is not None:
        ang = np.degrees(md.compute_angles(t, [[donor, acc, leav]])[:, 0])
    n = t.n_frames
    in_dist = d <= DIST_MAX
    in_ang = (ang >= ANGLE_MIN) if ang is not None else np.ones(n, bool)
    productive = in_dist & in_ang
    return {
        "cid": cid, "n_frames": n,
        "productive_occ": float(productive.mean()),
        "dist_window_occ": float(in_dist.mean()),
        "angle_window_occ": float(in_ang.mean()),
        "d_min": float(d.min()), "d_median": float(np.median(d)),
        "angle_median": float(np.median(ang)) if ang is not None else None,
    }


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "runs/srcar_3hp_v5_e1"
    mm = _mutmap(run)
    rows = []
    for dd in sorted(glob.glob(os.path.join(run, "md", "*"))):
        if not os.path.isdir(dd) or dd.endswith("_ligand_ff_cache"):
            continue
        r = score(dd)
        if not r:
            continue
        r["mutation"] = "WT" if r["cid"] == "_wt_reference" else mm.get(r["cid"], r["cid"])
        rows.append(r)
    rows.sort(key=lambda r: (r["mutation"] != "WT", -(r.get("productive_occ") or 0)))
    out = "docs/car_v5/e2_accommodation_table.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["mutation", "productive_occ", "dist_window_occ", "angle_window_occ",
            "d_min", "d_median", "angle_median", "n_frames"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        [w.writerow(r) for r in rows if not r.get("error")]
    print("productive window: O->Palpha <= %.1f A AND angle >= %.0f deg (crystal-grounded, 5MST ~2.8-3.0A)" % (
        DIST_MAX, ANGLE_MIN))
    print("%-24s %8s %8s %8s %7s %7s" % (
        "system", "prodOcc", "distOcc", "angOcc", "dMin", "dMed"))
    print("-" * 70)
    for r in rows:
        if r.get("error"):
            print("%-24s  ERROR %s" % (r["mutation"][:24], r["error"])); continue
        print("%-24s %8.2f %8.2f %8.2f %7.2f %7.2f" % (
            r["mutation"][:24], r["productive_occ"], r["dist_window_occ"],
            r["angle_window_occ"], r["d_min"], r["d_median"]))
    wt = next((r for r in rows if r["mutation"] == "WT" and not r.get("error")), None)
    if wt:
        better = [r["mutation"] for r in rows if not r.get("error") and r["mutation"] != "WT"
                  and (r.get("productive_occ") or 0) > (wt.get("productive_occ") or 0)]
        print("\nWT productive_occ=%.2f | candidates ABOVE WT: %s" % (
            wt["productive_occ"], better or "none"))
    print("wrote", out)


if __name__ == "__main__":
    main()
