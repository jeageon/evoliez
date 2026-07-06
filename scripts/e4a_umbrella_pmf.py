#!/usr/bin/env python
"""E4a driver — O_nuc->Palpha umbrella / PMF access-barrier screen (reviewer track E4a).

WHY: E2 found unbiased explicit MD never occupies the productive ~3 A window. But the near-attack
conformation is TRANSIENT, so the right observable is the FREE-ENERGY COST to ACCESS it, not "does it
sit there". Umbrella sampling along the O_nuc->Palpha distance + WHAM gives that cost per candidate,
using the existing classical FF -- cheaper and less risky than QM/MM, and the correct next tier.

TWO PHASES:
  (1) SAMPLE (server, OpenMM): for each candidate, run one short explicit-solvent MD per umbrella
      window with a harmonic O_nuc->Palpha bias (run_md(umbrella=(window_A, k)) -> a DCD per window).
      s10 wires this when validation.md.umbrella is set; or a caller loops run_md directly.
  (2) ANALYSE (this script, CPU/mdtraj + WHAM): read the per-window DCDs, extract the O_nuc->Palpha
      distance timeseries (nac atoms from analysis.json), WHAM -> PMF -> near-attack access cost.

Layout expected under <run>/e4a/<candidate>/window_<d0A>/{*.dcd, *_minimized.pdb, analysis.json}.
Run on the server:  PYTHONPATH=src python scripts/e4a_umbrella_pmf.py runs/car_v5_e4a --k 10 --window-nac 3.6

Claim discipline: reports SCREENING-LEVEL access-barrier evidence only -- NOT activity/kcat. Lower
access cost = the candidate reaches the near-attack geometry more cheaply; it is NOT a rate claim.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys


def _window_dist_angle(dcd, pdb, donor, acc, leaving):
    import mdtraj as md
    import numpy as np
    t = md.load(dcd, top=pdb)
    dist = md.compute_distances(t, [[donor, acc]])[:, 0] * 10.0   # nm -> A
    ang = None
    if leaving is not None:
        ang = np.degrees(md.compute_angles(t, [[donor, acc, leaving]])[:, 0])  # O_nuc-Palpha-O_leaving
    return dist, ang


def _nac_atoms(analysis_json):
    a = ((json.load(open(analysis_json)).get("nac") or {}).get("atoms")) or {}
    return a.get("donor_heavy"), a.get("acceptor"), a.get("leaving")


def analyse_candidate(cand_dir, k_kcal, near_attack_A, angle_min, T_K=300.0, equil_frac=0.2):
    from evoliez.md.umbrella import (access_free_energy, near_attack_angle_occupancy,
                                     wham, window_overlap)
    wins = sorted(glob.glob(os.path.join(cand_dir, "window_*")))
    centers, dists, angles = [], [], []
    for wd in wins:
        m = re.search(r"window_([0-9.]+)", os.path.basename(wd))
        dcd = glob.glob(os.path.join(wd, "*.dcd"))
        pdb = glob.glob(os.path.join(wd, "*_minimized.pdb"))
        aj = os.path.join(wd, "analysis.json")
        if not (m and dcd and pdb and os.path.exists(aj)):
            continue
        donor, acc, leaving = _nac_atoms(aj)
        if donor is None or acc is None:
            continue
        try:
            d, ang = _window_dist_angle(dcd[0], pdb[0], donor, acc, leaving)
        except Exception as exc:  # noqa: BLE001
            print("   (window %s read failed: %s)" % (os.path.basename(wd), str(exc)[:60]))
            continue
        cut = int(len(d) * equil_frac)                       # DISCARD equilibration frames
        centers.append(float(m.group(1)))
        dists.append(d[cut:])
        angles.append(ang[cut:] if ang is not None else None)
    if len(centers) < 2:
        return {"cid": os.path.basename(cand_dir), "error": "need >=2 sampled windows"}
    x, pmf = wham(dists, centers, k_kcal, T_K=T_K)
    cost = access_free_energy(x, pmf, near_attack_max_A=near_attack_A)
    overlap = window_overlap(dists)
    ang_occ, n_na = (None, 0)
    if all(a is not None for a in angles):
        ang_occ, n_na = near_attack_angle_occupancy(dists, angles, near_attack_A, angle_min)
    # convergence flag = enough windows overlap AND the near-attack window was actually reached
    converged = bool(overlap.get("sufficient") and cost is not None)
    return {"cid": os.path.basename(cand_dir), "n_windows": len(centers),
            "access_cost_kcal": cost, "near_attack_angle_occ": ang_occ, "n_near_attack": n_na,
            "overlap_min": overlap.get("min_overlap"), "overlap_sufficient": overlap.get("sufficient"),
            "converged": converged}


_CLAIM_CEILING = "screening-level reaction-geometry access evidence (NOT activity/kcat/activation-barrier)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=float, default=10.0, help="umbrella k (kcal/mol/A^2), must match sampling")
    ap.add_argument("--window-nac", type=float, default=3.6, help="near-attack window (A)")
    ap.add_argument("--angle-min", type=float, default=150.0, help="in-line angle threshold (deg)")
    ap.add_argument("--out", default="docs/car_v5/e4a_pmf_access_table.csv")
    a = ap.parse_args()
    e4a = os.path.join(a.run, "e4a")
    cands = sorted(d for d in glob.glob(os.path.join(e4a, "*")) if os.path.isdir(d))
    if not cands:
        print("no candidate dirs under %s -- run the umbrella sampling first "
              "(run_md(umbrella=(window,k)) per window)" % e4a)
        sys.exit(2)
    rows = [analyse_candidate(c, a.k, a.window_nac, a.angle_min) for c in cands]
    # ONLY converged candidates are interpretable; separate access-cost from in-line NAC occupancy
    ok = [r for r in rows if not r.get("error") and r.get("converged") and r.get("access_cost_kcal") is not None]
    ok.sort(key=lambda r: r["access_cost_kcal"])
    cols = ["cid", "access_cost_kcal", "near_attack_angle_occ", "n_near_attack",
            "overlap_min", "overlap_sufficient", "converged", "n_windows"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["claim_ceiling"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            if not r.get("error"):
                w.writerow({**r, "claim_ceiling": _CLAIM_CEILING})
    print("E4a PMF -- ACCESS COST (distance) is SEPARATE from in-line NAC occupancy (angle). "
          "Only 'converged' rows are interpretable.")
    print("%-24s %9s %8s %8s %6s %s" % (
        "candidate", "access", "angOcc", "olapMin", "conv", "nwin"))
    print("-" * 72)
    for r in rows:
        if r.get("error"):
            print("%-24s  %s" % (r["cid"][:24], r["error"])); continue
        c = r["access_cost_kcal"]
        ao = r["near_attack_angle_occ"]
        print("%-24s %9s %8s %8s %6s %d" % (
            r["cid"][:24], "%.2f" % c if c is not None else "unreach",
            "%.2f" % ao if ao is not None else "n/a",
            "%.2f" % r["overlap_min"] if r["overlap_min"] is not None else "n/a",
            "yes" if r["converged"] else "NO", r["n_windows"]))
    if len(ok) >= 2:
        spread = ok[-1]["access_cost_kcal"] - ok[0]["access_cost_kcal"]
        verdict = ("PASS: candidate access costs separate (>1 kcal/mol) -- %s" % _CLAIM_CEILING
                   if spread > 1.0 else
                   "CONDITIONAL: access costs flat (<1 kcal/mol) -> classical FF cannot discriminate "
                   "the reaction distance -> QM/MM-lite is the next tier")
        print("\naccess-cost spread (converged only): %.2f kcal/mol\n%s" % (spread, verdict))
    elif not ok:
        print("\nNo CONVERGED candidate (windows did not overlap or near-attack never reached) -> "
              "adjust window spacing / k / equilibration and re-sample; DO NOT interpret candidates yet.")
    print("\nCLAIM CEILING: %s" % _CLAIM_CEILING)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
