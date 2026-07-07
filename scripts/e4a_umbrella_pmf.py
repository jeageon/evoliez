#!/usr/bin/env python
"""E4a driver — O_nuc->Palpha umbrella / PMF access-barrier screen (reviewer track E4a).

WHY: E2 found unbiased explicit MD never occupies the productive ~3 A window. But the near-attack
conformation is TRANSIENT, so the right observable is the FREE-ENERGY COST to ACCESS it, not "does it
sit there". Umbrella sampling along the O_nuc->Palpha distance + WHAM gives that cost per candidate,
using the existing classical FF -- cheaper and less risky than QM/MM, and the correct next tier.

TWO PHASES:
  (1) SAMPLE (server, OpenMM): for each candidate, run one short explicit-solvent MD per umbrella
      window with a harmonic O_nuc->Palpha bias. s10 drives this when EVOLIEZ_E4A_UMBRELLA="dmin,dmax,n,k"
      is set (`evoliez run ... --to s10`); each window writes umbrella_samples.json (the full biased
      O_nuc->Palpha distance + read-only angle timeseries -- NO DCD; the NAC subframes ARE the samples).
  (2) ANALYSE (this script, pure-python WHAM -- no mdtraj): read the per-window umbrella_samples.json,
      WHAM -> PMF -> near-attack access cost + angle-occupancy + overlap QC.

Layout: <run>/md/<candidate>/e4a/window_<d0A>/umbrella_samples.json (as s10 writes it).
Run on the server:  PYTHONPATH=src python scripts/e4a_umbrella_pmf.py runs/srcar_3hp_v5_e4a --window-nac 3.6

Claim discipline: reports SCREENING-LEVEL access-barrier evidence only -- NOT activity/kcat. Lower
access cost = the candidate reaches the near-attack geometry more cheaply; it is NOT a rate claim.
"""
import argparse
import csv
import glob
import json
import os
import sys


def _read_window(window_dir):
    """Return (center_A, k_kcal, distances[], angles[]) from a window's umbrella_samples.json,
    or None if it is absent/empty (window did not sample)."""
    p = os.path.join(window_dir, "umbrella_samples.json")
    if not os.path.exists(p):
        return None
    try:
        s = json.load(open(p))
    except Exception:  # noqa: BLE001
        return None
    d = s.get("distances_A") or []
    if len(d) < 2:
        return None
    return float(s["window_A"]), float(s.get("k_kcal", 0.0)), d, (s.get("angles_deg") or None)


def analyse_candidate(cand_e4a_dir, k_kcal, near_attack_A, angle_min, T_K=300.0, equil_frac=0.2):
    """cand_e4a_dir = <run>/md/<cid>/e4a ; WHAM the per-window samples -> access cost + QC."""
    from evoliez.md.umbrella import (access_free_energy, near_attack_angle_occupancy,
                                     wham, window_overlap)
    cid = os.path.basename(os.path.dirname(cand_e4a_dir))
    wins = sorted(glob.glob(os.path.join(cand_e4a_dir, "window_*")))
    centers, dists, angles, k_seen = [], [], [], k_kcal
    for wd in wins:
        r = _read_window(wd)
        if r is None:
            continue
        c, kw, d, ang = r
        if kw > 0:
            k_seen = kw                                      # k recorded at sampling is authoritative
        cut = int(len(d) * equil_frac)                       # DISCARD equilibration frames
        centers.append(c)
        dists.append(d[cut:])
        angles.append(ang[cut:] if ang is not None else None)
    if len(centers) < 2:
        return {"cid": cid, "error": "need >=2 sampled windows (got %d)" % len(centers)}
    x, pmf = wham(dists, centers, k_seen, T_K=T_K)
    cost = access_free_energy(x, pmf, near_attack_max_A=near_attack_A)
    overlap = window_overlap(dists)
    ang_occ, n_na = (None, 0)
    if all(a is not None for a in angles):
        ang_occ, n_na = near_attack_angle_occupancy(dists, angles, near_attack_A, angle_min)
    # convergence flag = enough windows overlap AND the near-attack window was actually reached
    converged = bool(overlap.get("sufficient") and cost is not None)
    return {"cid": cid, "n_windows": len(centers), "k_kcal": round(k_seen, 3),
            "access_cost_kcal": cost, "near_attack_angle_occ": ang_occ, "n_near_attack": n_na,
            "overlap_min": overlap.get("min_overlap"), "overlap_sufficient": overlap.get("sufficient"),
            "converged": converged}


_CLAIM_CEILING = "screening-level reaction-geometry access evidence (NOT activity/kcat/activation-barrier)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=float, default=10.0, help="fallback umbrella k (kcal/mol/A^2) if not "
                    "recorded in umbrella_samples.json")
    ap.add_argument("--window-nac", type=float, default=3.6, help="near-attack window (A)")
    ap.add_argument("--angle-min", type=float, default=150.0, help="in-line angle threshold (deg)")
    ap.add_argument("--out", default="docs/car_v5/e4a_pmf_access_table.csv")
    a = ap.parse_args()
    # s10 writes windows under <run>/md/<cid>/e4a/window_<d0>/ ; a candidate dir is any e4a dir with windows
    cands = sorted(d for d in glob.glob(os.path.join(a.run, "md", "*", "e4a"))
                   if glob.glob(os.path.join(d, "window_*")))
    if not cands:
        print("no <run>/md/*/e4a/window_* dirs under %s -- run the umbrella sampling first: "
              "EVOLIEZ_E4A_UMBRELLA='2.8,5.4,10,10' evoliez run --config <e4a> --to s10" % a.run)
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
