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


def _nac_atoms(analysis_json):
    a = ((json.load(open(analysis_json)).get("nac") or {}).get("atoms")) or {}
    return a.get("donor_heavy"), a.get("acceptor")


def _window_distance(dcd, pdb, donor, acc):
    import mdtraj as md
    t = md.load(dcd, top=pdb)
    return md.compute_distances(t, [[donor, acc]])[:, 0] * 10.0   # nm -> A


def analyse_candidate(cand_dir, k_kcal, near_attack_A, T_K=300.0):
    from evoliez.md.umbrella import access_free_energy, wham
    wins = sorted(glob.glob(os.path.join(cand_dir, "window_*")))
    centers, samples = [], []
    for wd in wins:
        m = re.search(r"window_([0-9.]+)", os.path.basename(wd))
        dcd = glob.glob(os.path.join(wd, "*.dcd"))
        pdb = glob.glob(os.path.join(wd, "*_minimized.pdb"))
        aj = os.path.join(wd, "analysis.json")
        if not (m and dcd and pdb and os.path.exists(aj)):
            continue
        donor, acc = _nac_atoms(aj)
        if donor is None or acc is None:
            continue
        try:
            d = _window_distance(dcd[0], pdb[0], donor, acc)
        except Exception as exc:  # noqa: BLE001
            print("   (window %s read failed: %s)" % (os.path.basename(wd), str(exc)[:60]))
            continue
        centers.append(float(m.group(1)))
        samples.append(d)
    if len(centers) < 2:
        return {"cid": os.path.basename(cand_dir), "error": "need >=2 sampled windows"}
    x, pmf = wham(samples, centers, k_kcal, T_K=T_K)
    cost = access_free_energy(x, pmf, near_attack_max_A=near_attack_A)
    return {"cid": os.path.basename(cand_dir), "n_windows": len(centers),
            "access_cost_kcal": cost, "d_span": (min(centers), max(centers))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=float, default=10.0, help="umbrella k (kcal/mol/A^2), must match sampling")
    ap.add_argument("--window-nac", type=float, default=3.6, help="near-attack window (A)")
    ap.add_argument("--out", default="docs/car_v5/e4a_pmf_access_table.csv")
    a = ap.parse_args()
    e4a = os.path.join(a.run, "e4a")
    cands = sorted(d for d in glob.glob(os.path.join(e4a, "*")) if os.path.isdir(d))
    if not cands:
        print("no candidate dirs under %s -- run the umbrella sampling first "
              "(run_md(umbrella=(window,k)) per window)" % e4a)
        sys.exit(2)
    rows = [analyse_candidate(c, a.k, a.window_nac) for c in cands]
    ok = [r for r in rows if not r.get("error") and r.get("access_cost_kcal") is not None]
    ok.sort(key=lambda r: r["access_cost_kcal"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cid", "n_windows", "access_cost_kcal"], extrasaction="ignore")
        w.writeheader()
        [w.writerow(r) for r in rows if not r.get("error")]
    print("near-attack access cost (kcal/mol) -- lower = reaches O_nuc-Palpha <=%.1f A more cheaply" % a.window_nac)
    print("%-26s %8s %s" % ("candidate", "cost", "windows"))
    print("-" * 48)
    for r in rows:
        if r.get("error"):
            print("%-26s  %s" % (r["cid"][:26], r["error"])); continue
        c = r["access_cost_kcal"]
        print("%-26s %8s %d" % (r["cid"][:26], "%.2f" % c if c is not None else "unsampled", r["n_windows"]))
    if len(ok) >= 2:
        spread = ok[-1]["access_cost_kcal"] - ok[0]["access_cost_kcal"]
        print("\naccess-cost spread across candidates: %.2f kcal/mol "
              "(%s = discrimination; ~0 = classical FF cannot discriminate -> QM/MM-lite)"
              % (spread, ">1" if spread > 1 else "flat"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
