#!/usr/bin/env python
"""V6-3 driver: run an Amber-native O_nuc→Pα PMF on a built system and report the
WHAM curve, QC (window overlap, first/second-half convergence, in-line angle
occupancy) and the claim-safe verdict (converged -> screening prioritization;
non-converged -> diagnostic only, ranking prohibited).

Reads the V6-1 system manifest for the reactive-atom indices + masks. If a prior
run left an ``equil.rst`` in the system dir, it is reused as the shared start.

Usage:
    python scripts/run_amber_pmf.py --system /path/to/built/wt \
        --dmin 2.8 --dmax 4.6 --nwin 8 --k 100 --production-ns 0.15 --gpus 0,1,2
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoliez.amber.pmf import PmfSpec, run_pmf  # noqa: E402
from evoliez.md.umbrella import umbrella_windows  # noqa: E402


def spec_from_manifest(manifest: dict, windows, k, prod_ns) -> PmfSpec:
    reac = {a["role"]: a for a in manifest.get("reactive_atoms", [])}
    onuc, pa = reac.get("O_nuc"), reac.get("P_alpha")
    if not (onuc and pa):
        raise SystemExit("manifest missing O_nuc / P_alpha reactive atoms")
    olv = reac.get("O_leaving")
    angle_masks = ((onuc["mask"], pa["mask"], olv["mask"]) if olv else None)
    return PmfSpec(
        windows_A=windows, k_kcal=k,
        dist_i=onuc["amber_index"], dist_j=pa["amber_index"],
        dist_masks=(onuc["mask"], pa["mask"]), angle_masks=angle_masks,
        production_ns=prod_ns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="built system dir (complex.prmtop)")
    ap.add_argument("--dmin", type=float, default=2.8)
    ap.add_argument("--dmax", type=float, default=4.6)
    ap.add_argument("--nwin", type=int, default=8)
    ap.add_argument("--k", type=float, default=100.0)
    ap.add_argument("--production-ns", type=float, default=0.15)
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--jsonl", default="reports/provenance/amber_pmf_jobs.jsonl")
    args = ap.parse_args()

    sysdir = Path(args.system)
    manifest = json.loads((sysdir / "amber_system_manifest.json").read_text())
    # reuse a prior equilibrated restart if present
    for cand in ("equil.rst", "shared_equil.rst"):
        src = sysdir / cand
        if src.exists() and not (sysdir / "shared_equil.rst").exists():
            shutil.copy(src, sysdir / "shared_equil.rst")
            logging.info("reusing %s as shared PMF start", cand)
            break

    windows = umbrella_windows(args.dmin, args.dmax, args.nwin)
    spec = spec_from_manifest(manifest, windows, args.k, args.production_ns)
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    logging.info("PMF: %d windows %.2f-%.2f Å, k=%.0f, %.2fns/window on GPUs %s",
                 len(windows), args.dmin, args.dmax, args.k, args.production_ns, gpus)

    res = run_pmf(sysdir, spec, gpus, jsonl_path=Path(args.jsonl))

    print(f"\n{'='*72}\nV6-3 AMBER PMF — O_nuc->Pα access barrier\n{'='*72}")
    print(f"verdict        : {res.verdict}")
    print(f"classification : {res.classification}")
    print(f"access cost    : {res.access_cost_kcal} kcal/mol")
    if res.failure_reason:
        print(f"failure        : {res.failure_reason}")
    ov = res.qc.get("overlap", {})
    cv = res.qc.get("convergence", {})
    print(f"window overlap : mean={ov.get('mean_overlap')} min={ov.get('min_overlap')} "
          f"sufficient={ov.get('sufficient')}")
    print(f"convergence    : 1st-half access={cv.get('access_first_half_kcal')} "
          f"2nd-half={cv.get('access_second_half_kcal')} "
          f"drift={cv.get('access_drift_kcal')} consistent={cv.get('halves_consistent')}")
    print(f"in-line angle  : near-attack angle occ={res.qc.get('near_attack_angle_occupancy')} "
          f"(over {res.qc.get('near_attack_frames')} near-attack frames)")
    print(f"windows        : {res.qc.get('windows_with_samples')}/{res.qc.get('n_windows')} sampled")
    for w in res.windows:
        print(f"    center {w['center_A']:.3f} Å : n={w['n_samples']} mean={w['mean_A']}")
    print(f"claim ceiling  : {res.claim_ceiling}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
