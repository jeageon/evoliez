#!/usr/bin/env python
"""V6-2 driver: build WT + candidate CAR systems (V6-1) and run explicit-solvent
pmemd.cuda MD across the GPU pool (V6-2 scheduler), one independent job per GPU.

Prints per-job status + reaction-geometry analysis and appends provenance to
reports/provenance/amber_gpu_jobs.jsonl. Restart-safe: an already-built system or
an already-run stage is skipped.

Usage:
    python scripts/run_amber_md_car.py \
        --job wt:/path/wt_clean.pdb --job mut_00000:/path/mut_00000_anchored.pdb \
        --gpus 0,1,2 --production-ns 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoliez.adapters.amber_builder import AmberSystemBuilder  # noqa: E402
from evoliez.adapters.amber_scheduler import (  # noqa: E402
    AmberGpuScheduler, AmberMDJob,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_amber_system import car_spec  # noqa: E402


def build_if_needed(job_id: str, complex_pdb: Path, root: Path) -> Path:
    wd = root / job_id
    manifest = wd / "amber_system_manifest.json"
    if (wd / "complex.prmtop").exists() and manifest.exists():
        logging.info("system %s already built (%s)", job_id, wd)
        return wd
    logging.info("building system %s from %s", job_id, complex_pdb)
    AmberSystemBuilder(car_spec(), wd).build(complex_pdb)
    return wd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True,
                    help="job_id:complex.pdb (repeatable)")
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--production-ns", type=float, default=0.1)
    ap.add_argument("--equilibration-ps", type=float, default=100.0)
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--root", default="/mnt/data/jglee/v6_md_work/car")
    ap.add_argument("--jsonl", default="reports/provenance/amber_gpu_jobs.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    gpus = [int(g) for g in args.gpus.split(",") if g.strip() != ""]
    jobs = []
    for spec in args.job:
        jid, pdb = spec.split(":", 1)
        wd = build_if_needed(jid, Path(pdb), root)
        manifest = json.loads((wd / "amber_system_manifest.json").read_text())
        jobs.append(AmberMDJob(
            job_id=jid, workdir=wd, manifest=manifest, solvent="explicit",
            production_ns=args.production_ns, equilibration_ps=args.equilibration_ps,
            replicas=args.replicas))

    sched = AmberGpuScheduler(gpus, jsonl_path=Path(args.jsonl))
    logging.info("scheduling %d jobs across GPUs %s", len(jobs), gpus)
    results = sched.run(jobs)

    print(f"\n{'='*72}\nV6-2 AMBER GPU MD — {len(results)} jobs\n{'='*72}")
    for r in results:
        print(f"\n[{r.job_id}] status={r.status} gpu={r.gpu} "
              f"stages={'+'.join(r.stages_done)} {r.simulation_ns}ns "
              f"{r.wallclock_s}s")
        if r.failure_reason:
            print(f"  FAIL: {r.failure_reason[:300]}")
            continue
        a = r.analysis
        ad = a.get("access_distance", {})
        an = a.get("inline_angle", {})
        print(f"  access O_nuc->Pa : min={ad.get('series_min_A')} "
              f"mean={ad.get('series_mean_A')} Å, near-attack occ="
              f"{ad.get('near_attack_occupancy')} (<= {ad.get('near_attack_A')} Å)")
        print(f"  in-line angle    : mean={an.get('series_mean_deg')}°, "
              f"productive occ={an.get('productive_angle_occupancy')}")
        if "mg_retention" in a:
            mg = a["mg_retention"]
            print(f"  Mg bridge        : Mg-O_nuc={mg.get('mg_onuc_mean_A')} "
                  f"Mg-Pa={mg.get('mg_pa_mean_A')} Å, bridge occ="
                  f"{mg.get('bridge_occupancy')}")
        print(f"  ligand retention : {a.get('ligand_retention')}  "
              f"lig_rmsd_final={a.get('ligand_rmsd_final_A')} Å  "
              f"energy_drift={a.get('energy_drift')}")
        print(f"  claim ceiling    : {a.get('claim_ceiling')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
