#!/usr/bin/env python
"""V6-5 driver: QM/MM-lite reaction-core minimization for one or more built CAR
systems, from a near-attack frame each, comparing the QM-relaxed reactive-core
geometry (does the in-line angle become productive under QM?).

QM region = the reaction core (3-HP + ATP + Mg, net charge −3), semiempirical PM6
(sqm inside sander); the solvated protein stays MM, backbone restrained.

Usage:
    python scripts/run_qmmm_lite.py \
        --job wt:/path/wt:/path/wt/win_2.800/prod.rst \
        --job mut_00000:/path/mut_00000:/path/mut_00000/win_3.200/prod.rst
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoliez.amber.qmmm import QmmmLiteSpec, run_qmmm_lite  # noqa: E402
from evoliez.adapters.amber_scheduler import reactive_masks_from_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True,
                    help="job_id:system_dir:near_attack_rst (repeatable)")
    ap.add_argument("--qm-charge", type=int, default=-3)   # 3-HP −1 + ATP −4 + Mg +2
    ap.add_argument("--theory", default="PM6")
    ap.add_argument("--min-steps", type=int, default=300)
    ap.add_argument("--jsonl", default="reports/provenance/amber_qmmm_jobs.jsonl")
    args = ap.parse_args()

    results = []
    for spec_s in args.job:
        jid, sysdir, rst = spec_s.split(":", 2)
        sysdir, rst = Path(sysdir), Path(rst)
        manifest = json.loads((sysdir / "amber_system_manifest.json").read_text())
        masks = reactive_masks_from_manifest(manifest)
        spec = QmmmLiteSpec(qm_residues=["LIG", "ATP", "MG"], qm_charge=args.qm_charge,
                            qm_theory=args.theory, min_steps=args.min_steps)
        logging.info("QM/MM-lite %s from %s (QM=LIG,ATP,MG charge=%d %s)",
                     jid, rst.name, args.qm_charge, args.theory)
        res = run_qmmm_lite(sysdir, rst, spec, masks, tag=f"qmmm_{jid}")
        results.append((jid, res))

    jl = Path(args.jsonl)
    jl.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nV6-5 QM/MM-LITE — reaction-core (3-HP + ATP + Mg, PM6)\n{'='*72}")
    with jl.open("a") as fh:
        for jid, r in results:
            g = r.reactive_geometry
            print(f"\n[{jid}] status={r.status} QM_atoms={r.qm_atoms} "
                  f"E={r.qmmm_energy_kcal} kcal/mol")
            if r.failure_reason:
                print(f"  FAIL: {r.failure_reason[:300]}")
            else:
                print(f"  QM-relaxed core: O_nuc->Pa={g.get('d_onuc_pa')} Å  "
                      f"in-line angle={g.get('a_inline')}°  "
                      f"Mg-O_nuc={g.get('d_mg_onuc')} Mg-Pa={g.get('d_mg_pa')} Å")
            print(f"  claim ceiling  : {r.claim_ceiling}")
            fh.write(json.dumps({"schema": "amber_qmmm/v1", "job_id": jid,
                                 "status": r.status, "qmmm_energy_kcal": r.qmmm_energy_kcal,
                                 "reactive_geometry": g, "qm_atoms": r.qm_atoms,
                                 "claim_ceiling": r.claim_ceiling}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
