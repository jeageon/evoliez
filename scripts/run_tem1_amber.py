#!/usr/bin/env python
"""V6-6 driver: run TEM-1 beta-lactamase (non-CAR, non-redox) through the SAME V6
Amber framework, with protein-nucleophile geometry (catalytic Ser68 Ogamma attacking
the beta-lactam scissile carbonyl C). Real backend: a Boltz-predicted Michaelis
complex -> V6-1 Amber build -> V6-2 explicit-solvent pmemd.cuda MD -> the Ser68
Ogamma reaction geometry resolved from the topology + reported as evidence.

No metal, no co-substrate, no CAR-specific code. Claim-safe (no activity claims).

Usage:
    python scripts/run_tem1_amber.py <complex.pdb> [--gpu 0] [--production-ns 0.1]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoliez.adapters.amber_builder import (  # noqa: E402
    AmberBuildSpec, AmberSystemBuilder, LigandBuildSpec, MetalBuildSpec,
    ReactiveBuildSpec,
)
from evoliez.adapters.amber_scheduler import AmberGpuScheduler, AmberMDJob  # noqa: E402


def tem1_spec() -> AmberBuildSpec:
    return AmberBuildSpec(
        ligands=[LigandBuildSpec(
            id="LIG", role="design_ligand",
            smiles="CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
            net_charge=0, allow_am1bcc=True)],
        metal=MetalBuildSpec(enabled=False),                     # non-metal chemistry
        reactive=ReactiveBuildSpec(
            donor_protein="SER:OG:68",                            # protein nucleophile
            acceptor_smarts="[CX3;r4](=[OX1])[NX3;r4]",           # beta-lactam scissile C
            transfer_is_h=False,
            label="ser68_ogamma_attack_on_betalactam_carbonyl"),
        catalytic_residues=["S68", "T69", "K71"],                 # STFK motif
        protein_ff="ff14SB", water="tip3p", solvent="explicit", box_buffer_A=12.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("complex_pdb")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--production-ns", type=float, default=0.1)
    ap.add_argument("--work", default="/mnt/data/jglee/v6_tem1/build")
    ap.add_argument("--out", default="reports/provenance/tem1_amber_system_manifest.json")
    ap.add_argument("--jsonl", default="reports/provenance/tem1_amber_md.jsonl")
    args = ap.parse_args()

    wd = Path(args.work)
    builder = AmberSystemBuilder(tem1_spec(), wd)
    sysm = builder.build(Path(args.complex_pdb))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sysm.manifest, indent=2))
    print(f"\n{'='*72}\nV6-6 TEM-1 Amber system (non-CAR, non-redox)\n{'='*72}")
    print(f"n_atoms={sysm.n_atoms} net_charge={sysm.net_charge:+.3f} fingerprint={sysm.fingerprint}")
    print("reactive map (protein-nucleophile geometry):")
    for rm in sysm.reactive_map:
        tag = f"orig#{rm.orig_resid}" if rm.orig_resid else ""
        flag = "" if rm.status == "ok" else f" [{rm.status}]"
        print(f"    {rm.role:10} {rm.mask:16} -> #{rm.amber_index} ({rm.residue}{rm.resid} {rm.atom_name}) {tag}{flag}")

    # short explicit-solvent MD + reaction-geometry analysis (V6-2)
    sched = AmberGpuScheduler([args.gpu], jsonl_path=Path(args.jsonl))
    job = AmberMDJob(job_id="tem1_wt", workdir=wd, manifest=sysm.manifest,
                     solvent="explicit", production_ns=args.production_ns,
                     equilibration_ps=100.0)
    res = sched.run([job])[0]
    print(f"\nMD: status={res.status} {res.simulation_ns}ns {res.wallclock_s}s")
    if res.failure_reason:
        print(f"  FAIL: {res.failure_reason[:300]}")
        return 1
    a = res.analysis
    ad = a.get("access_distance", {})
    an = a.get("inline_angle", {})
    print(f"  Ser68 Ogamma -> beta-lactam carbonyl C: mean={ad.get('series_mean_A')} "
          f"min={ad.get('series_min_A')} Å")
    print(f"  Nuc-C=O angle (Burgi-Dunitz, read-only): mean={an.get('series_mean_deg')}° "
          f"productive-occ={an.get('productive_angle_occupancy')}")
    print(f"  ligand retention: {a.get('ligand_retention')}  "
          f"lig_rmsd_final={a.get('ligand_rmsd_final_A')} Å  drift={a.get('energy_drift')}")
    print(f"  claim ceiling: {a.get('claim_ceiling')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
