#!/usr/bin/env python
"""V6-1 driver: build the CAR (or a config-described) Amber system from a complex
PDB and print/record the manifest + reactive-atom map.

CAR default: SrCAR A-domain + 3-HP (design, -1, AM1-BCC) + ATP (cofactor, -4,
curated) + Mg (12-6-4), ff14SB/TIP3P explicit. Run from the repo root so
``params/atp_4minus/ATP.fixed.mol2`` resolves.

Usage:
    python scripts/build_amber_system.py <complex.pdb> [--work DIR] [--out JSON]
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
    ReactiveBuildSpec, AmberBuildError,
)


def car_spec() -> AmberBuildSpec:
    return AmberBuildSpec(
        ligands=[
            LigandBuildSpec(id="LIG", role="design_ligand", smiles="OCCC(=O)[O-]",
                            net_charge=-1, allow_am1bcc=True),
            LigandBuildSpec(
                id="ATP", role="cofactor",
                smiles="Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])"
                       "OP(=O)([O-])[O-])[C@@H](O)[C@H]1O",
                net_charge=-4, allow_am1bcc=False,
                charges_mol2="params/atp_4minus/ATP.fixed.mol2"),
        ],
        metal=MetalBuildSpec(enabled=True, ion="MG", element="MG",
                             ion_frcmod="frcmod.ions234lm_1264_tip3p"),
        reactive=ReactiveBuildSpec(
            donor_smarts="[OX1-]", acceptor_smarts="[PX4]([OX2][CX4])",
            transfer_is_h=False, label="3HP_carboxylate_to_ATP_alphaP"),
        catalytic_residues=["S268", "T269", "K273"],
        protein_ff="ff14SB", water="tip3p", solvent="explicit", box_buffer_A=12.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("complex_pdb")
    ap.add_argument("--work", default="/mnt/data/jglee/v6_build_work/car")
    ap.add_argument("--out", default="reports/provenance/amber_system_manifest.json")
    args = ap.parse_args()

    spec = car_spec()
    builder = AmberSystemBuilder(spec, Path(args.work))
    try:
        sysm = builder.build(Path(args.complex_pdb))
    except AmberBuildError as exc:
        print(f"\nBUILD FAILED: {exc}\n")
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sysm.manifest, indent=2))

    print(f"\n{'='*72}\nAMBER SYSTEM BUILT\n{'='*72}")
    print(f"prmtop        : {sysm.prmtop}")
    print(f"fingerprint   : {sysm.fingerprint}")
    print(f"n_atoms       : {sysm.n_atoms}")
    print(f"net_charge    : {sysm.net_charge:+.4f}")
    print(f"ligands       : " + ", ".join(
        f"{l['id']}({l['role']},{l['net_charge']:+d},{l['param_source']})"
        for l in sysm.manifest["ligands"]))
    m = sysm.manifest.get("metal")
    if m:
        print(f"metal         : {m['ion']}{m['charge']:+d} [{m['lj_model']}] {m['ion_frcmod']}")
    print("reactive map  :")
    for rm in sysm.reactive_map:
        tag = f"orig#{rm.orig_resid}" if rm.orig_resid else ""
        flag = "" if rm.status == "ok" else f"  [{rm.status}]"
        print(f"    {rm.role:10} {rm.mask:16} -> Amber #{rm.amber_index} "
              f"({rm.residue}{rm.resid} {rm.atom_name}) {tag}{flag}")
    print(f"\nmanifest      : {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
