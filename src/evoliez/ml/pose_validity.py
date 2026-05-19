"""Lightweight pose-sanity stand-in (expert review: PoseBusters-style check).

A minimal geometric plausibility check usable without external packages.
NOT a replacement for PoseBusters - real physical-validity / PDBbind
redocking validation needs the `posebusters` package + reference data
(see docs/BENCHMARKS.md). This catches gross failures (atomic clashes,
ligand expelled from the protein) as a cheap pre-filter / report flag.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from evoliez.features.geometry import dist, ligand_centroid
from evoliez.types import LigandAtom, ProteinStructure

_MIN_NONBONDED = 0.9  # Å; closer than this between non-adjacent atoms = clash


def pose_sanity(
    ligand_atoms: Sequence[LigandAtom],
    structure: Optional[ProteinStructure] = None,
    *,
    pocket_radius: float = 20.0,
) -> Dict[str, object]:
    n = len(ligand_atoms)
    clashes = 0
    for i in range(n):
        for j in range(i + 2, n):  # skip i, i+1 (likely bonded)
            if dist(ligand_atoms[i].coord, ligand_atoms[j].coord) < _MIN_NONBONDED:
                clashes += 1
    clash_free = clashes == 0

    in_pocket = True
    if structure and structure.residues and ligand_atoms:
        lc = ligand_centroid(ligand_atoms)
        pc = (
            sum(r.ca[0] for r in structure.residues) / len(structure.residues),
            sum(r.ca[1] for r in structure.residues) / len(structure.residues),
            sum(r.ca[2] for r in structure.residues) / len(structure.residues),
        )
        in_pocket = dist(lc, pc) <= pocket_radius + 25.0

    return {
        "n_ligand_atoms": n,
        "internal_clashes": clashes,
        "clash_free": clash_free,
        "in_pocket": in_pocket,
        "valid": bool(clash_free and in_pocket),
        "note": "geometric stand-in; use PoseBusters for physical validity",
    }
