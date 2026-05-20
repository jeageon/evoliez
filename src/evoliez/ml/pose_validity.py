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
    receptor_clashes = 0
    if structure and structure.residues and ligand_atoms:
        lc = ligand_centroid(ligand_atoms)
        pc = (
            sum(r.ca[0] for r in structure.residues) / len(structure.residues),
            sum(r.ca[1] for r in structure.residues) / len(structure.residues),
            sum(r.ca[2] for r in structure.residues) / len(structure.residues),
        )
        in_pocket = dist(lc, pc) <= pocket_radius + 25.0
        # Protein-ligand clash count: any (ligand-atom, residue-CA) pair
        # closer than _MIN_NONBONDED is a hard collision (CA proxy; a real
        # PoseBusters check uses all heavy atoms - this is the cheap
        # stand-in for the mock path).
        for la in ligand_atoms:
            for r in structure.residues:
                if dist(la.coord, r.ca) < _MIN_NONBONDED:
                    receptor_clashes += 1

    reasons = []
    if not clash_free:
        reasons.append(f"{clashes} internal ligand clashes (< 0.9 A)")
    if not in_pocket:
        reasons.append("ligand expelled from pocket region")
    if receptor_clashes:
        reasons.append(f"{receptor_clashes} protein-ligand atom clashes")
    valid = clash_free and in_pocket and receptor_clashes == 0
    return {
        "n_ligand_atoms": n,
        "internal_clashes": clashes,
        "receptor_clashes": receptor_clashes,
        "clash_free": clash_free,
        "in_pocket": in_pocket,
        "valid": bool(valid),
        "status": "valid" if valid else "invalid",
        "reasons": reasons,
        "note": "geometric stand-in; use PoseBusters for physical validity",
    }


def pose_rmsd(
    pose_atoms: Sequence[LigandAtom], ref_atoms: Sequence[LigandAtom]
) -> float:
    """Ligand heavy-atom RMSD vs a reference pose (Å) - the standard
    docking/pose-prediction success metric (success usually RMSD <= 2 Å).
    Assumes matched atom order (guaranteed by the atom-index lock)."""
    n = min(len(pose_atoms), len(ref_atoms))
    if n == 0:
        return 0.0
    ss = sum(
        dist(pose_atoms[i].coord, ref_atoms[i].coord) ** 2 for i in range(n)
    )
    return round((ss / n) ** 0.5, 4)


def plif_recovery(pred_ifp: Sequence, ref_ifp: Sequence) -> float:
    """Protein-Ligand Interaction Fingerprint recovery: fraction of the
    reference (residue, ligand-atom, type) contacts reproduced by the
    predicted pose. Takes IFPContact-like objects."""
    def key(c):
        return (c.residue_index, c.ligand_atom_id, c.itype)

    ref = {key(c) for c in ref_ifp}
    if not ref:
        return 0.0
    pred = {key(c) for c in pred_ifp}
    return round(len(ref & pred) / len(ref), 4)
