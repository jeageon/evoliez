"""Shared helpers for adapters: deterministic synthetic structures, pocket
placement, and a minimal PDB writer (no Biopython dependency required)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

from evoliez.types import Ligand, LigandAtom, Pose, ProteinStructure, Residue
from evoliez.utils.seeds import derive_seed

_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL", "X": "GLY",
}


def synthetic_structure(
    sequence: str, *, seed: int, method: str = "mock"
) -> ProteinStructure:
    """A deterministic alpha-helix-like backbone. Geometry is not physical but
    is stable and reproducible - enough to exercise contacts/graph/MD plumbing."""
    residues: List[Residue] = []
    rise, radius, turn = 1.5, 2.3, math.radians(100.0)
    for i, aa in enumerate(sequence):
        ang = i * turn
        ca = (radius * math.cos(ang), radius * math.sin(ang), i * rise)
        sc = (
            (radius + 1.8) * math.cos(ang),
            (radius + 1.8) * math.sin(ang),
            i * rise + 0.4,
        )
        h = derive_seed(seed, aa, str(i))
        residues.append(
            Residue(
                index=i + 1,
                aa=aa if aa in _THREE else "X",
                ca=ca,
                sidechain_centroid=sc,
                secondary_structure="H",
                sasa=round((h % 100) / 100.0, 3),
                plddt=round(70.0 + (h % 30), 2),
            )
        )
    return ProteinStructure(
        sequence=sequence,
        residues=residues,
        method=method,
        confidence=round(0.6 + (derive_seed(seed, "conf") % 35) / 100.0, 3),
    )


def place_ligand_in_pocket(
    structure: ProteinStructure, ligand: Ligand, *, seed: int
) -> List[LigandAtom]:
    """Translate the ligand so its centroid sits near a deterministic pocket
    residue, keeping its internal geometry."""
    if not structure.residues:
        return [LigandAtom(**vars(a)) for a in ligand.atoms]
    pocket_res = structure.residues[
        derive_seed(seed, "pocket") % len(structure.residues)
    ]
    target = pocket_res.sidechain_centroid or pocket_res.ca
    if ligand.atoms:
        cx = sum(a.coord[0] for a in ligand.atoms) / len(ligand.atoms)
        cy = sum(a.coord[1] for a in ligand.atoms) / len(ligand.atoms)
        cz = sum(a.coord[2] for a in ligand.atoms) / len(ligand.atoms)
    else:
        cx = cy = cz = 0.0
    dx, dy, dz = target[0] - cx + 3.0, target[1] - cy, target[2] - cz
    placed: List[LigandAtom] = []
    for a in ligand.atoms:
        na = LigandAtom(**vars(a))
        na.coord = (a.coord[0] + dx, a.coord[1] + dy, a.coord[2] + dz)
        placed.append(na)
    return placed


def write_min_pdb(
    path: Path,
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom] | None = None,
) -> None:
    """Minimal PDB (CA-only protein + ligand HETATMs). Lets users open mock
    runs in PyMOL and gives real tools a concrete file path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    serial = 1
    for res in structure.residues:
        x, y, z = res.ca
        lines.append(
            f"ATOM  {serial:>5} {'CA':<4}{_THREE.get(res.aa,'GLY'):>3} A"
            f"{res.index:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00"
            f"{res.plddt:6.2f}           C"
        )
        serial += 1
    if ligand_atoms:
        for a in ligand_atoms:
            x, y, z = a.coord
            lines.append(
                f"HETATM{serial:>5} {a.id[:4]:<4}LIG L   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                f"{a.element:>2}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def mock_redock(
    candidate_id: str,
    method: str,
    reference_atoms: Sequence[LigandAtom],
    *,
    instability: float,
    score_offset: float = 0.0,
) -> Pose:
    """Deterministic synthetic redocking pose. ``instability`` (0..1, derived
    from how disruptive a mutation is) drives RMSD-to-reference and escape."""
    seed = derive_seed(0xD0CC, candidate_id, method)
    jitter = 0.3 + 3.5 * instability
    moved: List[LigandAtom] = []
    for i, a in enumerate(reference_atoms):
        h = derive_seed(seed, str(i))
        off = ((h % 200) / 100.0 - 1.0) * jitter
        na = LigandAtom(**vars(a))
        na.coord = (a.coord[0] + off, a.coord[1] - off * 0.5, a.coord[2] + off * 0.3)
        moved.append(na)
    if reference_atoms:
        import numpy as np

        ref = np.array([a.coord for a in reference_atoms], float)
        new = np.array([a.coord for a in moved], float)
        rmsd_val = float(np.sqrt(((ref - new) ** 2).sum(axis=1).mean()))
    else:
        rmsd_val = 0.0
    score = round(-9.0 + 6.0 * instability + score_offset + (seed % 50) / 100.0, 3)
    return Pose(
        candidate_id=candidate_id,
        method=method,
        score=score,
        ligand_atoms=moved,
        rmsd_to_reference=round(rmsd_val, 3),
        cluster=seed % 4,
    )


def mock_dock_ensemble(
    candidate_id: str,
    method: str,
    reference_atoms: Sequence[LigandAtom],
    *,
    n_poses: int,
    base_instability: float = 0.15,
) -> List[Pose]:
    """Deterministic K-pose ensemble. Most poses cluster near the reference
    (low instability); a deterministic minority are displaced outliers so the
    statistical pose-selection step has something to reject."""
    poses: List[Pose] = []
    for p in range(n_poses):
        h = derive_seed(0xE17B, candidate_id, method, str(p))
        # ~25% of poses are outliers (large displacement)
        outlier = (h % 4) == 0
        inst = (0.75 + (h % 25) / 100.0) if outlier else (
            base_instability + (h % 30) / 300.0
        )
        pose = mock_redock(
            f"{candidate_id}#p{p}", method, reference_atoms,
            instability=inst, score_offset=(h % 20) / 100.0 - 0.1,
        )
        pose.candidate_id = candidate_id
        pose.cluster = p
        poses.append(pose)
    return poses
