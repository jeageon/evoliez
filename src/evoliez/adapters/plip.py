"""Standardized protein-ligand interaction fingerprint (user §3).

real: PLIP (https://github.com/pharmai/plip) on a pose PDB if available.
mock: deterministic geometry-rule fingerprint over the 8 standard
non-covalent interaction types. Output is an edge feature / weak label and a
report annotation - never a supervised label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from evoliez.config import Backend
from evoliez.features.geometry import dist
from evoliez.types import LigandAtom, ProteinStructure

IFP_TYPES = [
    "hbond", "salt_bridge", "hydrophobic", "pi_stack",
    "cation_pi", "halogen", "metal", "water_mediated",
]

_POS_AA = {"K", "R", "H"}
_NEG_AA = {"D", "E"}
_AROM_AA = {"F", "Y", "W", "H"}
_HYDRO_AA = {"A", "V", "L", "I", "M", "F", "W", "C"}
_METAL_AA = {"H", "D", "E", "C"}
_HALOGENS = {"F", "Cl", "Br", "I"}


@dataclass
class IFPContact:
    residue_index: int
    ligand_atom_id: str
    itype: str
    distance: float


def _classify(res_aa: str, atom: LigandAtom, d: float) -> str:
    if atom.element in _HALOGENS and d <= 4.0:
        return "halogen"
    if atom.formal_charge < 0 and res_aa in _POS_AA and d <= 4.0:
        return "salt_bridge"
    if atom.formal_charge > 0 and res_aa in _NEG_AA and d <= 4.0:
        return "salt_bridge"
    if atom.aromatic and res_aa in _AROM_AA and d <= 4.5:
        return "pi_stack"
    if atom.formal_charge > 0 and res_aa in _AROM_AA and d <= 4.5:
        return "cation_pi"
    if res_aa in _METAL_AA and atom.formal_charge != 0 and d <= 2.6:
        return "metal"
    if (atom.is_acceptor or atom.is_donor) and d <= 3.5:
        return "hbond"
    if atom.is_hydrophobic and res_aa in _HYDRO_AA and d <= 4.5:
        return "hydrophobic"
    return ""


def fingerprint(
    structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom],
    *,
    backend: Backend = Backend.mock,
    workdir: Path | None = None,
    cutoff: float = 4.5,
) -> List[IFPContact]:
    out: List[IFPContact] = []
    for r in structure.residues:
        ref = r.sidechain_centroid or r.ca
        for a in ligand_atoms:
            d = dist(ref, a.coord)
            if d <= cutoff:
                t = _classify(r.aa, a, d)
                if t:
                    out.append(IFPContact(r.index, a.id, t, round(d, 3)))
    return out


def per_atom_profile(contacts: Sequence[IFPContact]) -> dict:
    prof: dict = {}
    for c in contacts:
        prof.setdefault(c.ligand_atom_id, {t: 0 for t in IFP_TYPES})
        prof[c.ligand_atom_id][c.itype] += 1
    return prof


def type_counts(contacts: Sequence[IFPContact]) -> dict:
    counts = {t: 0 for t in IFP_TYPES}
    for c in contacts:
        counts[c.itype] += 1
    return counts
