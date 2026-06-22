"""Catalytic-power screen: near-attack-conformation (NAC) / reactive-geometry
occupancy from an MD trajectory.

s10's ``md_lite_score`` measures BINDING / STRUCTURAL stability (does the ligand
stay, is the pocket intact, are contacts kept). It does NOT measure CATALYTIC
competence. This module adds the reactive-geometry layer: per frame it asks
whether the reacting atoms sit in a near-attack conformation -- the transferring
atom within ``distance_max`` of the acceptor AND a productive
donor--transferring--acceptor angle -- and reports

    NAC occupancy = fraction of frames that are reaction-competent.

A mutant whose NAC occupancy beats WT has a geometrically more productive active
site: the screenable proxy for "catalytic power" short of a QM/MM barrier.

GENERIC by design: the reaction is described by a :class:`ReactiveSpec` (donor /
acceptor SMARTS + distance/angle criteria) that comes from config -- no enzyme,
cofactor or substrate is hardcoded here. The FDH formate->NADP-C4 hydride
transfer is just one spec the config can express. Pure RDKit + NumPy so the
geometry/occupancy core is unit-testable off the server; the engine layer only
has to supply per-frame coordinates and the donor/acceptor residue offsets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ReactiveSpec:
    """One bond-forming/breaking event to monitor.

    The donor heavy atom is atom ``donor_idx`` of a ``donor_smarts`` match; the
    transferring atom is (by default) the H on it, else ``donor_idx`` itself.
    The acceptor is atom ``acceptor_idx`` of an ``acceptor_smarts`` match. Donor
    and acceptor may live in different molecules/residues (e.g. formate vs NADP).
    """
    donor_smarts: str             # e.g. formate '[CX3H1](=O)[O-]'
    acceptor_smarts: str          # e.g. nicotinamide C4 (see configs)
    donor_idx: int = 0            # which atom of the donor match is the heavy donor
    acceptor_idx: int = 0         # which atom of the acceptor match is the acceptor
    transfer_is_h: bool = True    # transferring atom = the H on the donor heavy atom
    distance_max: float = 3.5     # Å, transferring atom -> acceptor (hydride ~2.7-3.5)
    angle_min: float = 150.0      # deg, donor_heavy - transferring - acceptor (~linear)
    label: str = "reaction"


@dataclass
class NACResult:
    occupancy: float = 0.0        # fraction of reaction-competent frames
    n_frames: int = 0
    n_reactive: int = 0
    distance_mean: float = float("nan")
    distance_min: float = float("nan")
    angle_mean: float = float("nan")
    distances: List[float] = field(default_factory=list)
    angles: List[float] = field(default_factory=list)
    atoms: Dict[str, int] = field(default_factory=dict)   # trajectory atom indices
    label: str = "reaction"
    note: str = ""

    def to_json(self) -> Dict[str, object]:
        return {
            "label": self.label, "nac_occupancy": self.occupancy,
            "n_frames": self.n_frames, "n_reactive": self.n_reactive,
            "distance_mean": self.distance_mean, "distance_min": self.distance_min,
            "angle_mean": self.angle_mean, "atoms": self.atoms, "note": self.note,
        }


# --------------------------------------------------------------------------- #
# RDKit atom resolution (donor / acceptor may be different molecules)
# --------------------------------------------------------------------------- #
def _first_match_atom(mol, smarts: str, idx: int) -> Optional[int]:
    from rdkit import Chem
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return None
    ms = mol.GetSubstructMatches(patt)
    if not ms or idx >= len(ms[0]):
        return None
    return ms[0][idx]


def identify_donor(mol, spec: ReactiveSpec) -> Optional[Dict[str, int]]:
    """Resolve {heavy, transfer} atom indices (local to ``mol``, which must
    carry explicit Hs) for the donor fragment. None if absent/ambiguous."""
    heavy = _first_match_atom(mol, spec.donor_smarts, spec.donor_idx)
    if heavy is None:
        return None
    out = {"heavy": heavy}
    if spec.transfer_is_h:
        hs = [n.GetIdx() for n in mol.GetAtomWithIdx(heavy).GetNeighbors()
              if n.GetSymbol() == "H"]
        if not hs:
            return None
        out["transfer"] = hs[0]
    else:
        out["transfer"] = heavy
    return out


def identify_acceptor(mol, spec: ReactiveSpec) -> Optional[int]:
    """Resolve the acceptor atom index (local to ``mol``). None if absent."""
    return _first_match_atom(mol, spec.acceptor_smarts, spec.acceptor_idx)


# --------------------------------------------------------------------------- #
# Geometry + occupancy core (no RDKit; pure NumPy -- unit-testable)
# --------------------------------------------------------------------------- #
def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum()))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c (degrees) at the vertex b."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return float("nan")
    cos = float(np.dot(v1, v2) / n)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def nac_from_frames(frames: Sequence[np.ndarray], donor_heavy: int,
                    transfer: int, acceptor: int, spec: ReactiveSpec) -> NACResult:
    """Per-frame transfer distance (transferring atom -> acceptor) and
    donor_heavy--transfer--acceptor angle; a frame is reaction-competent when
    distance <= spec.distance_max AND angle >= spec.angle_min. ``frames`` are
    (N,3) coordinate arrays in Angstrom; the three indices are into that frame.
    """
    dists: List[float] = []
    angles: List[float] = []
    hits = 0
    for fr in frames:
        d = _dist(fr[transfer], fr[acceptor])
        ang = _angle(fr[donor_heavy], fr[transfer], fr[acceptor])
        dists.append(round(d, 3))
        angles.append(round(ang, 1))
        if d <= spec.distance_max and ang >= spec.angle_min:
            hits += 1
    n = len(dists)
    return NACResult(
        occupancy=round(hits / n, 4) if n else 0.0,
        n_frames=n, n_reactive=hits,
        distance_mean=round(float(np.mean(dists)), 3) if dists else float("nan"),
        distance_min=round(float(np.min(dists)), 3) if dists else float("nan"),
        angle_mean=round(float(np.mean(angles)), 1) if angles else float("nan"),
        distances=dists, angles=angles,
        atoms={"donor_heavy": donor_heavy, "transfer": transfer,
               "acceptor": acceptor},
        label=spec.label,
    )


# the FDH formate -> NADP-C4 hydride transfer, as a ready ReactiveSpec. Lives
# here only as a documented DEFAULT/example; production configs pass their own.
FDH_HYDRIDE = ReactiveSpec(
    donor_smarts="[CX3H1](=O)[O-]",          # formate C-H (the hydride donor)
    acceptor_smarts="[cH1]([cH0]C(=O)[NX3])[cH1]",  # nicotinamide C4 (match atom 0)
    donor_idx=0, acceptor_idx=0, transfer_is_h=True,
    distance_max=3.5, angle_min=150.0, label="formate_hydride_to_NADP_C4",
)
