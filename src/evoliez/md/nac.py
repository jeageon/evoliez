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
    # NAC-validity gates (decouple placement / retention / reactive geometry, so an
    # unrestrained co-substrate that simply diffuses away is reported as INVALID rather
    # than as "low reactivity"). Lit. hydride-transfer NAC: dist<=3.0 Å, angle 132-180°.
    placement_distance_max: float = 4.0   # Å, frame-0 transfer->acceptor (Michaelis-like start)
    placement_angle_min: float = 0.0      # deg, frame-0 angle gate (0 = off by default)
    retention_distance_max: float = 6.0   # Å, "still in the active-site pocket" cutoff
    retention_min_fraction: float = 0.8   # require this fraction of frames retained for a valid NAC


# NAC validity states (decoupled from md_status: the MD may run perfectly while the
# NAC is uninterpretable because the co-substrate never sampled the reactive site).
NAC_VALID = "valid_unrestrained"
NAC_VALID_RESTRAINED = "valid_restrained_retention_screen"
NAC_SKIP_PLACEMENT = "skipped_bad_initial_cosubstrate_pose"
NAC_INVALID_DIFFUSED = "invalid_cosubstrate_diffused"
NAC_SKIP_NO_ATOMS = "skipped_missing_reactive_atoms"
_NAC_CONSUMABLE = frozenset({NAC_VALID, NAC_VALID_RESTRAINED})


def nac_status_is_valid(status: "Optional[str]") -> bool:
    """Whether a nac_status may feed ranking / ΔNAC (s11). Only states where the
    co-substrate was actually retained in a reactive arrangement qualify."""
    return status in _NAC_CONSUMABLE


@dataclass
class NACResult:
    occupancy: float = 0.0        # reaction-competent frames / ALL frames
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
    # validity accounting (NAC-1/2): separate placement, retention, reactive geometry
    status: str = NAC_VALID
    distance_initial: float = float("nan")
    angle_initial: float = float("nan")
    retention_fraction: float = float("nan")   # frames with the co-substrate still in the pocket
    n_retained: int = 0
    occupancy_retained: float = float("nan")   # reactive frames / RETAINED frames
    escape: bool = False
    restrained: bool = False

    @property
    def occupancy_or_none(self) -> "Optional[float]":
        """The occupancy ONLY when the NAC is interpretable, else None -- this is
        what ranking / ΔNAC must read (never the raw occupancy of a diffused pose)."""
        return self.occupancy if nac_status_is_valid(self.status) else None

    def to_json(self) -> Dict[str, object]:
        return {
            "label": self.label, "nac_status": self.status,
            "nac_occupancy": self.occupancy_or_none,
            "nac_occupancy_raw": self.occupancy,
            "occupancy_retained": self.occupancy_retained,
            "n_frames": self.n_frames, "n_reactive": self.n_reactive,
            "n_retained": self.n_retained, "retention_fraction": self.retention_fraction,
            "escape": self.escape, "restrained": self.restrained,
            "distance_initial": self.distance_initial, "distance_min": self.distance_min,
            "distance_mean": self.distance_mean,
            "angle_initial": self.angle_initial, "angle_mean": self.angle_mean,
            "atoms": self.atoms, "note": self.note,
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


def identify_leaving(mol, acceptor_idx: int) -> Optional[int]:
    """For a NON-hydride O->P attack, the leaving group is the BRIDGING oxygen on the acceptor
    P that ALSO bonds a SECOND P (e.g. the ATP alpha-beta bridging O — the bond that breaks in
    adenylation). Purely TOPOLOGICAL (coordinate-free), matching
    ``cosubstrate_placement.place_donor_at_acceptor``; used only to define the read-only
    O_nuc--Palpha--O_leaving angle. None if the acceptor is not a phosphorus with such a
    bridging O (e.g. the FDH hydride/aromatic-C acceptor), so the hydride path is untouched."""
    a = mol.GetAtomWithIdx(acceptor_idx)
    for nb in a.GetNeighbors():
        if nb.GetSymbol() == "O" and any(
                nn.GetSymbol() == "P" and nn.GetIdx() != acceptor_idx
                for nn in nb.GetNeighbors()):
            return nb.GetIdx()
    return None


def resolve_reactive_indices(
    mol_blocks: "Sequence[tuple]", spec: ReactiveSpec
) -> Optional[Dict[str, int]]:
    """Map the RDKit donor/acceptor atoms onto GLOBAL trajectory atom indices.

    ``mol_blocks`` is one entry per small molecule actually placed in the MD
    system, in the order it was added: ``(rdkit_mol_with_Hs, global_indices)``
    where ``global_indices[i]`` is the trajectory atom index of atom ``i`` of
    that RDKit mol (the OpenFF round-trip preserves atom order, so this is just
    the contiguous topology block the molecule occupies).

    Donor and acceptor are searched INDEPENDENTLY across every molecule, so the
    formate hydride donor and the NADP-C4 acceptor are correctly found even
    though they are SEPARATE residues. Returns ``{donor_heavy, transfer,
    acceptor}`` in global indices, or ``None`` if either side is absent (an
    honest "reaction partners not both present" -> NAC is skipped, never faked).
    """
    donor = donor_blk = None
    acceptor = acceptor_blk = acceptor_rd = None
    for rd, gidx in mol_blocks:
        if donor is None:
            d = identify_donor(rd, spec)
            if d is not None and max(d.values()) < len(gidx):
                donor, donor_blk = d, gidx
        if acceptor is None:
            a = identify_acceptor(rd, spec)
            if a is not None and a < len(gidx):
                acceptor, acceptor_blk, acceptor_rd = a, gidx, rd
    if donor is None or acceptor is None:
        return None
    out = {
        "donor_heavy": int(donor_blk[donor["heavy"]]),
        "transfer": int(donor_blk[donor["transfer"]]),
        "acceptor": int(acceptor_blk[acceptor]),
    }
    # O->P attack: also resolve the leaving-group O on the acceptor P (same mol block) so the
    # near-attack ANGLE is the real O_nuc-Palpha-O_leaving (V5-1) instead of the degenerate
    # donor==transfer NaN. Absent for the hydride case -> the 3-atom path is unchanged.
    if not spec.transfer_is_h:
        lv = identify_leaving(acceptor_rd, acceptor)
        if lv is not None and lv < len(acceptor_blk):
            out["leaving"] = int(acceptor_blk[lv])
    return out


def nac_from_subframes(
    subframes: "Sequence[np.ndarray]", spec: ReactiveSpec,
    atoms: "Optional[Dict[str, int]]" = None, restrained: bool = False,
    initial_subframe: "Optional[np.ndarray]" = None,
) -> NACResult:
    """NAC from pre-extracted 3-atom frames. Each ``subframes`` element is a
    (3,3) array in Angstrom whose rows are ``[donor_heavy, transfer, acceptor]``
    -- the engine collects only these three atoms per MD frame rather than the
    whole system. ``atoms`` (the resolved GLOBAL indices) is recorded for
    provenance. ``restrained`` flags a retention-restrained run (the valid status
    becomes ``valid_restrained_retention_screen``). ``initial_subframe`` may be
    supplied for the placement gate only; it is not counted in occupancy."""
    # A 4-row subframe carries the O->P leaving atom at row 3 (the engine appends it when
    # resolve_reactive_indices returns a "leaving" index); a 3-row subframe is the hydride case.
    _lv = 3 if (len(subframes) and len(subframes[0]) > 3) else None
    res = nac_from_frames(
        subframes, 0, 1, 2, spec, restrained=restrained,
        initial_frame=initial_subframe, leaving=_lv,
    )
    if atoms:
        res.atoms = dict(atoms)
    return res


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
                    transfer: int, acceptor: int, spec: ReactiveSpec,
                    restrained: bool = False,
                    initial_frame: "Optional[np.ndarray]" = None,
                    leaving: "Optional[int]" = None) -> NACResult:
    """Per-frame transfer distance (transferring atom -> acceptor) and near-attack angle;
    a frame is reaction-competent when distance <= spec.distance_max AND angle >= spec.angle_min.
    ``frames`` are (N,3) coordinate arrays in Angstrom; the indices are into that frame.
    ``initial_frame`` overrides the placement gate frame but is not counted in occupancy.

    ANGLE (ROADMAP_V5 V5-1): for a hydride transfer the angle is donor_heavy--transfer(H)--
    acceptor. For a NON-hydride O->P attack (``spec.transfer_is_h`` False) the transferring
    atom IS the donor heavy atom, so that 3-point angle is degenerate (vertex == first point ->
    NaN, forcing occupancy 0 for EVERY frame). When a ``leaving`` atom is supplied the angle is
    instead the real in-line nucleophilic O_nuc--acceptor(Palpha)--O_leaving (vertex at the
    acceptor). Distance is always the transferring-atom -> acceptor separation (O_nuc -> Palpha
    for the O-attack), which is well-defined either way.
    """
    _oatk = leaving is not None and not spec.transfer_is_h

    def _ang(fr: np.ndarray) -> float:
        if _oatk:
            return _angle(fr[donor_heavy], fr[acceptor], fr[leaving])
        return _angle(fr[donor_heavy], fr[transfer], fr[acceptor])

    dists: List[float] = []
    angles: List[float] = []
    hits = 0
    retained_hits = 0
    n_retained = 0
    for fr in frames:
        d = _dist(fr[transfer], fr[acceptor])
        ang = _ang(fr)
        dists.append(round(d, 3))
        angles.append(round(ang, 1))
        reactive = d <= spec.distance_max and ang >= spec.angle_min
        if reactive:
            hits += 1
        if d <= spec.retention_distance_max:           # co-substrate still in the pocket
            n_retained += 1
            if reactive:
                retained_hits += 1
    n = len(dists)
    if initial_frame is not None:
        initial_distance = round(_dist(initial_frame[transfer],
                                       initial_frame[acceptor]), 3)
        initial_angle = round(_ang(initial_frame), 1)
    else:
        initial_distance = dists[0] if dists else float("nan")
        initial_angle = angles[0] if angles else float("nan")
    res = NACResult(
        occupancy=round(hits / n, 4) if n else 0.0,
        n_frames=n, n_reactive=hits,
        distance_mean=round(float(np.mean(dists)), 3) if dists else float("nan"),
        distance_min=round(float(np.min(dists)), 3) if dists else float("nan"),
        angle_mean=round(float(np.mean(angles)), 1) if angles else float("nan"),
        distances=dists, angles=angles,
        atoms={"donor_heavy": donor_heavy, "transfer": transfer,
               "acceptor": acceptor},
        label=spec.label,
        distance_initial=initial_distance,
        angle_initial=initial_angle,
        n_retained=n_retained,
        retention_fraction=round(n_retained / n, 3) if n else float("nan"),
        occupancy_retained=round(retained_hits / n_retained, 4) if n_retained else float("nan"),
    )
    # Gate the NAC into a validity STATUS (decouple placement / retention / geometry).
    res.restrained = bool(restrained)
    res.escape = bool(n and res.retention_fraction < spec.retention_min_fraction)
    if not n:
        res.status = NAC_SKIP_NO_ATOMS
    elif (res.distance_initial > spec.placement_distance_max
          or res.angle_initial < spec.placement_angle_min):
        # never started in a Michaelis-like pose (e.g. formate placed 8 Å away)
        res.status = NAC_SKIP_PLACEMENT
        res.note = (f"initial pose out of range (d0={res.distance_initial} Å, "
                    f"a0={res.angle_initial}°)")
    elif res.retention_fraction < spec.retention_min_fraction:
        # co-substrate diffused out of the pocket -> occupancy reflects diffusion, not geometry
        res.status = NAC_INVALID_DIFFUSED
        res.note = (f"co-substrate diffused (retained {res.retention_fraction} < "
                    f"{spec.retention_min_fraction}); occupancy not interpretable")
    else:
        res.status = NAC_VALID_RESTRAINED if restrained else NAC_VALID
    return res


# the FDH formate -> NADP-C4 hydride transfer, as a ready ReactiveSpec. Lives
# here only as a documented DEFAULT/example; production configs pass their own.
FDH_HYDRIDE = ReactiveSpec(
    donor_smarts="[CX3H1](=O)[O-]",          # formate C-H (the hydride donor)
    acceptor_smarts="[cH1]([cH0]C(=O)[NX3])[cH1]",  # nicotinamide C4 (match atom 0)
    donor_idx=0, acceptor_idx=0, transfer_is_h=True,
    distance_max=3.5, angle_min=150.0, label="formate_hydride_to_NADP_C4",
)
