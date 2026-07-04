"""Productive (near-attack) placement of a free co-substrate for the NAC screen (NAC-4).

The catalytic NAC measures whether the active site can MAINTAIN the reactive geometry,
but that is only meaningful if the co-substrate STARTS in a reactive pose. The structure
predictor (Boltz) places the free formate essentially at random -- in the restrained-NAC
run 9 / 12 candidates were ``skipped_bad_initial_cosubstrate_pose`` because the formate
began mis-oriented or far from the acceptor.

This module builds the hydride-transfer geometry EXPLICITLY relative to the cofactor's
acceptor atom (nicotinamide C4): the transferring atom sits ``nac_distance`` Å off the
ring face, the donor heavy atom is collinear beyond it (donor--H...acceptor ~linear), and
the rest of the donor completes its planar sp2 centre. Which ring FACE is chosen can be
biased toward a point (the catalytic carboxylate-binding residues), so the formate lands
on the catalytic face rather than against the cofactor's own bulk. Pure geometry (NumPy +
RDKit topology); the MD + retention restraint then relax any residual clash.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Ideal hydride-transfer NAC geometry (Å / unitless). The transferring H approaches the
# nicotinamide C4 face at ~2.7-2.8 Å (near-attack, below the 3.0-3.5 Å reactive cutoff),
# collinear with the donor C so the C-H...C4 angle starts ~180° (well above any gate).
_NAC_DISTANCE = 2.8
_CH_BOND = 1.1
_CO_BOND = 1.25
_SIN120 = np.sqrt(3.0) / 2.0


def _perpendicular(n: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to ``n``."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, seed)
    return u / np.linalg.norm(u)


def ring_normal(coords: np.ndarray, ring_idx: Sequence[int]) -> np.ndarray:
    """Unit normal of the near-planar ring spanned by ``ring_idx`` (SVD: the smallest
    singular vector of the centred ring points)."""
    pts = coords[list(ring_idx)]
    _, _, vh = np.linalg.svd(pts - pts.mean(axis=0))
    n = vh[-1]
    return n / np.linalg.norm(n)


def acceptor_ring(mol, acceptor_idx: int) -> Sequence[int]:
    """The 6-membered ring containing the acceptor (the nicotinamide pyridinium)."""
    for ring in mol.GetRingInfo().AtomRings():
        if acceptor_idx in ring and len(ring) == 6:
            return ring
    raise ValueError(f"acceptor atom {acceptor_idx} is not in a 6-membered ring")


def productive_geometry(acceptor_pos: np.ndarray, normal: np.ndarray,
                        face_toward: "Optional[np.ndarray]" = None,
                        nac_distance: float = _NAC_DISTANCE):
    """Return ``(p_H, p_C, p_O1, p_O2)`` for a formate placed in the near-attack
    geometry off the ring face along ``normal``. If ``face_toward`` is given, the face
    pointing toward it is used (e.g. the catalytic carboxylate-binding residues)."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    if face_toward is not None and np.dot(np.asarray(face_toward, float) - acceptor_pos, n) < 0:
        n = -n
    u = _perpendicular(n)
    p_H = acceptor_pos + nac_distance * n           # transferring H near the C4 face
    p_C = p_H + _CH_BOND * n                         # donor C collinear beyond H (~linear)
    p_O1 = p_C + _CO_BOND * (0.5 * n + _SIN120 * u)  # planar sp2 carboxylate
    p_O2 = p_C + _CO_BOND * (0.5 * n - _SIN120 * u)
    return p_H, p_C, p_O1, p_O2


def place_cosubstrate(formate_coords: np.ndarray, donor_heavy: int, transfer: int,
                      acceptor_pos: np.ndarray, normal: np.ndarray,
                      face_toward: "Optional[np.ndarray]" = None,
                      nac_distance: float = _NAC_DISTANCE) -> np.ndarray:
    """New coordinates for the formate (same atom order as ``formate_coords``) placed in
    the productive geometry. ``donor_heavy`` / ``transfer`` are the formate's C and H
    indices; the remaining heavy atoms are its two carboxylate O's."""
    p_H, p_C, p_O1, p_O2 = productive_geometry(
        np.asarray(acceptor_pos, float), normal, face_toward, nac_distance)
    others = [i for i in range(len(formate_coords)) if i not in (donor_heavy, transfer)]
    if len(others) < 2:
        raise ValueError("formate must have >=2 non-donor atoms (the carboxylate O's)")
    out = np.array(formate_coords, float, copy=True)
    out[donor_heavy] = p_C
    out[transfer] = p_H
    out[others[0]] = p_O1
    out[others[1]] = p_O2
    # any extra atoms (shouldn't exist for formate) keep their original coords
    return out


def place_formate_for_nac(nadp_mol, formate_mol, donor_spec, acceptor_spec,
                          catalytic_coords: "Optional[np.ndarray]" = None,
                          nac_distance: float = _NAC_DISTANCE):
    """Reposition ``formate_mol``'s conformer into the near-attack geometry off
    ``nadp_mol``'s acceptor face, biased toward the catalytic residues. Identifies the
    acceptor / donor via the same SMARTS the NAC screen uses. Returns the new formate
    coordinates (in formate_mol's atom order), or ``None`` if the reacting atoms can't
    be resolved (the caller then leaves the predictor's pose untouched)."""
    from evoliez.md.nac import identify_acceptor, identify_donor

    acc = identify_acceptor(nadp_mol, acceptor_spec)
    don = identify_donor(formate_mol, donor_spec)
    if acc is None or don is None:
        return None
    nadp_coords = np.asarray(nadp_mol.GetConformer().GetPositions(), float)
    acc_pos = nadp_coords[acc]
    normal = ring_normal(nadp_coords, acceptor_ring(nadp_mol, acc))
    if catalytic_coords is not None and len(catalytic_coords):
        face = np.asarray(catalytic_coords, float).mean(axis=0)
    else:
        # No catalytic anchor: use the EXPOSED ring face -- away from the cofactor's own
        # ribose/adenine bulk, which is the side the substrate pocket / catalytic residues
        # occupy -- so the formate doesn't land against NADP itself.
        face = acc_pos + (acc_pos - nadp_coords.mean(axis=0))
    formate_coords = np.asarray(formate_mol.GetConformer().GetPositions(), float)
    return place_cosubstrate(formate_coords, don["heavy"], don["transfer"],
                             acc_pos, normal, face_toward=face, nac_distance=nac_distance)


def place_donor_at_acceptor(acceptor_mol, donor_mol, donor_spec, acceptor_spec,
                            nac_distance: float = 3.2):
    """Near-attack placement for an ADENYLATION-style reaction where the DONOR molecule
    (e.g. the DESIGN ligand 3-HP, carboxylate O) attacks a NON-RING acceptor atom (e.g.
    the ATP alpha-P on a cofactor) — the inverse of the FDH formate case. Rigid-translates
    ``donor_mol`` so its donor atom sits ``nac_distance`` Å off the acceptor along the
    in-line axis (anti to the acceptor's leaving group — the bridging O toward the beta-P).
    Returns new donor coords (donor_mol atom order), or None if atoms can't be resolved."""
    from evoliez.md.nac import identify_acceptor, identify_donor
    acc = identify_acceptor(acceptor_mol, acceptor_spec)
    don = identify_donor(donor_mol, donor_spec)
    if acc is None or don is None:
        return None
    ac = np.asarray(acceptor_mol.GetConformer().GetPositions(), float)
    acc_pos = ac[acc]
    a_atom = acceptor_mol.GetAtomWithIdx(acc)
    ld = None
    for nb in a_atom.GetNeighbors():                 # leaving group = bridging O (acc-O-P')
        if nb.GetSymbol() == "O" and any(
                nn.GetSymbol() == "P" and nn.GetIdx() != acc for nn in nb.GetNeighbors()):
            ld = acc_pos - ac[nb.GetIdx()]; break
    if ld is None:                                   # fallback: exposed face (away from acceptor bulk)
        ld = acc_pos - ac.mean(axis=0)
    ld = ld / np.linalg.norm(ld)
    target = acc_pos + ld * float(nac_distance)
    dc = np.asarray(donor_mol.GetConformer().GetPositions(), float)
    d_idx = don.get("transfer", don["heavy"])        # for O-attack (transfer_is_h=False) heavy==transfer
    return dc + (target - dc[d_idx])
