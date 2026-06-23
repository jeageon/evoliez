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
