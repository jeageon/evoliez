"""Catalytic-metal (Mg2+) placement for the adenylation near-attack screen (ROADMAP_V5 V5-2).

The adenylation reaction is stabilised by a Mg2+ that BRIDGES the substrate carboxylate O⁻ and
the ATP alpha-phosphate O⁻; without it the two anions repel and the reactive O drifts out
(~3.2 Å -> ~4.9 Å in the CAR run), so dynamic NAC reads 0 for a purely electrostatic reason,
not a biological one. Mg2+ is a bare +2 ion, so it is placed at the STRUCTURE level (an ``MG``
HETATM the Amber ion parameters handle) rather than through the OpenFF small-molecule path.

Two pure, unit-testable pieces (the OpenMM System build just consumes the resulting PDB):
  * ``bridging_metal_position`` — WHERE the ion goes. DETERMINISTIC: WT and every mutant MUST get
    the SAME Mg placement or ΔNAC = mutant - WT is confounded, so the perpendicular side is chosen
    from a fixed reference, never from anything mutant-specific.
  * ``mg_hetatm_line`` — the PDB record to inject.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

MG_COORD_DIST = 2.1   # Å, typical Mg2+---O coordination distance


def bridging_metal_position(
    nucleophile_o: Sequence[float],
    phosphate_o: Sequence[float],
    coord_dist: float = MG_COORD_DIST,
    reference: "Optional[Sequence[float]]" = None,
) -> np.ndarray:
    """Position a bridging metal roughly ``coord_dist`` from BOTH the nucleophile O and the
    phosphate O (so it screens the two negative charges). Returns an (3,) array.

    Geometry: on the perpendicular bisector of the two O atoms. If they are closer than
    ``2*coord_dist`` the ion sits off the midline by ``h = sqrt(coord_dist^2 - d^2)`` (d =
    half the O–O distance); if they are farther apart than the ion can bridge, it falls back to
    the midpoint (coordination stretched — the caller/report should flag it). ``reference`` (e.g.
    the protein/pocket centroid) picks the perpendicular SIDE deterministically so the placement
    is identical for WT and every mutant; absent, a fixed global axis is used.
    """
    a = np.asarray(nucleophile_o, float)
    b = np.asarray(phosphate_o, float)
    mid = (a + b) / 2.0
    ab = b - a
    dist = float(np.linalg.norm(ab))
    if dist < 1e-6:
        return mid                                  # degenerate: atoms coincide
    d = dist / 2.0
    if d >= coord_dist:
        return mid                                  # can't bridge at coord_dist -> best effort
    axis = ab / dist
    # a perpendicular direction, chosen DETERMINISTICALLY
    if reference is not None:
        r = np.asarray(reference, float) - mid
        perp = r - np.dot(r, axis) * axis           # component of (reference-mid) perp to axis
        # place the ion on the side AWAY from the reference (toward the open/solvent face)
        perp = -perp
    else:
        # fixed global axis (z, then x if degenerate) -> reproducible with no reference
        ref = np.array([0.0, 0.0, 1.0])
        perp = ref - np.dot(ref, axis) * axis
        if np.linalg.norm(perp) < 1e-6:
            ref = np.array([1.0, 0.0, 0.0])
            perp = ref - np.dot(ref, axis) * axis
    n = float(np.linalg.norm(perp))
    if n < 1e-6:                                     # reference collinear with axis -> midpoint
        return mid
    perp = perp / n
    h = float(np.sqrt(max(0.0, coord_dist * coord_dist - d * d)))
    return mid + h * perp


def mg_hetatm_line(position: Sequence[float], serial: int = 9999,
                   res_seq: int = 999, chain: str = "M", element: str = "MG") -> str:
    """A single PDB HETATM record for a Mg2+ ion at ``position`` (Å). Uses the ``MG`` residue +
    element so the Amber ion parameters recognise it. Columns follow the fixed PDB format."""
    x, y, z = (float(v) for v in position)
    name = "MG"
    return (
        f"HETATM{serial:>5d} {name:<4s}{'MG':>3s} {chain:1s}{res_seq:>4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2s}2+"
    )


def insert_mg_into_pdb(pdb_text: str, position: Sequence[float], **kw) -> str:
    """Insert an MG HETATM before the final END/terminal record of ``pdb_text``. Deterministic
    (fixed serial/res unless overridden) so WT and mutant builds are byte-comparable."""
    line = mg_hetatm_line(position, **kw)
    lines = pdb_text.splitlines()
    # insert before a trailing END/ENDMDL if present, else append
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(("END", "ENDMDL", "CONECT", "MASTER")):
            lines.insert(i, line)
            return "\n".join(lines) + "\n"
    lines.append(line)
    return "\n".join(lines) + "\n"
