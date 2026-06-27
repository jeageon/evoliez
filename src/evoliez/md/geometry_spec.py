"""Generic, config-driven functional-geometry layer.

FDH's hydride-transfer NAC (formate->NADP C4) is ONE instance of a functional
geometry. In general the user declares a LIST of geometry terms — each a simple,
composable primitive evaluated per MD frame:

  - ``distance``: atom A — atom B within [min, max] Å  (e.g. donor-acceptor,
    a salt bridge, a proton-relay hop, a metal-ligand bond)
  - ``angle``:   atom A — atom B — atom C within [min, max]°  (e.g. attack angle,
    a catalytic-dyad geometry, an H-bond angle)

Richer functional states are COMPOSITIONS of these terms: a near-attack
conformation = one distance + one angle; a catalytic triad = a few distances +
angles; a proton relay = a chain of distance terms; metal coordination = several
distance terms to the metal. Each term reports an occupancy (fraction of frames
satisfying it); the spec's overall satisfaction is the (weighted) min/mean of its
terms, so a functional state holds only when ALL its geometry holds.

Atoms are selected by SMARTS against the ligand/co-substrate molecules (the same
mechanism as md.nac), so nothing is FDH-specific. Pure NumPy + RDKit SMARTS;
unit-testable on synthetic frames. (Selectors into PROTEIN atoms — catalytic
residues, a coordinating His — are a planned extension that needs the topology;
the term schema already carries the fields.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .nac import _angle, _dist

DISTANCE = "distance"
ANGLE = "angle"


@dataclass
class GeometryTerm:
    """One functional-geometry primitive. ``kind`` = 'distance' | 'angle'.

    Atoms are the ``*_idx``-th atom of the first match of ``*_smarts`` across the
    ligand molecules. A distance term uses a,b; an angle term uses a,b,c (vertex b)."""
    kind: str
    label: str = "term"
    a_smarts: str = ""
    a_idx: int = 0
    b_smarts: str = ""
    b_idx: int = 0
    c_smarts: str = ""
    c_idx: int = 0
    distance_min: float = 0.0
    distance_max: float = 3.5
    angle_min: float = 0.0
    angle_max: float = 180.0
    weight: float = 1.0


@dataclass
class GeometryTermResult:
    label: str
    kind: str
    status: str = "ok"            # ok | skipped_missing_atoms | error
    occupancy: float = 0.0        # fraction of frames satisfying the term
    mean: Optional[float] = None
    min: Optional[float] = None
    n_frames: int = 0
    atoms: dict = field(default_factory=dict)
    note: str = ""

    def to_json(self) -> dict:
        return {
            "label": self.label, "kind": self.kind, "status": self.status,
            "occupancy": round(self.occupancy, 4),
            "mean": (round(self.mean, 3) if self.mean is not None else None),
            "min": (round(self.min, 3) if self.min is not None else None),
            "n_frames": self.n_frames, "atoms": self.atoms, "note": self.note,
        }


def _resolve_global(mol_blocks, smarts: str, idx: int) -> Optional[int]:
    """Global trajectory index of the idx-th atom of the first SMARTS match across
    the ligand molecules. mol_blocks = [(rdkit_mol_with_Hs, global_indices), ...]."""
    if not smarts:
        return None
    from .nac import _first_match_atom
    for mol, gidx in mol_blocks:
        loc = _first_match_atom(mol, smarts, idx)
        if loc is not None and 0 <= loc < len(gidx):
            return int(gidx[loc])
    return None


def evaluate_geometry_term(
    frames: Sequence[np.ndarray], mol_blocks, term: GeometryTerm,
) -> GeometryTermResult:
    a = _resolve_global(mol_blocks, term.a_smarts, term.a_idx)
    b = _resolve_global(mol_blocks, term.b_smarts, term.b_idx)
    atoms = {"a": a, "b": b}
    if a is None or b is None:
        return GeometryTermResult(term.label, term.kind,
                                  status="skipped_missing_atoms", atoms=atoms,
                                  note="a/b SMARTS not matched")
    c = None
    if term.kind == ANGLE:
        c = _resolve_global(mol_blocks, term.c_smarts, term.c_idx)
        atoms["c"] = c
        if c is None:
            return GeometryTermResult(term.label, term.kind,
                                      status="skipped_missing_atoms", atoms=atoms,
                                      note="c SMARTS not matched")
    vals: List[float] = []
    for fr in frames:
        fr = np.asarray(fr, float)
        if term.kind == DISTANCE:
            vals.append(_dist(fr[a], fr[b]))
        elif term.kind == ANGLE:
            vals.append(_angle(fr[a], fr[b], fr[c]))
        else:
            return GeometryTermResult(term.label, term.kind, status="error",
                                      atoms=atoms, note=f"unknown kind {term.kind!r}")
    if not vals:
        return GeometryTermResult(term.label, term.kind, status="ok", atoms=atoms,
                                  n_frames=0, note="no frames")
    if term.kind == DISTANCE:
        ok = [term.distance_min <= v <= term.distance_max for v in vals]
    else:
        ok = [term.angle_min <= v <= term.angle_max for v in vals]
    return GeometryTermResult(
        term.label, term.kind, status="ok",
        occupancy=sum(ok) / len(vals), mean=float(np.mean(vals)),
        min=float(np.min(vals)), n_frames=len(vals), atoms=atoms)


def evaluate_geometry_spec(
    frames: Sequence[np.ndarray], mol_blocks, terms: Sequence[GeometryTerm],
) -> List[GeometryTermResult]:
    return [evaluate_geometry_term(frames, mol_blocks, t) for t in terms]


def spec_satisfaction(results: Sequence[GeometryTermResult], terms: Sequence[GeometryTerm]) -> float:
    """Overall functional-state occupancy = weighted MIN of the valid terms'
    occupancies (a functional state holds only when ALL its geometry holds). 0.0 if
    any required term is unresolved (honest: a missing partner is not satisfaction)."""
    valid = [(r, t) for r, t in zip(results, terms) if r.status == "ok"]
    if len(valid) != len(terms) or not valid:
        return 0.0
    return min(r.occupancy for r, _ in valid)
