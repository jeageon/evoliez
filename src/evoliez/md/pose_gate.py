"""Reference-like pose gate (role-aware) — the structural anchor of EvoLiEZ
validation.

For each non-protein ligand it decides whether a TEST structure preserves the
REFERENCE binding pose (``reference_like``), has moved into a different pose
(``alternative_pose``), or has left the site (``displaced``):

  - pocket-aligned pose RMSD : superpose the pocket Cα onto the reference, then
    measure the ligand heavy-atom RMSD — "is the ligand in the same place?"
  - internal-shape RMSD      : superpose the ligand onto its own reference copy —
    "is the ligand folded the same way?" (alignment-free of the pocket)

This separates the *mutation effect* from *pose-search noise*: a mutant whose
cofactor pose is not reference_like is tagged ``alternative_pose`` (an alternative
binding-mode HYPOTHESIS), never silently accepted as an "improved mutant".

Role-aware: thresholds differ by ligand role (a metal must coordinate tightly; a
substrate may sample more). Pure NumPy + a minimal PDB reader — no MD-engine
dependency, unit-testable on synthetic coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REFERENCE_LIKE = "reference_like"
ALTERNATIVE_POSE = "alternative_pose"
DISPLACED = "displaced"
SKIPPED = "skipped_no_correspondence"


@dataclass(frozen=True)
class PoseThresholds:
    """Role-specific gate cutoffs (Å)."""
    pocket_rmsd_max: float = 5.0     # pocket-aligned ligand RMSD for reference_like
    internal_rmsd_max: float = 2.0   # ligand internal-shape RMSD for reference_like
    displaced_rmsd: float = 10.0     # beyond this = displaced (not even alternative)


# Defaults per ligand role. Tight for metals/cofactors (a fixed functional state),
# looser for substrate/TS proxies (legitimately mobile before reaction).
ROLE_THRESHOLDS: Dict[str, PoseThresholds] = {
    "design_ligand": PoseThresholds(3.0, 2.0, 8.0),
    "cofactor": PoseThresholds(3.0, 2.0, 8.0),
    "substrate": PoseThresholds(4.0, 2.5, 10.0),
    "transition_state_proxy": PoseThresholds(4.0, 2.5, 10.0),
    "product": PoseThresholds(4.0, 2.5, 10.0),
    "metal": PoseThresholds(1.5, 1.0, 4.0),
    "context": PoseThresholds(2.5, 1.5, 6.0),
    "ion": PoseThresholds(2.5, 1.5, 6.0),
    "other": PoseThresholds(5.0, 2.5, 10.0),
}


def thresholds_for(role: Optional[str]) -> PoseThresholds:
    return ROLE_THRESHOLDS.get((role or "other").lower(), ROLE_THRESHOLDS["other"])


# --------------------------------------------------------------------- geometry
def kabsch_apply(P: np.ndarray, Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Rigid-align P onto Q (Kabsch, no scaling); apply the same transform to X."""
    P = np.asarray(P, float); Q = np.asarray(Q, float); X = np.asarray(X, float)
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (X - Pc) @ R.T + Qc


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


@dataclass
class PoseGateResult:
    ligand_id: str
    role: str
    status: str
    pocket_pose_rmsd: Optional[float]
    internal_rmsd: Optional[float]
    n_ligand_atoms: int
    n_pocket_ca: int
    note: str = ""

    @property
    def reference_like(self) -> bool:
        return self.status == REFERENCE_LIKE

    def to_json(self) -> dict:
        return {
            "ligand_id": self.ligand_id, "role": self.role, "status": self.status,
            "pocket_pose_rmsd": (round(self.pocket_pose_rmsd, 3)
                                 if self.pocket_pose_rmsd is not None else None),
            "internal_rmsd": (round(self.internal_rmsd, 3)
                              if self.internal_rmsd is not None else None),
            "n_ligand_atoms": self.n_ligand_atoms, "n_pocket_ca": self.n_pocket_ca,
            "note": self.note,
        }


def evaluate_pose(
    ref_pocket_ca: np.ndarray, test_pocket_ca: np.ndarray,
    ref_lig: np.ndarray, test_lig: np.ndarray,
    *, ligand_id: str = "ligand", role: str = "other",
    thresholds: Optional[PoseThresholds] = None,
) -> PoseGateResult:
    """Core gate on coordinate arrays (caller guarantees atom correspondence:
    pocket Cα residue-for-residue, ligand atom-for-atom)."""
    thr = thresholds or thresholds_for(role)
    ref_pocket_ca = np.asarray(ref_pocket_ca, float)
    test_pocket_ca = np.asarray(test_pocket_ca, float)
    ref_lig = np.asarray(ref_lig, float)
    test_lig = np.asarray(test_lig, float)
    if (len(ref_pocket_ca) < 3 or len(ref_pocket_ca) != len(test_pocket_ca)
            or len(ref_lig) == 0 or len(ref_lig) != len(test_lig)):
        return PoseGateResult(ligand_id, role, SKIPPED, None, None,
                              len(ref_lig), len(ref_pocket_ca),
                              note="atom-correspondence mismatch")
    pose = rmsd(kabsch_apply(test_pocket_ca, ref_pocket_ca, test_lig), ref_lig)
    internal = rmsd(kabsch_apply(test_lig, ref_lig, test_lig), ref_lig)
    if pose > thr.displaced_rmsd:
        status = DISPLACED
    elif pose <= thr.pocket_rmsd_max and internal <= thr.internal_rmsd_max:
        status = REFERENCE_LIKE
    else:
        status = ALTERNATIVE_POSE
    return PoseGateResult(ligand_id, role, status, pose, internal,
                          len(ref_lig), len(ref_pocket_ca))


# ----------------------------------------------------------------- PDB plumbing
@dataclass
class _Atom:
    record: str; name: str; resname: str; chain: str
    resseq: int; element: str; xyz: Tuple[float, float, float]


def read_pdb_atoms(path) -> List[_Atom]:
    """Minimal ATOM/HETATM reader (no external deps)."""
    out: List[_Atom] = []
    for ln in Path(path).read_text().splitlines():
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        try:
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError:
            continue
        elem = ln[76:78].strip() or ln[12:16].strip()[:1]
        try:
            resseq = int(ln[22:26])
        except ValueError:
            resseq = 0
        out.append(_Atom(ln[:6].strip(), ln[12:16].strip(), ln[17:20].strip(),
                         ln[21:22].strip(), resseq, elem.upper(), (x, y, z)))
    return out


def _ca(atoms: Sequence[_Atom]) -> Dict[Tuple[str, int], _Atom]:
    return {(a.chain, a.resseq): a for a in atoms
            if a.record == "ATOM" and a.name == "CA"}


def _het_residues(atoms: Sequence[_Atom], resname: Optional[str] = None,
                  heavy_only: bool = True) -> List[List[_Atom]]:
    """HETATM grouped by (chain, resseq), LARGEST group first. Optional resname
    filter. The largest group is the design ligand (e.g. NADP); the next is the
    co-substrate (e.g. formate) — so ligand_rank=0/1 selects them even when an MD
    PDB labels both 'UNK'."""
    groups: Dict[Tuple[str, int], List[_Atom]] = {}
    for a in atoms:
        if a.record != "HETATM":
            continue
        if resname and a.resname != resname:
            continue
        if heavy_only and a.element == "H":
            continue
        groups.setdefault((a.chain, a.resseq), []).append(a)
    return sorted(groups.values(), key=len, reverse=True)


def gate_from_pdb(
    ref_pdb, test_pdb, *, ligand_resname: Optional[str] = None,
    ligand_rank: int = 0, ligand_id: str = "ligand", role: str = "other",
    pocket_cutoff: float = 12.0, thresholds: Optional[PoseThresholds] = None,
) -> PoseGateResult:
    """Convenience: gate a test PDB against a reference PDB. The ligand is the
    ``ligand_rank``-th largest HETATM residue (rank 0 = design ligand / NADP, rank 1
    = co-substrate / formate) optionally filtered by ``ligand_resname`` — this
    survives MD PDBs that label every het 'UNK'. Pocket = protein Cα within
    ``pocket_cutoff`` Å of any reference ligand atom (matched to the test by
    (chain, resseq)); only Cα present in BOTH are used, so a point mutation dropping
    a side chain does not break the pocket set."""
    ref, test = read_pdb_atoms(ref_pdb), read_pdb_atoms(test_pdb)
    r_groups = _het_residues(ref, ligand_resname)
    t_groups = _het_residues(test, ligand_resname)
    if len(r_groups) <= ligand_rank or len(t_groups) <= ligand_rank:
        return PoseGateResult(ligand_id, role, SKIPPED, None, None,
                              0, 0, note="ligand rank not present")
    r_lig, t_lig = r_groups[ligand_rank], t_groups[ligand_rank]
    r_lig_xyz = np.array([a.xyz for a in r_lig], float)
    # pocket Cα (reference) within cutoff of the reference ligand
    r_ca = _ca(ref); t_ca = _ca(test)
    lig_c = r_lig_xyz
    keys, r_pts, t_pts = [], [], []
    for key, a in r_ca.items():
        if key not in t_ca:
            continue
        p = np.array(a.xyz, float)
        if np.sqrt(((lig_c - p) ** 2).sum(1)).min() <= pocket_cutoff:
            keys.append(key); r_pts.append(a.xyz); t_pts.append(t_ca[key].xyz)
    if len(keys) < 3:
        return PoseGateResult(ligand_id, role, SKIPPED, None, None,
                              len(r_lig), len(keys), note="pocket Cα < 3")
    return evaluate_pose(
        np.array(r_pts, float), np.array(t_pts, float),
        r_lig_xyz, np.array([a.xyz for a in t_lig], float),
        ligand_id=ligand_id, role=role, thresholds=thresholds)
