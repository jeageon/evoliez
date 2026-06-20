"""Shared helpers for adapters: deterministic synthetic structures, pocket
placement, and a minimal PDB writer (no Biopython dependency required)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Sequence

from evoliez.types import Ligand, LigandAtom, Pose, ProteinStructure, Residue
from evoliez.utils.seeds import derive_seed


class RealToolError(RuntimeError):
    """A real-backend tool produced no usable result and mock fallback is
    forbidden (the default). Raised instead of silently degrading a real run to
    a mock/synthetic artifact (audit P0 #3)."""


def mock_fallback_allowed() -> bool:
    """True only when the run explicitly permits mock degradation in a real run
    (Config.allow_mock_fallback -> EVOLIEZ_ALLOW_MOCK_FALLBACK, set by
    Pipeline.run). Default False: a missing/failed real tool hard-fails."""
    return os.environ.get("EVOLIEZ_ALLOW_MOCK_FALLBACK") == "1"


def fail_unless_mock_allowed(detail: str) -> None:
    """On the REAL path, raise unless mock fallback was explicitly enabled.
    Caller logs + returns its mock contract only when this returns normally."""
    if not mock_fallback_allowed():
        raise RealToolError(
            detail + " — refusing to substitute a mock result in a real run. "
            "Fix the tool/inputs, or set allow_mock_fallback=true "
            "(EVOLIEZ_ALLOW_MOCK_FALLBACK=1) to permit degradation."
        )

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

    def _rec(rec: str, serial: int, name: str, resn: str, chain: str,
             resseq: int, x: float, y: float, z: float, b: float,
             elem: str) -> str:
        # STRICT PDB columns - OpenMM's PDBFile reader is column-exact and
        # rejects anything else with "Misaligned residue name". Layout:
        # 1-6 rec | 7-11 serial | 13-16 name | 17 altLoc | 18-20 resName
        # | 22 chain | 23-26 resSeq | 31-54 xyz | 55-60 occ | 61-66 b
        # | 77-78 element.
        nm = f" {name:<3}" if len(name) < 4 else name[:4]
        return (
            f"{rec:<6}{serial:>5d} {nm}"
            f" {resn[:3]:>3s} {chain:1.1s}{resseq:>4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{b:6.2f}          {elem:>2s}"
        )

    lines: List[str] = []
    serial = 1
    for res in structure.residues:
        x, y, z = res.ca
        lines.append(_rec("ATOM", serial, "CA",
                          _THREE.get(res.aa, "GLY"), "A", res.index,
                          x, y, z, res.plddt, "C"))
        serial += 1
    if ligand_atoms:
        for a in ligand_atoms:
            x, y, z = a.coord
            lines.append(_rec("HETATM", serial, a.id[:4], "LIG", "L", 1,
                              x, y, z, 0.0, a.element or "C"))
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def is_full_atom_pdb(path: Path) -> bool:
    """True if the PDB carries more than a CA-only trace (>=1 non-CA protein
    ATOM record). ``write_min_pdb`` (mock / our CA-only ProteinStructure) emits
    ONLY CA records — Vina/GNINA dock into a sidechain-less pocket and
    FoldX/Rosetta cannot build residue templates from such a file."""
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("ATOM") and line[12:16].strip() not in ("CA", ""):
                return True
    except OSError:
        return False
    return False


def full_atom_receptor_pdb(structure: ProteinStructure, out_path: Path,
                           keep_het_chains=None) -> bool:
    """Write a FULL-ATOM, protein-only receptor PDB for REAL docking/stability.

    Real Boltz writes a full-atom complex to ``structure.pdb_path``; our
    in-memory ``ProteinStructure`` is a CA-only trace, so ``write_min_pdb``
    produces a backbone-only receptor that makes real Vina/GNINA/DiffDock
    scientifically meaningless and FoldX/Rosetta fail. Prefer the real
    structure: copy its protein ``ATOM`` records (dropping ligand ``HETATM`` —
    docking re-adds the reference ligand separately, FoldX builds the mutant
    from the WT) into ``out_path``.

    Returns ``False`` when no full-atom structure is available (mock / CA-only
    upstream) so the caller degrades HONESTLY (mock dock / unavailable ddG)
    instead of running the tool on a CA-only trace. Mirrors the guard OpenMM
    already applies (`openmm_engine._run_real`)."""
    src = getattr(structure, "pdb_path", None)
    if not src:
        return False
    src = Path(src)
    if not src.exists() or not is_full_atom_pdb(src):
        return False
    # Protein ATOM/TER always; HETATM only for chains in `keep_het_chains` — used
    # to keep CO-MODELLED cofactors/substrates as FIXED context when docking one
    # ligand in the presence of the others (multi-ligand co-docking).
    keep_het = set(keep_het_chains or ())
    kept = [ln for ln in src.read_text().splitlines()
            if ln.startswith(("ATOM", "TER"))
            or (ln.startswith("HETATM") and ln[21:22] in keep_het)]
    if not any(ln.startswith("ATOM") for ln in kept):
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(kept) + "\nEND\n")
    return True


def het_chains_in_pdb(pdb_path) -> List[str]:
    """Sorted distinct HETATM (ligand) chain IDs in a complex PDB."""
    try:
        lines = Path(pdb_path).read_text().splitlines()
    except OSError:
        return []
    return sorted({ln[21:22] for ln in lines if ln.startswith("HETATM")})


def parse_pdb_het_chain(pdb_path, chain: str) -> List[LigandAtom]:
    """LigandAtoms (placed coords) of ONE HETATM chain in a complex PDB — a
    co-modelled ligand's actual pose, used as the redock reference for that
    ligand and as fixed context when docking the others."""
    atoms: List[LigandAtom] = []
    try:
        lines = Path(pdb_path).read_text().splitlines()
    except OSError:
        return atoms
    for ln in lines:
        if ln.startswith("HETATM") and ln[21:22] == chain:
            try:
                x, y, z = float(ln[30:38]), float(ln[38:46]), float(ln[46:54])
            except ValueError:
                continue
            elem = ln[76:78].strip() or ln[12:14].strip().lstrip("0123456789")[:1]
            atoms.append(LigandAtom(id=ln[12:16].strip() or f"{elem}{len(atoms)}",
                                    element=elem, coord=(x, y, z)))
    return atoms


def parse_sdf_first_pose(path: Path) -> List[LigandAtom]:
    """Atom coords + elements of the FIRST molecule in a V2000 SDF (the best /
    rank-1 docked pose for gnina/diffdock). Returns [] if unparseable so the
    caller falls back to the reference atoms."""
    try:
        text = Path(path).read_text()
    except OSError:
        return []
    block = text.split("$$$$", 1)[0].splitlines()
    if len(block) < 4:
        return []
    try:
        natoms = int(block[3][0:3])
    except ValueError:
        return []
    atoms: List[LigandAtom] = []
    for ln in block[4:4 + natoms]:
        tok = ln.split()
        if len(tok) < 4:
            continue
        try:
            x, y, z = float(tok[0]), float(tok[1]), float(tok[2])
        except ValueError:
            continue
        elem = tok[3]
        atoms.append(LigandAtom(id=f"{elem}{len(atoms)}", element=elem,
                                coord=(x, y, z)))
    return atoms


def parse_sdf_all_poses(path: Path) -> List[List[LigandAtom]]:
    """Atom coords + elements of EVERY molecule in a multi-record V2000 SDF,
    in file order (``$$$$``-separated). gnina writes its N docked modes as N
    molecules in ONE output SDF and DiffDock can write several ranks per file;
    this returns one atom list per record (the rank-i pose for record i).

    Returns ``[]`` (not ``[[]]``) when nothing parses, so a caller can fall back
    to the reference. Mirrors ``parse_sdf_first_pose`` per record."""
    try:
        text = Path(path).read_text()
    except OSError:
        return []
    poses: List[List[LigandAtom]] = []
    # Trailing "$$$$\n" yields an empty final chunk; skip blocks with no header.
    for chunk in text.split("$$$$"):
        block = chunk.splitlines()
        # Drop leading blank lines so a record after a "$$$$" delimiter still has
        # its title/counts line at the expected offset.
        while block and not block[0].strip():
            block.pop(0)
        if len(block) < 4:
            continue
        try:
            natoms = int(block[3][0:3])
        except ValueError:
            continue
        atoms: List[LigandAtom] = []
        for ln in block[4:4 + natoms]:
            tok = ln.split()
            if len(tok) < 4:
                continue
            try:
                x, y, z = float(tok[0]), float(tok[1]), float(tok[2])
            except ValueError:
                continue
            elem = tok[3]
            atoms.append(LigandAtom(id=f"{elem}{len(atoms)}", element=elem,
                                    coord=(x, y, z)))
        if atoms:
            poses.append(atoms)
    return poses


def tool_version(executable: str, *args: str) -> str:
    """Best-effort one-line version string for a real tool (``<exe> --version``
    by default). Recorded into ``Pose.engine_version`` for the audit trail; on
    any failure (tool missing, non-zero exit, dry-run) returns "" rather than
    raising — provenance is a nice-to-have, never a run-blocker."""
    from evoliez.utils.subprocess_utils import run, which

    if which(executable) is None:
        return ""
    flag = list(args) or ["--version"]
    try:
        res = run([executable, *flag], check=False)
    except Exception:
        return ""
    out = (res.stdout or res.stderr or "").strip()
    return out.splitlines()[0].strip() if out else ""


def mock_redock_modes(
    candidate_id: str,
    method: str,
    reference_atoms: Sequence[LigandAtom],
    *,
    n_modes: int,
    score_type: str = "",
    higher_is_better: bool = False,
    with_cnn: bool = False,
    base_instability: float = 0.1,
    score_offset: float = 0.0,
) -> List[Pose]:
    """Deterministic N-mode ranked ensemble for the mock backend, descending in
    plausibility: rank 1 sits closest to the reference with the best score, later
    ranks drift further with monotonically worse scores. Lets ``redock_all`` /
    ``redock_batch`` tests exercise multi-pose parsing without real gnina/diffdock.

    ``higher_is_better`` flips the score ordering (diffdock confidence: rank 1 is
    the HIGHEST). ``with_cnn`` populates the gnina CNN provenance fields. Each
    pose carries its ``rank`` (1-based) and ``score_type``; the score still flows
    through the same ``mock_redock`` jitter model so coords/RMSD stay realistic."""
    poses: List[Pose] = []
    for r in range(n_modes):
        # rank 1 -> smallest instability; later ranks drift further out.
        inst = base_instability + 0.18 * r
        h = derive_seed(0xA11, candidate_id, method, str(r))
        pose = mock_redock(
            candidate_id, method, reference_atoms,
            instability=inst, score_offset=score_offset,
        )
        # Re-key the score so ranks are MONOTONIC (mock_redock's raw score is a
        # function of instability, but we want a clean, strictly-ordered ladder
        # for assertions). Best magnitude at rank 1, degrading by rank.
        base = -9.0 + score_offset
        if higher_is_better:
            # diffdock confidence: rank 1 highest (least negative), descending.
            pose.score = round(-0.4 - 0.6 * r, 4)
        else:
            # affinity-like: rank 1 most negative (best), ascending toward 0.
            pose.score = round(base + 1.2 * r, 4)
        pose.rank = r + 1
        pose.cluster = r
        pose.score_type = score_type
        if with_cnn:
            # plausible CNN provenance: CNNscore in (0,1], CNNaffinity ~ pK.
            pose.cnn_score = round(max(0.0, 0.85 - 0.07 * r), 4)
            pose.cnn_affinity = round(6.5 - 0.5 * r, 4)
        poses.append(pose)
    return poses


def lock_pose_to_reference(
    parsed: Sequence[LigandAtom],
    reference: Sequence[LigandAtom],
    *,
    candidate_id: str = "",
    method: str = "",
    logger=None,
) -> tuple[List[LigandAtom], "float | None"]:
    """Relabel docked atoms onto the canonical reference (ids/chemistry kept,
    docked xyz adopted) and compute heavy-atom RMSD-to-reference. Shared by all
    three real dockers so pose-escape / redocking-consistency are computable
    uniformly (gnina/diffdock previously discarded coords -> rmsd None -> a
    falsely 'perfect' redocking_consistency in s09). Mirrors the heavy-atom
    fallback Vina established. Returns (atoms, rmsd|None); on an unverified
    lock, returns (reference, None)."""
    from evoliez.features.ligand import relabel_to_canonical

    ref = list(reference)
    locked, ok = relabel_to_canonical(list(parsed), ref)
    if ok and locked:
        rref = ([a for a in ref if (a.element or "").upper() != "H"]
                if len(locked) != len(ref) else ref)
        rmsd = None
        if len(rref) == len(locked) and locked:
            rmsd = round(math.sqrt(sum(
                sum((locked[i].coord[k] - rref[i].coord[k]) ** 2
                    for k in range(3)) for i in range(len(locked))
            ) / len(locked)), 3)
        return locked, rmsd
    if parsed and logger is not None:
        logger.warning(
            "%s pose atom ids NOT verified vs reference (%d vs %d heavy) for "
            "%s; RMSD-to-reference unavailable", method or "docking",
            sum(1 for a in parsed if (a.element or "").upper() != "H"),
            sum(1 for a in ref if (a.element or "").upper() != "H"),
            candidate_id,
        )
    return ref, None


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
