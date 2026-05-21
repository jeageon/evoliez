"""GNINA CNN docking / rescoring (spec section 10). GPU tool (server only).

real: gnina with autobox around the reference ligand.
mock: deterministic perturbed pose (shared helper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from evoliez.adapters.base import mock_redock, write_min_pdb
from evoliez.adapters.receptor_io import (
    NotFullAtomReceptor, resolve_real_receptor_pdb,
)
from evoliez.config import Backend, DockingConfig
from evoliez.logging_utils import get_logger
from evoliez.types import LigandAtom, Pose, ProteinStructure
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.gnina")
METHOD = "gnina"


def redock(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    instability: float,
    backend: Backend,
    dry_run: bool = False,
) -> Pose:
    if backend is Backend.real:
        return _redock_real(
            candidate_id, structure, reference_atoms, cfg, workdir, dry_run=dry_run
        )
    # CNN scoring tends to be a touch more optimistic than Vina; small offset
    return mock_redock(
        candidate_id, METHOD, reference_atoms, instability=instability,
        score_offset=-0.5,
    )


def _redock_real(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    dry_run: bool,
) -> Pose:
    try:
        rec = resolve_real_receptor_pdb(structure, candidate_id=candidate_id)
    except NotFullAtomReceptor as exc:
        log.warning("gnina: %s", exc)
        return Pose(
            candidate_id=candidate_id, method=METHOD, score=0.0,
            ligand_atoms=list(reference_atoms),
            skipped="skipped_no_full_atom_structure",
            pose_validity_status="unknown",
            pose_validity_reasons=[str(exc)],
        )
    require("gnina")
    apply_gpu_selection()
    workdir.mkdir(parents=True, exist_ok=True)
    # Ligand-only temp PDB: shared across candidates because reference_atoms
    # is the same reference ligand pose for every candidate in a run. Write
    # once and gnina re-reads the same file (~30x fewer disk writes on a
    # 30-candidate run; output is bit-identical because the file content
    # is the same).
    lig = workdir / "_shared_ref_lig.pdb"
    out = workdir / f"{candidate_id}_gnina_out.sdf"
    if not lig.exists():
        write_min_pdb(lig, structure.__class__(sequence="", residues=[]), reference_atoms)
    run(
        ["gnina", "-r", str(rec), "-l", str(lig), "--autobox_ligand", str(lig),
         "--num_modes", str(cfg.poses_per_candidate),
         "--cnn_scoring", "rescore", "--cnn", "fast",          # P0.3: GNINA 1.3 CNN
         "-o", str(out)],
        dry_run=dry_run,
    )
    if dry_run or not out.exists():
        return mock_redock(candidate_id, METHOD, reference_atoms, instability=0.2)
    score, cnn, atoms = _parse_gnina(out)
    locked, rmsd = _lock_to_reference(atoms, list(reference_atoms),
                                      candidate_id)
    return Pose(
        candidate_id=candidate_id, method=METHOD, score=score,
        ligand_atoms=locked, rmsd_to_reference=rmsd, cluster=0,
        cnn_score=cnn.get("CNNscore"),
        cnn_vs=cnn.get("CNN_VS"),
        cnn_affinity=cnn.get("CNN_affinity"),
    )


def _lock_to_reference(parsed, ref, candidate_id: str):
    """Atom-id lock + RMSD computation, shared with vina._parse_vina.
    Returns (ligand_atoms_for_pose, rmsd_to_reference_or_None). When
    the parse failed or the heavy-atom count diverges beyond what
    relabel_to_canonical can reconcile, we fall back to the reference
    atoms with rmsd=None - same honest-skip semantics as Vina, so s09's
    ligand_escape gate has a real RMSD to gate on instead of `None or
    0.0` (vacuously True before this fix)."""
    import math

    from evoliez.features.ligand import relabel_to_canonical

    locked, ok = relabel_to_canonical(parsed, ref) if parsed else (None, False)
    rmsd = None
    if ok and locked:
        rref = ([a for a in ref if (a.element or "").upper() != "H"]
                if len(locked) != len(ref) else list(ref))
        if len(rref) == len(locked):
            rmsd = round(math.sqrt(sum(
                sum((locked[i].coord[k] - rref[i].coord[k]) ** 2
                    for k in range(3)) for i in range(len(locked))
            ) / len(locked)), 3)
        return locked, rmsd
    if parsed:
        log.warning(
            "GNINA pose atom ids NOT verified vs reference (%d vs %d "
            "heavy) for %s; RMSD-to-reference unavailable",
            sum(1 for a in parsed if (a.element or "").upper() != "H"),
            sum(1 for a in ref if (a.element or "").upper() != "H"),
            candidate_id,
        )
    return list(ref), None


def _parse_gnina(out):
    """Parse a gnina SDF.

    Returns ``(best_minimizedAffinity, cnn_features_for_best_pose,
    best_pose_atoms)``. Expert audit follow-up: previously this dropped
    atom coordinates and the caller returned `reference_atoms` as the
    pose ligand_atoms, which silently made `pose.rmsd_to_reference =
    None` -> `None or 0.0 = 0.0` -> s09's `ligand_escape > 4.5` gate
    NEVER tripped for real GNINA. Now we parse the SDF atom block too
    and hand it back so the caller can compute RMSD vs reference.

    GNINA 1.3 emits ``CNNscore``, ``CNN_VS``, and ``CNN_affinity`` next
    to ``minimizedAffinity`` per pose; persisting them as Pose features
    lets the reranker use the CNN signal alongside Vina-style physics
    scoring.
    """
    from pathlib import Path

    from evoliez.types import LigandAtom

    lines = Path(out).read_text().splitlines()

    def _read_tag_value(i: int) -> "float | None":
        for j in range(i + 1, min(i + 3, len(lines))):
            tok = lines[j].strip().split()
            if tok:
                try:
                    return float(tok[0])
                except ValueError:
                    continue
        return None

    def _parse_atom_block(start: int, n_atoms: int):
        """SDF V2000 atom block: `xxxxxxxxxxyyyyyyyyyyzzzzzzzzzz aaa ...`
        with x/y/z each 10 chars wide. We use whitespace split rather
        than column slicing for robustness across writers (RDKit,
        OpenBabel, gnina); the format is loose enough in practice."""
        atoms = []
        for k in range(n_atoms):
            line = lines[start + k] if start + k < len(lines) else ""
            tok = line.split()
            if len(tok) < 4:
                return None
            try:
                x, y, z = float(tok[0]), float(tok[1]), float(tok[2])
            except ValueError:
                return None
            elem = tok[3]
            atoms.append(LigandAtom(id=f"{elem}{k}", element=elem,
                                    coord=(x, y, z)))
        return atoms

    # Walk records: record header (3 lines) + counts + atom block +
    # bond block + properties. Per-record we capture score / cnn /
    # atoms; at $$$$ we close the record.
    pose_records = []  # [(score, {CNN..}, [LigandAtom...]), ...]
    i = 0
    while i < len(lines):
        # Each SDF record opens with 3 header lines (name, info, blank
        # or comment), then a counts line: "  N1 N2  0  0 ... V2000".
        # Find the next counts line; not all writers emit a clean
        # 3-line header so we scan defensively.
        counts_i = -1
        for j in range(i, min(i + 8, len(lines))):
            if "V2000" in lines[j] or "V3000" in lines[j]:
                counts_i = j
                break
        if counts_i < 0:
            break
        try:
            n_atoms = int(lines[counts_i][:3].strip())
        except (ValueError, IndexError):
            n_atoms = 0
        atoms = _parse_atom_block(counts_i + 1, n_atoms) if n_atoms else []
        cur_score = None
        cur_cnn: dict = {}
        # Walk forward from the bond block looking for tags until $$$$.
        j = counts_i + 1 + n_atoms
        while j < len(lines):
            ln = lines[j]
            if "<minimizedAffinity>" in ln:
                v = _read_tag_value(j)
                if v is not None:
                    cur_score = v
            else:
                for tag in ("CNNscore", "CNN_VS", "CNN_affinity"):
                    if f"<{tag}>" in ln:
                        v = _read_tag_value(j)
                        if v is not None:
                            cur_cnn[tag] = v
            if ln.strip() == "$$$$":
                pose_records.append((cur_score, cur_cnn, atoms))
                j += 1
                break
            j += 1
        i = j

    if not pose_records:
        return 0.0, {}, []
    # Best (lowest minimizedAffinity); CNN features + atom coords come
    # from the same pose. Skip None-score records (malformed pose).
    scored = [r for r in pose_records if r[0] is not None]
    if not scored:
        return 0.0, {}, []
    best = min(scored, key=lambda r: r[0])
    return round(best[0], 4), best[1], best[2]
