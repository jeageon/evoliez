"""DiffDock diffusion docking (spec section 10). GPU tool (server only).

real: DiffDock inference CLI.
mock: deterministic perturbed pose (shared helper).
"""

from __future__ import annotations

import os
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

log = get_logger("evoliez.diffdock")
METHOD = "diffdock"


def _resolve_diffdock_install() -> Path:
    """Locate the DiffDock install root via `EVOLIEZ_DIFFDOCK`.
    DiffDock's `inference` module is script-style and only importable
    when cwd is the DiffDock checkout; the old `python -m inference`
    silently relied on the caller having chdir'd there. Fail loudly so
    real backend doesn't silently fall through to a misbehaving
    subprocess."""
    root = os.environ.get("EVOLIEZ_DIFFDOCK", "").strip()
    if not root:
        raise RuntimeError(
            "EVOLIEZ_DIFFDOCK env var is not set. Point it at your "
            "DiffDock install root (the directory containing inference.py "
            "or inference/__init__.py). Mock backend works without this; "
            "real backend requires it."
        )
    rp = Path(root)
    if not (rp / "inference.py").exists() and not (rp / "inference" /
                                                     "__init__.py").exists():
        raise RuntimeError(
            f"EVOLIEZ_DIFFDOCK={root} but no inference module found there. "
            f"Set the env var to the DiffDock install root."
        )
    return rp


def redock(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    instability: float,
    smiles: str,
    backend: Backend,
    dry_run: bool = False,
) -> Pose:
    if backend is Backend.real:
        return _redock_real(
            candidate_id, structure, smiles, cfg, workdir,
            reference_atoms, dry_run=dry_run,
        )
    return mock_redock(
        candidate_id, METHOD, reference_atoms, instability=instability,
        score_offset=-0.2,
    )


def _redock_real(
    candidate_id: str,
    structure: ProteinStructure,
    smiles: str,
    cfg: DockingConfig,
    workdir: Path,
    reference_atoms: Sequence[LigandAtom],
    *,
    dry_run: bool,
) -> Pose:
    # P0.1: real DiffDock refuses CA-only receptors - its diffusion model
    # was trained on full-atom PDB inputs; CA-only triggers garbage poses.
    try:
        rec = resolve_real_receptor_pdb(structure, candidate_id=candidate_id)
    except NotFullAtomReceptor as exc:
        log.warning("diffdock: %s", exc)
        return Pose(
            candidate_id=candidate_id, method=METHOD, score=0.0,
            ligand_atoms=list(reference_atoms),
            skipped="skipped_no_full_atom_structure",
            pose_validity_status="unknown",
            pose_validity_reasons=[str(exc)],
        )
    require("python")
    install_root = _resolve_diffdock_install()
    apply_gpu_selection()
    workdir.mkdir(parents=True, exist_ok=True)
    csv = workdir / f"{candidate_id}_input.csv"
    csv.write_text(
        "complex_name,protein_path,ligand_description,protein_sequence\n"
        f"{candidate_id},{rec},{smiles},\n"
    )
    out = workdir / f"{candidate_id}_dd_out"
    # `python -m inference` was cwd-relative; only worked if the caller
    # was chdir'd into the DiffDock repo. Now we explicitly chdir into
    # the install root (via cwd=) so the inference module is importable.
    run(
        ["python", "-m", "inference", "--protein_ligand_csv", str(csv),
         "--out_dir", str(out), "--samples_per_complex",
         str(cfg.poses_per_candidate)],
        dry_run=dry_run, cwd=install_root,
    )
    if dry_run or not out.exists():
        return mock_redock(candidate_id, METHOD, reference_atoms, instability=0.2)
    score, atoms = _parse_diffdock(out)
    locked, rmsd = _lock_to_reference(atoms, list(reference_atoms),
                                      candidate_id)
    return Pose(candidate_id=candidate_id, method=METHOD, score=score,
                ligand_atoms=locked, rmsd_to_reference=rmsd, cluster=0)


def _lock_to_reference(parsed, ref, candidate_id: str):
    """Shared with gnina._lock_to_reference - returns (atoms, rmsd_or_None).
    Without this, DiffDock returned `reference_atoms` as ligand_atoms ->
    `pose.rmsd_to_reference = None` -> s09 `None or 0.0 = 0.0` -> the
    ligand-escape gate was vacuously true."""
    import math

    from evoliez.features.ligand import relabel_to_canonical

    locked, ok = relabel_to_canonical(parsed, ref) if parsed else (None, False)
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
            "DiffDock pose atom ids NOT verified vs reference (%d vs %d "
            "heavy) for %s; RMSD-to-reference unavailable",
            sum(1 for a in parsed if (a.element or "").upper() != "H"),
            sum(1 for a in ref if (a.element or "").upper() != "H"),
            candidate_id,
        )
    return list(ref), None


def _parse_diffdock(out_dir):
    """DiffDock writes `rank1_confidence-X.XX.sdf`; the confidence is encoded
    in the filename. Returns ``(confidence_score, best_pose_atoms)`` -
    previously this dropped the atom block, leaving the caller with no
    RMSD signal for downstream gating (see _lock_to_reference docstring)."""
    import re
    from pathlib import Path

    from evoliez.types import LigandAtom

    confs = sorted(Path(out_dir).rglob("rank1*.sdf"))
    if not confs:
        return 0.0, []
    m = re.search(r"confidence(-?\d+\.?\d*)", confs[0].name)
    score = round(float(m.group(1)), 4) if m else -7.0
    # Parse the atom block from the SDF (single record per file in
    # DiffDock's rank1 output).
    lines = confs[0].read_text().splitlines()
    counts_i = -1
    for j, ln in enumerate(lines[:8]):
        if "V2000" in ln or "V3000" in ln:
            counts_i = j
            break
    if counts_i < 0:
        return score, []
    try:
        n_atoms = int(lines[counts_i][:3].strip())
    except (ValueError, IndexError):
        return score, []
    atoms = []
    for k in range(n_atoms):
        line = lines[counts_i + 1 + k] if counts_i + 1 + k < len(lines) else ""
        tok = line.split()
        if len(tok) < 4:
            return score, []
        try:
            x, y, z = float(tok[0]), float(tok[1]), float(tok[2])
        except ValueError:
            return score, []
        atoms.append(LigandAtom(id=f"{tok[3]}{k}", element=tok[3],
                                coord=(x, y, z)))
    return score, atoms
