"""DiffDock diffusion docking (spec section 10). GPU tool (server only).

real: DiffDock inference CLI.
mock: deterministic perturbed pose (shared helper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from evoliez.adapters.base import (
    lock_pose_to_reference,
    mock_redock,
    parse_sdf_first_pose,
    write_min_pdb,
)
from evoliez.config import Backend, DockingConfig
from evoliez.logging_utils import get_logger
from evoliez.types import LigandAtom, Pose, ProteinStructure
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.diffdock")
METHOD = "diffdock"


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
    require("python")
    apply_gpu_selection()
    workdir.mkdir(parents=True, exist_ok=True)
    rec = workdir / f"{candidate_id}_rec.pdb"
    write_min_pdb(rec, structure)
    csv = workdir / f"{candidate_id}_input.csv"
    csv.write_text(
        "complex_name,protein_path,ligand_description,protein_sequence\n"
        f"{candidate_id},{rec},{smiles},\n"
    )
    out = workdir / f"{candidate_id}_dd_out"
    run(
        ["python", "-m", "inference", "--protein_ligand_csv", str(csv),
         "--out_dir", str(out), "--samples_per_complex",
         str(cfg.poses_per_candidate)],
        dry_run=dry_run,
    )
    if dry_run:
        return mock_redock(candidate_id, METHOD, reference_atoms, instability=0.2)
    if not out.exists():
        log.warning(
            "diffdock produced no output for %s (%s); using mock fallback "
            "(NOT a real dock)", candidate_id, out,
        )
        return mock_redock(candidate_id, METHOD, reference_atoms, instability=0.2)
    score = _parse_diffdock(out)
    # Adopt the rank-1 docked pose coordinates (was discarded -> no RMSD ->
    # falsely 'perfect' redocking_consistency in s09).
    ranks = sorted(Path(out).rglob("rank1*.sdf"))
    locked, rmsd = lock_pose_to_reference(
        parse_sdf_first_pose(ranks[0]) if ranks else [], reference_atoms,
        candidate_id=candidate_id, method=METHOD, logger=log,
    )
    return Pose(candidate_id=candidate_id, method=METHOD, score=score,
                ligand_atoms=locked, rmsd_to_reference=rmsd, cluster=0)


def _parse_diffdock(out_dir) -> float:
    """DiffDock writes `rank1_confidence-X.XX.sdf`; the confidence is encoded
    in the filename. Return it (higher = better; 0.0 if no pose)."""
    import re
    from pathlib import Path

    confs = sorted(Path(out_dir).rglob("rank1*.sdf"))
    if not confs:
        return 0.0
    m = re.search(r"confidence(-?\d+\.?\d*)", confs[0].name)
    return round(float(m.group(1)), 4) if m else -7.0
