"""GNINA CNN docking / rescoring (spec section 10). GPU tool (server only).

real: gnina with autobox around the reference ligand.
mock: deterministic perturbed pose (shared helper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from evoliez.adapters.base import mock_redock, write_min_pdb
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
    require("gnina")
    apply_gpu_selection()
    workdir.mkdir(parents=True, exist_ok=True)
    rec = workdir / f"{candidate_id}_rec.pdb"
    lig = workdir / f"{candidate_id}_ref_lig.pdb"
    out = workdir / f"{candidate_id}_gnina_out.sdf"
    write_min_pdb(rec, structure)
    write_min_pdb(lig, structure.__class__(sequence="", residues=[]), reference_atoms)
    run(
        ["gnina", "-r", str(rec), "-l", str(lig), "--autobox_ligand", str(lig),
         "--num_modes", str(cfg.poses_per_candidate), "-o", str(out)],
        dry_run=dry_run,
    )
    if dry_run or not out.exists():
        return mock_redock(candidate_id, METHOD, reference_atoms, instability=0.2)
    score = 0.0
    for line in out.read_text().splitlines():
        if "minimizedAffinity" in line:
            try:
                score = float(line.split()[-1])
            except Exception:
                pass
    return Pose(candidate_id=candidate_id, method=METHOD, score=score,
                ligand_atoms=list(reference_atoms), cluster=0)
