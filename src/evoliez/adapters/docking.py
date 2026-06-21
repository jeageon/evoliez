"""Docking-ensemble dispatcher (spec section 10.2).

Returns multiple poses per complex so the interaction model can use pose
variety as data augmentation. ``mock`` produces a deterministic K-pose
ensemble; ``real`` returns the tool's best pose (full K-mode parsing of
real docking output is a documented real-backend enhancement).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from evoliez.adapters import diffdock, gnina, vina
from evoliez.adapters.base import mock_dock_ensemble
from evoliez.config import Backend, DockingConfig
from evoliez.logging_utils import get_logger
from evoliez.types import LigandAtom, Pose, ProteinStructure

log = get_logger("evoliez.docking")


def dock_ensemble(
    method: str,
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    n_poses: int,
    smiles: str,
    backend: Backend,
    dry_run: bool = False,
) -> List[Pose]:
    if backend is Backend.mock:
        return mock_dock_ensemble(
            candidate_id, method, reference_atoms, n_poses=n_poses
        )

    # real: one best pose from the configured tool (K-mode parsing TODO)
    if method == "gnina":
        # SMILES lets gnina pick the reference-consistent mode (min symmetry-
        # corrected RMSD), not its CNN-rank-1 (flipped for large cofactors).
        pose = gnina.redock(candidate_id, structure, reference_atoms, cfg,
                            workdir, instability=0.1, backend=backend,
                            dry_run=dry_run, smiles=smiles)
    elif method == "diffdock":
        pose = diffdock.redock(candidate_id, structure, reference_atoms, cfg,
                               workdir, instability=0.1, smiles=smiles,
                               backend=backend, dry_run=dry_run)
    else:
        pose = vina.redock(candidate_id, structure, reference_atoms, cfg,
                           workdir, instability=0.1, backend=backend,
                           dry_run=dry_run)
    if not dry_run:
        log.info(
            "real %s: 1 pose for %s (multi-pose parsing is a real-backend "
            "enhancement; mock yields the full ensemble)",
            method, candidate_id,
        )
    return [pose]
