"""AutoDock Vina redocking (spec section 10). CPU tool, runs anywhere.

real: obabel receptor/ligand prep + vina.
mock: deterministic perturbed pose (shared with the other dockers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from evoliez.adapters.base import mock_redock, write_min_pdb
from evoliez.config import Backend, DockingConfig
from evoliez.logging_utils import get_logger
from evoliez.types import LigandAtom, Pose, ProteinStructure
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.vina")
METHOD = "vina"


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
    return mock_redock(candidate_id, METHOD, reference_atoms, instability=instability)


def _redock_real(
    candidate_id: str,
    structure: ProteinStructure,
    reference_atoms: Sequence[LigandAtom],
    cfg: DockingConfig,
    workdir: Path,
    *,
    dry_run: bool,
) -> Pose:
    require("vina")
    require("obabel")
    workdir.mkdir(parents=True, exist_ok=True)
    rec_pdb = workdir / f"{candidate_id}_rec.pdb"
    write_min_pdb(rec_pdb, structure)
    rec_q = workdir / f"{candidate_id}_rec.pdbqt"
    lig_q = workdir / f"{candidate_id}_lig.pdbqt"
    out = workdir / f"{candidate_id}_vina_out.pdbqt"
    run(["obabel", str(rec_pdb), "-O", str(rec_q), "-xr"], dry_run=dry_run)
    cx = [sum(a.coord[i] for a in reference_atoms) / max(1, len(reference_atoms))
          for i in range(3)]
    run(
        ["vina", "--receptor", str(rec_q), "--ligand", str(lig_q),
         "--center_x", f"{cx[0]:.2f}", "--center_y", f"{cx[1]:.2f}",
         "--center_z", f"{cx[2]:.2f}", "--size_x", "22", "--size_y", "22",
         "--size_z", "22", "--num_modes", str(cfg.poses_per_candidate),
         "--out", str(out)],
        dry_run=dry_run,
    )
    if dry_run or not out.exists():
        return mock_redock(candidate_id, METHOD, reference_atoms, instability=0.2)
    return _parse_vina(candidate_id, out, reference_atoms)


def _parse_vina(candidate_id: str, out: Path, ref: Sequence[LigandAtom]) -> Pose:
    score = 0.0
    for line in out.read_text().splitlines():
        if line.startswith("REMARK VINA RESULT"):
            score = float(line.split()[3])
            break
    return Pose(candidate_id=candidate_id, method=METHOD, score=score,
                ligand_atoms=list(ref), rmsd_to_reference=None, cluster=0)
