"""GNINA CNN docking / rescoring (spec section 10). GPU tool (server only).

real: gnina with autobox around the reference ligand.
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
    if dry_run:
        return mock_redock(candidate_id, METHOD, reference_atoms, instability=0.2)
    if not out.exists():
        # Real run produced no output (tool crash / wrong path). Degrade per
        # spec 23, but LOUDLY - a fabricated mock pose must not pass silently
        # as a real dock score into ranking.
        log.warning(
            "gnina produced no output for %s (%s); using mock fallback "
            "(NOT a real dock)", candidate_id, out,
        )
        return mock_redock(candidate_id, METHOD, reference_atoms, instability=0.2)
    # Adopt the docked pose coordinates (was discarded: ligand_atoms=reference,
    # rmsd=None -> s09 read a falsely 'perfect' redocking_consistency).
    locked, rmsd = lock_pose_to_reference(
        parse_sdf_first_pose(out), reference_atoms,
        candidate_id=candidate_id, method=METHOD, logger=log,
    )
    return Pose(candidate_id=candidate_id, method=METHOD,
                score=_parse_gnina(out),
                ligand_atoms=locked, rmsd_to_reference=rmsd, cluster=0)


def _parse_gnina(out) -> float:
    """Best (lowest) minimizedAffinity from a gnina SDF. In real SDF the tag
    value is on the line AFTER `> <minimizedAffinity>` (the old in-line
    float() parse missed it)."""
    from pathlib import Path

    lines = Path(out).read_text().splitlines()
    scores = []
    for i, ln in enumerate(lines):
        if "<minimizedAffinity>" in ln:
            for j in range(i + 1, min(i + 3, len(lines))):
                tok = lines[j].strip().split()
                if tok:
                    try:
                        scores.append(float(tok[0]))
                        break
                    except ValueError:
                        continue
        elif "minimizedAffinity" in ln:  # rare single-line form
            try:
                scores.append(float(ln.split()[-1]))
            except ValueError:
                pass
    return round(min(scores), 4) if scores else 0.0
