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
    lig = workdir / f"{candidate_id}_ref_lig.pdb"
    out = workdir / f"{candidate_id}_gnina_out.sdf"
    # Ligand-only temp PDB: write_min_pdb is fine because the ligand atoms
    # are the dock target, not the receptor.
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
    score, cnn = _parse_gnina(out)
    return Pose(
        candidate_id=candidate_id, method=METHOD, score=score,
        ligand_atoms=list(reference_atoms), cluster=0,
        cnn_score=cnn.get("CNNscore"),
        cnn_vs=cnn.get("CNN_VS"),
        cnn_affinity=cnn.get("CNN_affinity"),
    )


def _parse_gnina(out):
    """Parse a gnina SDF.

    Returns ``(best_minimizedAffinity, cnn_features_for_best_pose)``.
    GNINA 1.3 emits ``CNNscore``, ``CNN_VS``, and ``CNN_affinity`` next to
    ``minimizedAffinity`` per pose; persisting them as Pose features lets
    the reranker use the CNN signal alongside Vina-style physics scoring.
    """
    from pathlib import Path

    lines = Path(out).read_text().splitlines()

    def _read_tag_value(i: int) -> float | None:
        for j in range(i + 1, min(i + 3, len(lines))):
            tok = lines[j].strip().split()
            if tok:
                try:
                    return float(tok[0])
                except ValueError:
                    continue
        return None

    pose_records = []                       # [(score, {CNNscore: .., ...}), ...]
    cur_score = None
    cur_cnn: dict = {}
    pose_open = False
    for i, ln in enumerate(lines):
        if "<minimizedAffinity>" in ln:
            v = _read_tag_value(i)
            if v is not None:
                cur_score = v
                pose_open = True
        for tag in ("CNNscore", "CNN_VS", "CNN_affinity"):
            if f"<{tag}>" in ln:
                v = _read_tag_value(i)
                if v is not None:
                    cur_cnn[tag] = v
        # SDF record terminator: stash + reset
        if ln.strip() == "$$$$" and pose_open:
            pose_records.append((cur_score, cur_cnn))
            cur_score, cur_cnn, pose_open = None, {}, False
    # Trailing pose without explicit $$$$
    if pose_open and cur_score is not None:
        pose_records.append((cur_score, cur_cnn))

    if not pose_records:
        return 0.0, {}
    # Best (lowest minimizedAffinity); CNN features come from the same pose.
    best = min(pose_records, key=lambda r: r[0])
    return round(best[0], 4), best[1]
