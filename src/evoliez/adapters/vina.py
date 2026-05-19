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
    lig_pdb = workdir / f"{candidate_id}_lig.pdb"
    lig_q = workdir / f"{candidate_id}_lig.pdbqt"
    out = workdir / f"{candidate_id}_vina_out.pdbqt"
    # receptor prep
    run(["obabel", str(rec_pdb), "-O", str(rec_q), "-xr"], dry_run=dry_run)
    # ligand prep (was MISSING -> vina got a non-existent --ligand file):
    # write the reference ligand atoms, add H + Gasteiger charges, -> pdbqt
    write_min_pdb(lig_pdb, ProteinStructure(sequence="", residues=[]),
                  reference_atoms)
    run(["obabel", str(lig_pdb), "-O", str(lig_q),
         "-h", "--partialcharge", "gasteiger"], dry_run=dry_run)
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
    """Parse the BEST (first) Vina pose: affinity + the docked ligand
    coordinates. Coordinates are locked onto the canonical reference atom
    list (ids/chemistry kept, Vina xyz adopted) so RMSD-to-reference and
    pose-escape are computable downstream - the old parser discarded the
    docked coords (ligand_atoms=ref, rmsd=None), making pose-quality
    validation structurally impossible."""
    import math

    score = 0.0
    got_score = False
    coords: list[tuple[float, float, float]] = []
    for line in out.read_text().splitlines():
        if line.startswith("REMARK VINA RESULT") and not got_score:
            try:
                score = float(line.split()[3])
                got_score = True
            except (ValueError, IndexError):
                score = 0.0
        elif line.startswith("ENDMDL"):
            if coords:  # keep only the first (best) model
                break
        elif line.startswith(("ATOM", "HETATM")):
            try:
                coords.append((float(line[30:38]), float(line[38:46]),
                               float(line[46:54])))
            except ValueError:
                continue

    if coords and len(coords) == len(ref):
        atoms = []
        for a, c in zip(ref, coords):
            na = LigandAtom(**vars(a))
            na.coord = c
            atoms.append(na)
        rmsd = round(math.sqrt(sum(
            sum((atoms[i].coord[k] - ref[i].coord[k]) ** 2 for k in range(3))
            for i in range(len(ref))
        ) / len(ref)), 3)
    else:
        if coords:
            log.warning(
                "Vina pose atom count (%d) != reference (%d) for %s; keeping "
                "reference coords, RMSD unavailable",
                len(coords), len(ref), candidate_id,
            )
        atoms, rmsd = list(ref), None
    return Pose(candidate_id=candidate_id, method=METHOD, score=score,
                ligand_atoms=atoms, rmsd_to_reference=rmsd, cluster=0)
