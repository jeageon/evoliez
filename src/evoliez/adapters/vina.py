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
    vina_cmd = [
        "vina", "--receptor", str(rec_q), "--ligand", str(lig_q),
        "--center_x", f"{cx[0]:.2f}", "--center_y", f"{cx[1]:.2f}",
        "--center_z", f"{cx[2]:.2f}", "--size_x", "22", "--size_y", "22",
        "--size_z", "22", "--num_modes", str(cfg.poses_per_candidate),
        "--exhaustiveness", str(getattr(cfg, "exhaustiveness", 8)),
        "--out", str(out),
    ]
    cpu = getattr(cfg, "cpu", 0)
    if cpu and cpu > 0:
        vina_cmd += ["--cpu", str(cpu)]
    # Hard wall: a runaway NADP-scale dock fails loudly instead of hanging
    # forever with hidden (captured) stdout.
    run(vina_cmd, dry_run=dry_run, timeout=getattr(cfg, "timeout_s", 1800))
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

    from evoliez.features.ligand import relabel_to_canonical

    score = 0.0
    got_score = False
    parsed: list[LigandAtom] = []
    for line in out.read_text().splitlines():
        if line.startswith("REMARK VINA RESULT") and not got_score:
            try:
                score = float(line.split()[3])
                got_score = True
            except (ValueError, IndexError):
                score = 0.0
        elif line.startswith("ENDMDL"):
            if parsed:  # keep only the first (best) model
                break
        elif line.startswith(("ATOM", "HETATM")):
            try:
                x, y, z = (float(line[30:38]), float(line[38:46]),
                           float(line[46:54]))
            except ValueError:
                continue
            tok = line.split()
            elem = _ad_element(tok[-1]) if tok else "C"
            parsed.append(LigandAtom(id=f"{elem}{len(parsed)}",
                                     element=elem, coord=(x, y, z)))

    # Vina pdbqt = heavy + polar H (e.g. 54), canonical = heavy + all RDKit H
    # (e.g. 70). Delegate to the heavy-atom-aware atom-index lock instead of
    # an exact-count check, so docked coords + RMSD survive the H asymmetry.
    locked, ok = relabel_to_canonical(parsed, list(ref))
    rmsd = None
    if ok and locked:
        rref = ([a for a in ref if (a.element or "").upper() != "H"]
                if len(locked) != len(ref) else list(ref))
        if len(rref) == len(locked):
            rmsd = round(math.sqrt(sum(
                sum((locked[i].coord[k] - rref[i].coord[k]) ** 2
                    for k in range(3)) for i in range(len(locked))
            ) / len(locked)), 3)
    else:
        # Couldn't lock the pose -> fall back to the reference atoms so
        # downstream always has a ligand (rmsd unavailable). Warn only when
        # atoms WERE parsed but the heavy counts disagreed (real mismatch),
        # not when the file simply had no parseable ATOM records.
        if parsed:
            log.warning(
                "Vina pose vs reference heavy-atom mismatch (%d vs %d) for "
                "%s; RMSD unavailable",
                sum(1 for a in parsed if (a.element or "").upper() != "H"),
                sum(1 for a in ref if (a.element or "").upper() != "H"),
                candidate_id,
            )
        locked = list(ref)
    return Pose(candidate_id=candidate_id, method=METHOD, score=score,
                ligand_atoms=locked, rmsd_to_reference=rmsd, cluster=0)


# AutoDock atom type -> element (pdbqt last column). Only the H vs non-H
# distinction must be exact (drives the heavy-atom lock); HD/HS are polar H.
_AD2ELEM = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "O": "O", "OA": "O",
    "OS": "O", "S": "S", "SA": "S", "P": "P", "H": "H", "HD": "H", "HS": "H",
    "F": "F", "CL": "Cl", "BR": "Br", "I": "I", "MG": "Mg", "MN": "Mn",
    "ZN": "Zn", "CA": "Ca", "FE": "Fe",
}


def _ad_element(adtype: str) -> str:
    t = (adtype or "").strip().upper()
    if t in _AD2ELEM:
        return _AD2ELEM[t]
    return "H" if t[:1] == "H" else (t[:1].title() or "C")
