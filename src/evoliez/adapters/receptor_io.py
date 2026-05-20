"""Receptor-PDB resolution for real-backend docking (plan P0.1).

The previous default `write_min_pdb(rec_pdb, structure)` emits the internal
ProteinStructure - which is a CA-only trace upstream of real Boltz - and
hands that to Vina/GNINA/DiffDock. The dockers then run against a stick-
figure receptor and produce scientifically meaningless but plausible-
looking scores. This module forces real docking to either:

1. Use a full-atom PDB already on disk (typically ``structure.pdb_path``
   from a Boltz prediction), verified by :func:`is_full_atom_pdb`; or
2. Refuse with :class:`NotFullAtomReceptor`, so the caller writes
   ``Pose(skipped="...")`` and the candidate is honestly NOT docked.

Mock paths keep using :func:`write_mock_receptor_pdb` (a thin wrapper around
``write_min_pdb``) because the mock funnel is deterministic and CA-only
geometry is acceptable there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from evoliez.adapters.base import write_min_pdb
from evoliez.logging_utils import get_logger
from evoliez.types import LigandAtom, ProteinStructure

log = get_logger("evoliez.receptor")


class NotFullAtomReceptor(Exception):
    """Real docking refused because the receptor isn't a full-atom PDB.

    Carries the candidate id + the reason so the caller can record it as
    ``Pose(skipped="...")`` instead of silently downgrading to mock.
    """


def is_full_atom_pdb(path: Path) -> bool:
    """True if ``path`` contains any non-CA backbone/sidechain ATOM record.

    Mirrors :func:`evoliez.adapters.openmm_engine._is_full_atom_pdb` so MD
    and docking agree on what "full-atom" means.
    """
    try:
        for line in path.read_text().splitlines():
            if line.startswith("ATOM") and line[12:16].strip() not in ("CA", ""):
                return True
    except OSError:
        return False
    return False


def resolve_real_receptor_pdb(
    structure: ProteinStructure, *, candidate_id: str = "?",
) -> Path:
    """Return a usable full-atom receptor PDB for real docking.

    Raises :class:`NotFullAtomReceptor` if no full-atom source is available
    so the caller emits ``Pose(skipped="skipped_no_full_atom_structure")``
    instead of feeding a CA stick figure to Vina/GNINA/DiffDock.
    """
    src = getattr(structure, "pdb_path", None)
    if src and Path(src).exists() and is_full_atom_pdb(Path(src)):
        log.debug("real receptor %s -> %s", candidate_id, src)
        return Path(src)
    raise NotFullAtomReceptor(
        f"real docking requires a full-atom receptor for {candidate_id} "
        f"but structure.pdb_path={src!r} is missing or CA-only"
    )


def write_mock_receptor_pdb(
    out_path: Path, structure: ProteinStructure,
    ligand_atoms: Sequence[LigandAtom] = (),
) -> Path:
    """CA-only receptor PDB for the MOCK docking path only.

    Real backends MUST use :func:`resolve_real_receptor_pdb` instead; this
    helper exists so the mock path's writer signature is explicit (and
    grep'able) and never accidentally winds up on the real path.
    """
    write_min_pdb(out_path, structure, ligand_atoms)
    return out_path
