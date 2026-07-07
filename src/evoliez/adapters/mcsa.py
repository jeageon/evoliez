"""Catalytic-mechanism annotation source (M-CSA / BRENDA / heuristic).

real: load a user-supplied M-CSA-style JSON (catalytic residues, roles,
cofactor, reactive atoms) if ``mechanism.annotation_file`` is set.
mock / default: derive a mechanism from config (catalytic residues, cofactor)
+ ligand chemistry, so the pipeline runs without an external DB.

Mechanism annotation is a FEATURE / prior, never a supervised label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from evoliez.logging_utils import get_logger

log = get_logger("evoliez.mcsa")

# coarse residue -> catalytic role priors
_ACID_BASE = set("HDEK")
_NUCLEOPHILE = set("SCTY")
_METAL_COORD = set("HDEC")


def load_annotation(path: Optional[str]) -> Optional[dict]:
    if not path or not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:  # pragma: no cover
        log.warning("could not read mechanism annotation %s: %s", path, exc)
        return None


def heuristic_roles(
    catalytic_positions: List[int],
    residue_aa: Dict[int, str],
    cofactor: Optional[str],
) -> Dict[int, Dict[str, bool]]:
    """Assign coarse catalytic roles from the residue identity."""
    roles: Dict[int, Dict[str, bool]] = {}
    for p in catalytic_positions:
        aa = residue_aa.get(p, "X")
        roles[p] = {
            "catalytic": True,
            "acid_base": aa in _ACID_BASE,
            "nucleophile": aa in _NUCLEOPHILE,
            "electrophile": aa in ("R", "K") and cofactor is None,
            "metal_coord": bool(cofactor)
            and cofactor.lower() in ("zn", "mg", "mn", "fe", "cu", "ni", "co")
            and aa in _METAL_COORD,
            "cofactor_binding": bool(cofactor),
        }
    return roles
