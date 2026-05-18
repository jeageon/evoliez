"""Provenance / reproducibility (user §16).

Records the inputs, tool/version, seeds and ranking-formula version so a
mutation score can be reproduced later (paper / patent / tool).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evoliez import __version__


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


RANKING_FORMULA_VERSION = "2026-05-19.v5"  # bump when scoring math changes


def build_provenance(
    *, sequence: str, ligand_smiles: str, config_dict: dict,
    seed: int, backend: str, gnn_ckpt: Optional[str] = None,
) -> dict:
    return {
        "evoliez_version": __version__,
        "ranking_formula_version": RANKING_FORMULA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "backend": backend,
        "seed": seed,
        "input_sequence_sha1": _sha1(sequence),
        "ligand_smiles": ligand_smiles,
        "ligand_sha1": _sha1(ligand_smiles),
        "config_sha1": _sha1(json.dumps(config_dict, sort_keys=True,
                                        default=str)),
        "gnn_checkpoint": gnn_ckpt,
        "tools": {
            "boltz": "diffusion-ensemble adapter (real|mock)",
            "ligandmpnn": "adapter (real|mock)",
            "openmm": "MD-lite adapter (real|mock)",
            "foldx": "stability adapter (real|mock)",
        },
        "note": "Boltz/affinity/pLDDT used as features, never labels "
                "(see docs/ML_DATA_POLICY.md).",
    }


def write_provenance(path: Path, prov: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2))
    return path
