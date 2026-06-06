"""Provenance / reproducibility (user §16).

Records the inputs, tool/version, seeds and ranking-formula version so a
mutation score can be reproduced later (paper / patent / tool).
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evoliez import __version__


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


RANKING_FORMULA_VERSION = "2026-05-19.v5"  # bump when scoring math changes


def _git_sha() -> Optional[str]:
    """Exact source revision (with a -dirty suffix for uncommitted changes) so
    a run can be tied to the code that produced it. None outside a git repo."""
    try:
        repo = Path(__file__).resolve().parents[3]
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if sha.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _tool_provenance() -> dict:
    """Real, reconstructible tool state: exact versions of the importable
    scientific stack, plus WHICH binary each CLI-only tool resolved to on PATH
    (a version query could hang, but the resolved path pins the build). Replaces
    the previous static 'adapter (real|mock)' placeholder strings."""
    pkgs = {
        p: _pkg_version(p)
        for p in (
            "boltz", "openmm", "openff-toolkit", "openmmforcefields",
            "rdkit", "torch", "xgboost", "scikit-learn", "pdbfixer",
            "biopython", "numpy", "pandas",
        )
    }
    bins = {
        b: shutil.which(b)
        for b in (
            "boltz", "mmseqs", "jackhmmer", "blastp", "mafft", "foldseek",
            "vina", "gnina", "foldx", "rosetta_scripts",
        )
    }
    return {
        "python_packages": {k: v for k, v in pkgs.items() if v},
        "cli_tools_on_path": {k: v for k, v in bins.items() if v},
    }


def build_provenance(
    *, sequence: str, ligand_smiles: str, config_dict: dict,
    seed: int, backend: str, gnn_ckpt: Optional[str] = None,
) -> dict:
    return {
        "evoliez_version": __version__,
        "git_sha": _git_sha(),
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
        "tools": _tool_provenance(),
        "note": "Boltz/affinity/pLDDT used as features, never labels "
                "(see docs/ML_DATA_POLICY.md).",
    }


def write_provenance(path: Path, prov: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2))
    return path
