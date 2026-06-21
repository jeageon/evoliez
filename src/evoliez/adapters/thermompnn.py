"""Mutation stability ΔΔG via ThermoMPNN — open-source FoldX/Rosetta replacement.

ThermoMPNN (Kuhlman-Lab, MIT, PNAS 2024) is a ProteinMPNN-based ΔΔG-of-folding
predictor. FoldX and Rosetta both need licenses and are absent here, so this is the
open, paper-citable stability signal selected by ``validation.stability.method:
thermompnn``.

real: shell out to ``scripts/thermompnn_ssm.py`` in the isolated ``thermompnn`` conda
env (``EVOLIEZ_THERMOMPNN_PYTHON`` interpreter, ``EVOLIEZ_THERMOMPNN`` repo). That
runs ONE site-saturation pass on the WT structure -> a ΔΔG table keyed by author
residue number + mutant AA, cached per WT pdb (so 467 candidates trigger one SSM, not
467 model loads). A candidate's ΔΔG is the ADDITIVE sum of its single-substitution
ΔΔGs — the sanctioned multi-site baseline (ThermoMPNN scores substitutions
independently, like FoldX for independent sites). Sign: positive = destabilizing
(FoldX convention, so this is a sign-preserving drop-in for _instability / the s09
MD-selection ``- ddG`` term).

CAVEAT: ThermoMPNN sees the protein backbone only — co-modelled NADP/formate are
invisible. It scores FOLD stability, complementary to the redock-consistency + real
mutant-Boltz Δ signals (which DO see the ligand). Document as such in the paper.

mock: the shared chemistry proxy (``foldx._mock``).
Honest degradation: tool / full-atom WT / table unavailable, or a mutation absent from
the table (wildtype mismatch), -> ``ddg_fold=None`` (NEVER a fabricated 0.0); s09 marks
it ``stability_unavailable`` and routes the candidate to MD instead of rewarding it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

from evoliez.adapters.base import full_atom_receptor_pdb
from evoliez.adapters.foldx import _mock
from evoliez.config import Backend, StabilityConfig
from evoliez.logging_utils import get_logger
from evoliez.types import Mutation, ProteinStructure

log = get_logger("evoliez.thermompnn")

# WT pdb path -> SSM table {resnum: {"wt": "A", "ddg": {mut: ΔΔG}}}. The WT is the
# same for every candidate in a run, so the SSM runs once and the rest hit this cache.
_SSM_CACHE: Dict[str, dict] = {}


def estimate_stability(
    candidate_id: str,
    structure: ProteinStructure,
    mutations: Sequence[Mutation],
    cfg: StabilityConfig,
    workdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
) -> Dict[str, float]:
    if backend is not Backend.real or cfg.method != "thermompnn":
        return _mock(candidate_id, structure, mutations)
    table = _ssm_table(structure, Path(workdir), dry_run=dry_run)
    if table is None:
        if dry_run:
            return _mock(candidate_id, structure, mutations)
        log.warning("thermompnn: SSM table unavailable for %s; ddG unavailable",
                    candidate_id)
        return {"ddg_fold": None, "clash_score": 0.0}
    total = 0.0
    for m in mutations:
        rec = table.get(str(m.position))
        if rec is None or rec.get("wt") != m.wt or m.mut not in rec.get("ddg", {}):
            log.warning(
                "thermompnn: %s mutation %s%s%s not in SSM table (wildtype "
                "mismatch / position missing); ddG unavailable",
                candidate_id, m.wt, m.position, m.mut,
            )
            return {"ddg_fold": None, "clash_score": 0.0}
        total += rec["ddg"][m.mut]
    return {"ddg_fold": round(total, 3), "clash_score": 0.0}


def _runner_path() -> str:
    # scripts/thermompnn_ssm.py at the repo root (src/evoliez/adapters/ -> repo).
    default = Path(__file__).resolve().parents[3] / "scripts" / "thermompnn_ssm.py"
    return os.environ.get("EVOLIEZ_THERMOMPNN_RUNNER", str(default))


def _ssm_table(structure: ProteinStructure, workdir: Path, *, dry_run: bool) -> Optional[dict]:
    """Run (once, cached) the ThermoMPNN site-saturation on the WT structure and
    return its ΔΔG table. None when no full-atom WT pdb / the tool is unavailable /
    the run fails (honest — never a fabricated table)."""
    workdir.mkdir(parents=True, exist_ok=True)
    wt_pdb = workdir / "wt_thermompnn.pdb"
    # Protein-only WT (HETATM dropped) — ThermoMPNN is protein-backbone only anyway.
    if not full_atom_receptor_pdb(structure, wt_pdb):
        return None
    key = str(wt_pdb)
    if key in _SSM_CACHE:
        return _SSM_CACHE[key]
    out_json = workdir / "thermompnn_ssm.json"
    if out_json.exists():                       # reuse across resumes (WT-only, deterministic)
        table = json.loads(out_json.read_text()).get("by_resnum", {})
        _SSM_CACHE[key] = table
        return table
    if dry_run:
        return None
    from evoliez.utils.subprocess_utils import run
    repo = os.environ.get("EVOLIEZ_THERMOMPNN")
    py = os.environ.get("EVOLIEZ_THERMOMPNN_PYTHON")
    if not repo or not py or not Path(py).exists():
        log.warning("thermompnn: EVOLIEZ_THERMOMPNN / EVOLIEZ_THERMOMPNN_PYTHON unset "
                    "or interpreter missing; ddG unavailable")
        return None
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "disabled")
    try:
        run([py, _runner_path(), str(wt_pdb), "A", str(out_json), repo],
            cwd=repo, env=env, dry_run=False)
    except Exception as exc:  # noqa: BLE001 - degrade honestly on any runner failure
        log.warning("thermompnn SSM run failed (%s); ddG unavailable", exc)
        return None
    if not out_json.exists():
        return None
    table = json.loads(out_json.read_text()).get("by_resnum", {})
    _SSM_CACHE[key] = table
    return table
