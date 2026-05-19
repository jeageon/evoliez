"""Optional Rosetta cartesian_ddG stability backend (spec section 14.1).

Provided for parity with the spec. Selected via ``stability.method: rosetta``.
real: cartesian_ddg protocol. Falls back to the FoldX-style mock otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from evoliez.adapters.base import write_min_pdb
from evoliez.adapters.foldx import _mock as _ddg_mock
from evoliez.config import Backend
from evoliez.logging_utils import get_logger
from evoliez.types import Mutation, ProteinStructure
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.rosetta")


def estimate_stability(
    candidate_id: str,
    structure: ProteinStructure,
    mutations: Sequence[Mutation],
    workdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
) -> Dict[str, float]:
    if backend is not Backend.real:
        return _ddg_mock(candidate_id, structure, mutations)
    require("cartesian_ddg.default.linuxgccrelease")
    workdir.mkdir(parents=True, exist_ok=True)
    pdb = workdir / f"{candidate_id}.pdb"
    write_min_pdb(pdb, structure)
    muts = workdir / "mutations.txt"
    muts.write_text(
        "total {}\n".format(len(mutations))
        + "".join(f"{m.wt} {m.position} {m.mut}\n" for m in mutations)
    )
    run(
        ["cartesian_ddg.default.linuxgccrelease", "-s", str(pdb),
         "-ddg:mut_file", str(muts), "-ddg:iterations", "3",
         "-fa_max_dis", "9.0", "-out:path:all", str(workdir)],
        dry_run=dry_run,
    )
    if dry_run:
        return _ddg_mock(candidate_id, structure, mutations)
    return {"ddg_fold": round(_parse_rosetta(workdir), 3),
            "clash_score": 0.0}


def _parse_rosetta(workdir) -> float:
    """Mean of the numeric tokens in cartesian_ddg `*.ddg` output."""
    from pathlib import Path

    for f in Path(workdir).glob("*.ddg"):
        nums = [float(x) for x in f.read_text().split() if _is_float(x)]
        if nums:
            return sum(nums) / len(nums)
    return 0.0


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False
