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
    ddg = _parse_rosetta(workdir)
    if ddg is None:                       # honest failure, not a fake 0.0
        return {"ddg_fold": None, "clash_score": 0.0}
    return {"ddg_fold": round(ddg, 3), "clash_score": 0.0}


def _parse_rosetta(workdir):
    """cartesian_ddg `*.ddg`: ΔΔG = mean(MUT total) - mean(WT total), reading
    the total score (first float) from each WT/MUT round line. The old code
    averaged ALL numeric tokens in the file, conflating WT vs mutant energies,
    round indices and per-residue terms into a meaningless number. If no WT/MUT
    round labels are present (unrecognised format) it falls back to that mean
    but LOGS a warning, so a format change degrades visibly, not silently."""
    from pathlib import Path

    for f in sorted(Path(workdir).glob("*.ddg")):
        text = f.read_text()
        wt: list[float] = []
        mut: list[float] = []
        for line in text.splitlines():
            val = _first_float(line)
            if val is None:
                continue
            up = line.upper()
            if "MUT" in up:
                mut.append(val)
            elif "WT" in up:
                wt.append(val)
        if wt and mut:
            return sum(mut) / len(mut) - sum(wt) / len(wt)
        nums = [float(x) for x in text.split() if _is_float(x)]
        if nums:
            log.warning(
                "rosetta .ddg has no WT/MUT round labels; falling back to the "
                "mean of %d numeric tokens - verify the cartesian_ddg output "
                "format and the parser", len(nums),
            )
            return sum(nums) / len(nums)
    # No .ddg file / no numeric content: return None (not 0.0, the BEST ddG)
    # so a failed cartesian_ddg run is recorded as 'stability unavailable'.
    log.warning(
        "rosetta produced no parseable *.ddg in %s; stability unavailable "
        "(not scored as max-stable)", workdir,
    )
    return None


def _first_float(line: str):
    """First float token on a line (':' treated as a separator)."""
    for tok in line.replace(":", " ").split():
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False
