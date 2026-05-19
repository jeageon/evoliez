"""Mutation stability (spec section 14.1).

real: FoldX BuildModel (academic license; see docs/DATA_AND_WEIGHTS.md).
mock: a chemistry-informed deterministic ddG proxy (hydrophobicity / volume /
charge mismatch) - not physical, but monotonic and reproducible for screening.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

from evoliez.adapters.base import write_min_pdb
from evoliez.config import Backend, StabilityConfig
from evoliez.logging_utils import get_logger
from evoliez.types import Mutation, ProteinStructure
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.foldx")

# Kyte-Doolittle hydropathy and approximate side-chain volume (A^3).
_HYDRO = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
_VOL = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}
_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}


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
    if backend is Backend.real and cfg.method == "foldx":
        return _foldx_real(
            candidate_id, structure, mutations, workdir, dry_run=dry_run
        )
    return _mock(candidate_id, structure, mutations)


def _foldx_real(
    candidate_id: str,
    structure: ProteinStructure,
    mutations: Sequence[Mutation],
    workdir: Path,
    *,
    dry_run: bool,
) -> Dict[str, float]:
    require("foldx")
    workdir.mkdir(parents=True, exist_ok=True)
    pdb = workdir / f"{candidate_id}.pdb"
    write_min_pdb(pdb, structure)
    mut_file = workdir / "individual_list.txt"
    mut_file.write_text(
        ",".join(f"{m.wt}A{m.position}{m.mut}" for m in mutations) + ";\n"
    )
    run(
        ["foldx", "--command=BuildModel", f"--pdb={pdb.name}",
         f"--mutant-file={mut_file.name}", "--output-dir", str(workdir)],
        cwd=workdir, dry_run=dry_run,
    )
    if dry_run:
        return _mock(candidate_id, structure, mutations)
    return {"ddg_fold": round(_parse_foldx(workdir), 3), "clash_score": 0.0}


def _parse_foldx(workdir) -> float:
    """FoldX BuildModel `Dif_*.fxout`: skip the header block, the first
    numeric column of a data row is total ΔΔG (kcal/mol)."""
    from pathlib import Path

    for f in Path(workdir).glob("Dif_*.fxout"):
        for line in f.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) > 1:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return 0.0


def _mock(
    candidate_id: str, structure: ProteinStructure, mutations: Sequence[Mutation]
) -> Dict[str, float]:
    by_idx = {r.index: r for r in structure.residues}
    ddg = 0.0
    clash = 0.0
    for m in mutations:
        dh = abs(_HYDRO.get(m.wt, 0) - _HYDRO.get(m.mut, 0)) / 9.0
        dv = abs(_VOL.get(m.wt, 120) - _VOL.get(m.mut, 120)) / 170.0
        dc = abs(_CHARGE.get(m.wt, 0) - _CHARGE.get(m.mut, 0))
        r = by_idx.get(m.position)
        buried = (1.0 - r.sasa) if r is not None else 0.5
        contrib = 1.4 * dv * (0.5 + buried) + 0.8 * dh * buried + 0.6 * dc
        seed = derive_seed(0xF01D, candidate_id, str(m))
        contrib += (seed % 60) / 100.0 - 0.3
        ddg += contrib
        clash += max(0.0, dv - 0.4) * buried
    return {
        "ddg_fold": round(ddg, 3),
        "clash_score": round(clash, 3),
        "buried_unsat": round(clash * 0.5, 3),
    }
