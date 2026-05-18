"""Ligand-aware sequence design (spec section 12.3).

real: LigandMPNN (https://github.com/dauparas/LigandMPNN).
mock: deterministic position-wise sampling over designable residues, biased
toward chemically sensible substitutions, so candidates are reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

from evoliez.adapters.base import write_min_pdb
from evoliez.config import Backend, MutationGenConfig
from evoliez.logging_utils import get_logger
from evoliez.types import Complex, Mutation
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.ligandmpnn")

AA = "ACDEFGHIKLMNPQRSTVWY"
# coarse chemistry-aware preference pools (spec 12.4)
_POOL = {
    "anionic_contact": "KRHST",
    "aromatic_contact": "FYWH",
    "hydrophobic": "LIVMF",
    "polar": "STNQYH",
}


def design_sequences(
    cx: Complex,
    designable: Sequence[int],
    cfg: MutationGenConfig,
    workdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
) -> List[Tuple[List[Mutation], float]]:
    if backend is Backend.real:
        return _design_real(cx, designable, cfg, workdir, dry_run=dry_run)
    return _design_mock(cx, designable, cfg)


def _design_real(
    cx: Complex,
    designable: Sequence[int],
    cfg: MutationGenConfig,
    workdir: Path,
    *,
    dry_run: bool,
) -> List[Tuple[List[Mutation], float]]:
    require("python")  # LigandMPNN is invoked via its run.py
    apply_gpu_selection()
    workdir.mkdir(parents=True, exist_ok=True)
    pdb = workdir / "input_complex.pdb"
    write_min_pdb(pdb, cx.structure, cx.ligand.atoms)
    fixed = sorted(set(r.index for r in cx.structure.residues) - set(designable))
    fixed_str = " ".join(f"A{p}" for p in fixed)
    out = workdir / "lmpnn_out"
    cmd = [
        "python", "run.py",
        "--model_type", "ligand_mpnn",
        "--pdb_path", str(pdb),
        "--out_folder", str(out),
        "--number_of_batches", str(max(1, cfg.ligandmpnn_samples // 8)),
        "--batch_size", "8",
        "--temperature", str(cfg.ligandmpnn_temperature),
    ]
    if fixed_str:
        cmd += ["--fixed_residues", fixed_str]
    run(cmd, dry_run=dry_run)
    if dry_run:
        return _design_mock(cx, designable, cfg)
    return _parse_lmpnn(out, cx.structure.sequence)


def _parse_lmpnn(
    out: Path, wt: str
) -> List[Tuple[List[Mutation], float]]:
    results: List[Tuple[List[Mutation], float]] = []
    for fa in sorted(out.rglob("*.fa")) + sorted(out.rglob("*.fasta")):
        header, seq = None, []
        for line in fa.read_text().splitlines():
            if line.startswith(">"):
                if header is not None and seq:
                    results.append(_diff("".join(seq), wt, header))
                header, seq = line, []
            else:
                seq.append(line.strip())
        if header is not None and seq:
            results.append(_diff("".join(seq), wt, header))
    return [r for r in results if r[0]]


def _diff(seq: str, wt: str, header: str) -> Tuple[List[Mutation], float]:
    muts = [
        Mutation(wt=wt[i], position=i + 1, mut=seq[i])
        for i in range(min(len(seq), len(wt)))
        if seq[i] != wt[i] and seq[i] in AA
    ]
    score = 0.0
    if "score=" in header:
        try:
            score = -float(header.split("score=")[1].split(",")[0])
        except Exception:
            score = 0.0
    return muts, score


def _design_mock(
    cx: Complex, designable: Sequence[int], cfg: MutationGenConfig
) -> List[Tuple[List[Mutation], float]]:
    seq = cx.structure.sequence
    by_idx = {r.index: r for r in cx.structure.residues}
    designable = list(designable)
    results: List[Tuple[List[Mutation], float]] = []
    n = cfg.ligandmpnn_samples
    for k in range(n):
        s = derive_seed(0x11AC, str(k))
        # 1-3 substitutions per sample
        npos = 1 + (s % 3)
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        chosen = []
        pool = designable[:] or [r.index for r in cx.structure.residues]
        for _ in range(min(npos, len(pool))):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            chosen.append(pool[s % len(pool)])
        muts: List[Mutation] = []
        for pos in sorted(set(chosen)):
            r = by_idx.get(pos)
            if r is None:
                continue
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            pref = _POOL["polar"] if r.sasa < 0.4 else _POOL["hydrophobic"]
            newaa = pref[s % len(pref)]
            if newaa != r.aa:
                muts.append(Mutation(wt=r.aa, position=pos, mut=newaa))
        if muts:
            logp = -0.1 * len(muts) - (s % 100) / 1000.0
            results.append((muts, round(logp, 4)))
    # de-duplicate by mutation string
    seen = {}
    for muts, lp in results:
        key = ";".join(str(m) for m in muts)
        if key not in seen or lp > seen[key][1]:
            seen[key] = (muts, lp)
    return list(seen.values())
