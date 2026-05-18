"""Homolog search + MSA (spec sections 7 & 8).

real: MMseqs2 / jackhmmer / BLASTp for search, MAFFT for alignment.
mock: deterministic synthetic homologs at controlled identity, trivially
aligned (same length), so evolutionary features have realistic structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from evoliez.config import Backend, HomologConfig, MSAConfig
from evoliez.logging_utils import get_logger
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.msa")
AA = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class Homolog:
    id: str
    sequence: str
    identity: float
    coverage: float
    annotation: str = ""
    cluster_id: int = 0


# --------------------------------------------------------------------------- #
# Homolog search
# --------------------------------------------------------------------------- #
def search_homologs(
    sequence: str,
    cfg: HomologConfig,
    workdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
) -> List[Homolog]:
    if backend is Backend.real:
        return _search_real(sequence, cfg, workdir, dry_run=dry_run)
    return _search_mock(sequence, cfg)


def _search_real(
    sequence: str, cfg: HomologConfig, workdir: Path, *, dry_run: bool
) -> List[Homolog]:
    workdir.mkdir(parents=True, exist_ok=True)
    query = workdir / "query.fasta"
    query.write_text(f">query\n{sequence}\n")
    out = workdir / "hits.m8"
    if cfg.database is None:
        raise ValueError(
            "homologs.database is required for backend=real "
            "(point it at a UniRef/BFD DB on /mnt/data2)"
        )
    if cfg.method == "mmseqs2":
        require("mmseqs")
        tmp = workdir / "mmseqs_tmp"
        run(
            ["mmseqs", "easy-search", str(query), cfg.database, str(out), str(tmp),
             "--max-seqs", str(cfg.max_sequences), "-e", str(cfg.evalue_max),
             "--format-output",
             "query,target,fident,alnlen,evalue,tseq"],
            dry_run=dry_run,
        )
    elif cfg.method == "jackhmmer":
        require("jackhmmer")
        run(
            ["jackhmmer", "--noali", "-N", "3", "-E", str(cfg.evalue_max),
             "--tblout", str(out), str(query), cfg.database],
            dry_run=dry_run,
        )
    else:  # blastp
        require("blastp")
        run(
            ["blastp", "-query", str(query), "-db", cfg.database,
             "-evalue", str(cfg.evalue_max), "-max_target_seqs",
             str(cfg.max_sequences), "-outfmt",
             "6 sseqid pident qcovs evalue sseq", "-out", str(out)],
            dry_run=dry_run,
        )
    if dry_run or not out.exists():
        return _search_mock(sequence, cfg)
    return _parse_hits(out, cfg)


def _parse_hits(out: Path, cfg: HomologConfig) -> List[Homolog]:
    homologs: List[Homolog] = []
    for i, line in enumerate(out.read_text().splitlines()):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        try:
            ident = float(parts[2])
            ident = ident / 100.0 if ident > 1.0 else ident
            seq = parts[-1].replace("-", "")
        except (ValueError, IndexError):
            continue
        if not (cfg.identity_min <= ident <= cfg.identity_max):
            continue
        homologs.append(
            Homolog(id=f"hit_{i}", sequence=seq, identity=ident, coverage=1.0)
        )
    return homologs[: cfg.max_sequences]


def _search_mock(sequence: str, cfg: HomologConfig) -> List[Homolog]:
    n = min(120, max(24, cfg.max_sequences // 40))
    homologs: List[Homolog] = []
    L = len(sequence)
    for k in range(n):
        # identity sweeps the configured band so strata are non-degenerate
        frac = k / max(1, n - 1)
        ident = cfg.identity_max - frac * (cfg.identity_max - cfg.identity_min)
        seq = list(sequence)
        nmut = int(round((1.0 - ident) * L))
        s = derive_seed(0x5EED, str(k))
        for _ in range(nmut):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            pos = s % L
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            seq[pos] = AA[s % len(AA)]
        homologs.append(
            Homolog(
                id=f"mock_homolog_{k:03d}",
                sequence="".join(seq),
                identity=round(ident, 3),
                coverage=1.0,
                annotation="synthetic",
                cluster_id=k % max(1, cfg.max_sequences // 500 + 1),
            )
        )
    return homologs


# --------------------------------------------------------------------------- #
# MSA
# --------------------------------------------------------------------------- #
def build_msa(
    target_id: str,
    target_seq: str,
    homologs: List[Homolog],
    cfg: MSAConfig,
    workdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
) -> List[Tuple[str, str]]:
    if backend is Backend.real and not cfg.remote_server and not dry_run:
        return _align_real(target_id, target_seq, homologs, workdir, dry_run=dry_run)
    # mock / dry-run / remote_server fallback: synthetic homologs are equal
    # length to the target, so the alignment is the identity alignment.
    msa: List[Tuple[str, str]] = [(target_id, target_seq)]
    for h in homologs:
        s = h.sequence
        if len(s) < len(target_seq):
            s = s + "-" * (len(target_seq) - len(s))
        msa.append((h.id, s[: len(target_seq)]))
    return msa


def _align_real(
    target_id: str,
    target_seq: str,
    homologs: List[Homolog],
    workdir: Path,
    *,
    dry_run: bool,
) -> List[Tuple[str, str]]:
    require("mafft")
    workdir.mkdir(parents=True, exist_ok=True)
    fin = workdir / "to_align.fasta"
    fout = workdir / "alignment.fasta"
    with fin.open("w") as fh:
        fh.write(f">{target_id}\n{target_seq}\n")
        for h in homologs:
            fh.write(f">{h.id}\n{h.sequence}\n")
    res = run(["mafft", "--auto", "--anysymbol", str(fin)], dry_run=dry_run)
    if dry_run:
        return [(target_id, target_seq)]
    fout.write_text(res.stdout)
    return _read_fasta_aln(fout, target_id)


def _read_fasta_aln(path: Path, target_id: str) -> List[Tuple[str, str]]:
    seqs: List[Tuple[str, str]] = []
    cid, buf = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cid is not None:
                seqs.append((cid, "".join(buf)))
            cid, buf = line[1:].strip().split()[0], []
        else:
            buf.append(line.strip())
    if cid is not None:
        seqs.append((cid, "".join(buf)))
    seqs.sort(key=lambda kv: 0 if kv[0] == target_id else 1)
    return seqs
