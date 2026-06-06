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
    out = (workdir / "aln.sto" if cfg.method == "jackhmmer"
           else workdir / "hits.m8")

    # dry-run previews the command without needing the DB/tool installed.
    # The missing-database hard check belongs to a real run (and to
    # `evoliez doctor`), NOT to dry-run preview - otherwise --dry-run UX
    # breaks for the default example config.
    if dry_run:
        db = cfg.database or "<HOMOLOG_DB: set homologs.database>"
        preview = {
            "mmseqs2": ["mmseqs", "easy-search", str(query), db,
                        str(out), str(workdir / "mmseqs_tmp")],
            "jackhmmer": ["jackhmmer", "-N", "3", "-A", str(out), str(query),
                          db],
        }.get(cfg.method, ["blastp", "-query", str(query), "-db", db,
                            "-out", str(out)])
        run(preview, dry_run=True)
        if cfg.database is None:
            log.warning(
                "[dry-run] homologs.database is unset - a REAL run requires "
                "it (or set msa.remote_server: true)"
            )
        return _search_mock(sequence, cfg)

    if cfg.database is None:
        raise ValueError(
            "homologs.database is required for backend=real "
            "(point it at a UniRef/BFD DB on /mnt/data2), "
            "or set msa.remote_server: true"
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
        # -A writes the hit MSA (Stockholm); --tblout alone has NO aligned
        # sequences, so the old shared parser produced invalid homologs.
        run(
            ["jackhmmer", "-N", "3", "-E", str(cfg.evalue_max),
             "-A", str(out), "--tblout", str(workdir / "hits.tbl"),
             str(query), cfg.database],
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
    return _parse_hits(out, cfg, sequence)


def _band(cfg: HomologConfig, ident: float) -> bool:
    return cfg.identity_min <= ident <= cfg.identity_max


def _kmers(seq: str, k: int = 4) -> set:
    s = seq.replace("-", "").replace(".", "").upper()
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def assign_clusters(homologs: List[Homolog], cfg: HomologConfig) -> List[Homolog]:
    """Greedy k-mer (k=4) Jaccard clustering -> subfamily ``cluster_id``.

    The real search parsers return homologs WITHOUT a cluster_id (it defaults
    to 0), which collapses ``s06b._pick_representatives`` to a single
    representative and silently turns the family-geometry ensemble into
    'top-N by identity'. This assigns real subfamily clusters with NO external
    tool (works for mmseqs/blast/jackhmmer alike): the highest-identity
    homologs seed clusters; each remaining sequence joins the most-similar
    existing cluster when its k-mer Jaccard clears a threshold derived from
    ``cfg.cluster_identity``, else it opens a new cluster. Runs on both
    backends so mock exercises the same code path (no more hidden real-only
    regression). For a more rigorous pass, ``mmseqs easy-cluster`` could
    replace this, but this keeps the dependency closure light.
    """
    if not homologs:
        return homologs
    k = 4
    p = min(0.99, max(0.05, cfg.cluster_identity))
    jthr = (p ** k) / (2.0 - p ** k)          # pairwise identity p -> Jaccard
    order = sorted(range(len(homologs)),
                   key=lambda i: homologs[i].identity, reverse=True)
    kmer_cache = {i: _kmers(homologs[i].sequence, k) for i in order}
    centroids: List[set] = []
    cluster_of: dict = {}
    for i in order:
        km = kmer_cache[i]
        best_c, best_j = -1, 0.0
        for c, cen in enumerate(centroids):
            inter = len(km & cen)
            if not inter:
                continue
            j = inter / (len(km) + len(cen) - inter)
            if j > best_j:
                best_c, best_j = c, j
        if best_c >= 0 and best_j >= jthr:
            cluster_of[i] = best_c
        else:
            cluster_of[i] = len(centroids)
            centroids.append(km)
    for i, h in enumerate(homologs):
        h.cluster_id = cluster_of[i]
    log.info(
        "clustered %d homologs into %d subfamily group(s) "
        "(k-mer Jaccard >= %.2f ~ %.0f%% identity)",
        len(homologs), len(centroids), jthr, p * 100,
    )
    return homologs


def _pairwise_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i] and a[i] != "-")
    cols = sum(1 for i in range(n) if a[i] != "-" or b[i] != "-")
    return same / cols if cols else 0.0


def _parse_mmseqs_m8(out: Path, cfg: HomologConfig) -> List[Homolog]:
    """format-output: query,target,fident,alnlen,evalue,tseq"""
    hs: List[Homolog] = []
    for i, line in enumerate(out.read_text().splitlines()):
        p = line.rstrip("\n").split("\t")
        if len(p) < 6:
            continue
        try:
            ident = float(p[2])
        except ValueError:
            continue
        ident = ident / 100.0 if ident > 1.0 else ident
        seq = p[5].replace("-", "")
        if seq and _band(cfg, ident):
            hs.append(Homolog(f"mm_{i}", seq, round(ident, 4), 1.0))
    return hs[: cfg.max_sequences]


def _parse_blast_m8(out: Path, cfg: HomologConfig) -> List[Homolog]:
    """outfmt 6: sseqid pident qcovs evalue sseq  (pident is col 1!)"""
    hs: List[Homolog] = []
    for i, line in enumerate(out.read_text().splitlines()):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5:
            continue
        try:
            pident = float(p[1])
            cov = float(p[2])
        except ValueError:
            continue
        ident = pident / 100.0 if pident > 1.0 else pident
        seq = p[4].replace("-", "")
        if seq and _band(cfg, ident):
            hs.append(Homolog(f"bl_{i}", seq, round(ident, 4),
                              round(cov / 100.0 if cov > 1.0 else cov, 4)))
    return hs[: cfg.max_sequences]


def _parse_stockholm(out: Path, cfg: HomologConfig,
                     query_seq: str) -> List[Homolog]:
    """jackhmmer -A Stockholm MSA: aggregate aligned rows per sequence,
    identity computed vs the query."""
    rows: dict[str, str] = {}
    for line in out.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s == "//":
            continue
        parts = s.split()
        if len(parts) != 2:
            continue
        name, aln = parts
        rows[name] = rows.get(name, "") + aln
    hs: List[Homolog] = []
    for i, (name, aln) in enumerate(rows.items()):
        seq = aln.replace("-", "").replace(".", "").upper()
        if not seq:
            continue
        ident = _pairwise_identity(seq, query_seq)
        if _band(cfg, ident):
            hs.append(Homolog(f"jh_{i}", seq, round(ident, 4), 1.0))
    return hs[: cfg.max_sequences]


def _parse_hits(out: Path, cfg: HomologConfig,
                query_seq: str = "") -> List[Homolog]:
    if cfg.method == "mmseqs2":
        hs = _parse_mmseqs_m8(out, cfg)
    elif cfg.method == "jackhmmer":
        hs = _parse_stockholm(out, cfg, query_seq)
    else:
        hs = _parse_blast_m8(out, cfg)
    return assign_clusters(hs, cfg)


def _search_mock(sequence: str, cfg: HomologConfig) -> List[Homolog]:
    n = min(120, max(24, cfg.max_sequences // 40))
    homologs: List[Homolog] = []
    L = len(sequence)
    # Deterministic synthetic subfamily count, INDEPENDENT of max_sequences
    # (the old k%(max_sequences//500+1) tied subfamily structure to an
    # unrelated search-size knob, so a small-search smoke config silently
    # collapsed to one cluster). Always >= 3 so the s06b subfamily-holdout
    # validator has groups to hold out. The synthetic sweep doesn't form real
    # k-mer subfamilies, so cluster_id is assigned directly here; assign_clusters
    # (the real-backend path) is covered by its own unit test.
    n_subfam = max(3, min(8, n // 8))
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
                cluster_id=k % n_subfam,
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
