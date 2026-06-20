"""Homolog search + MSA (spec sections 7 & 8).

real: MMseqs2 / jackhmmer / BLASTp for search, MAFFT for alignment.
mock: deterministic synthetic homologs at controlled identity, trivially
aligned (same length), so evolutionary features have realistic structure.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from evoliez.config import Backend, HomologConfig, MSAConfig
from evoliez.logging_utils import get_logger
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run, tool_env

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
    source: str = "sequence"   # sequence | structure (which retriever found it)
    # Target-column-anchored alignment row (length == target length): set by the
    # Foldseek (structural) and local-mmseqs (sequence) parsers from each hit's
    # NATIVE alignment, so s03 can stack the structure + local tracks onto the
    # ColabFold a3m in ONE target-column MSA (user §7). None when no native
    # alignment was captured (ColabFold-mined / mock / jackhmmer / blast).
    aligned: Optional[str] = None


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
    remote_server: bool = False,
) -> List[Homolog]:
    if backend is Backend.real:
        return _search_real(sequence, cfg, workdir, dry_run=dry_run,
                            remote_server=remote_server)
    return _search_mock(sequence, cfg)


# Concrete sequence-search tools that can each run as an INDEPENDENT first-class
# source (its own DB via cfg.databases[tool]); hhblits has its own adapter.
_SEQ_TOOLS = {"mmseqs2", "jackhmmer", "blastp"}


def _db_for(cfg: HomologConfig, tool: str) -> Optional[str]:
    """Per-tool DB (homologs.databases[tool]) else the shared homologs.database."""
    return (cfg.databases or {}).get(tool) or cfg.database


def _cache_key(sequence: str, source: str, cfg: HomologConfig) -> str:
    """Content hash of the inputs that determine THIS track's homolog result
    (query + its DB + its search params), so the cache is reused across runs and
    unrelated config tweaks, and invalidated only when this track's inputs change."""
    parts = [source, sequence, _db_for(cfg, source) or "", str(cfg.evalue_max),
             str(cfg.identity_min), str(cfg.identity_max),
             str(cfg.max_sequences), str(cfg.cluster_identity)]
    if source == "mmseqs2":
        parts.append(str(cfg.mmseqs_gpu))
    elif source == "jackhmmer":
        parts.append(str(cfg.jackhmmer_chunks))
    elif source == "hhblits":
        parts.append(str(cfg.hhblits_iterations))
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:16]


def _cache_path(workdir: Path, source: str, sequence: str = "",
                cfg: Optional[HomologConfig] = None) -> Path:
    """Where a source's homolog cache lives. With cfg.cache_dir set -> a
    PERSISTENT, content-keyed file (survives run-dir purges); else the run-dir
    file (lost when the run fingerprint changes)."""
    if cfg is not None and cfg.cache_dir:
        return (Path(cfg.cache_dir)
                / f"{source}.{_cache_key(sequence, source, cfg)}.json")
    return workdir / f"_cache_{source}.json"


def _save_homologs(homologs: List[Homolog], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(h) for h in homologs]))


def _load_homologs(path: Path) -> List[Homolog]:
    return [Homolog(**d) for d in json.loads(path.read_text())]


def gather_homologs(
    sequence: str,
    cfg: HomologConfig,
    workdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
    remote_server: bool = False,
    msa_dir: "Path | None" = None,
) -> List[Homolog]:
    """Integrated multi-source homolog acquisition (roadmap P1.1).

    Runs each source in ``cfg.sources`` (``sequence`` = local mmseqs/jackhmmer/
    blastp DB OR ColabFold remote MSA when ``remote_server``; ``structure`` =
    Foldseek) and MERGES the results into one homolog set with a unified
    subfamily cluster-id space, so s03's MSA and s06b's family model see
    structure + sequence homologs together. A single source returns unchanged
    (backward compatible). ``msa_dir`` shares the ColabFold a3m with s03."""
    # Concrete, INDEPENDENT retrievers (each stands on its own, all merge):
    #   local     - local mmseqs/jackhmmer/blastp vs homologs.database
    #   colabfold - ColabFold remote MSA (sequence; no local DB)
    #   foldseek  - Foldseek structural homologs
    # Abstract aliases keep old configs working: 'sequence' -> local OR
    # colabfold (by remote_server), 'structure' -> foldseek.
    raw = list(cfg.sources or ["sequence"])
    if cfg.use_foldseek and not ({"structure", "foldseek"} & set(raw)):
        raw.append("structure")
    norm: List[str] = []
    for s in raw:
        norm.append({"sequence": "colabfold" if remote_server else "local",
                     "structure": "foldseek"}.get(s, s))
    norm = list(dict.fromkeys(norm))                 # dedupe, keep order

    # Per-source result cache: each track is an INDEPENDENT retriever, so cache
    # its homologs and REUSE them on re-run — the finished tracks stay put and
    # only the missing/slow one is (re)computed (delete _cache_<source>.json to
    # force a recompute). Lets us nail one track at a time (user §7) instead of
    # re-running all five whenever one needs another attempt.
    def _gather_one(s: str) -> List[Homolog]:
        """Acquire ONE track's homologs (cache hit OR run+save). Each track
        writes to its OWN workdir/<name> subdir and OWN cache file (norm is
        deduped above), so this is safe to run on a worker thread with no
        shared-state hazard."""
        cache = _cache_path(workdir, s, sequence, cfg)
        if not dry_run and cache.exists():
            hs = _load_homologs(cache)
            log.info("reusing cached '%s' track: %d homologs (rm %s to recompute)",
                     s, len(hs), cache.name)
            return hs
        hs = _run_source(s, sequence, cfg, workdir, backend=backend,
                         dry_run=dry_run, remote_server=remote_server,
                         msa_dir=msa_dir)
        if hs and not dry_run and backend is Backend.real:
            _save_homologs(hs, cache)
        return hs

    # The 5 tracks are fully independent subprocess searches (no data dependency;
    # _merge_sources runs only after all complete), so run them on a thread pool
    # — each track blocks in subprocess.run/Popen, which releases the GIL, so
    # cold-cache wall time becomes ~MAX(tracks) instead of SUM(tracks).
    #
    # SHARED-SERVER CPU WATCHDOG: the box throttles a job to ~0.2 core if it
    # pins ~48 cores for 10+ min, and several tracks fan OUT internally — a
    # heavy track (mmseqs2/jackhmmer/hhblits/blastp/local) each consumes up to
    # cfg.search_threads cores (jackhmmer_chunks + hhblits EACH spawn
    # search_threads workers). Running all 5 concurrently would multiply that.
    # So bound total worker threads to <= ~16 cores (project memory: keep heavy
    # jobs <= 16-24 cores) by capping concurrency at 16 // search_threads, i.e.
    # max_workers * search_threads <= 16 regardless of the search_threads knob
    # (default 4 -> 4 concurrent tracks). Hardcoded cap (no config field).
    _WATCHDOG_CORE_CAP = 16
    threads = max(1, getattr(cfg, "search_threads", 4) or 1)
    max_workers = max(1, min(len(norm), _WATCHDOG_CORE_CAP // threads))

    results: dict[str, List[Homolog]] = {}
    if max_workers <= 1 or len(norm) <= 1:
        for s in norm:
            results[s] = _gather_one(s)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for s, hs in zip(norm, ex.map(_gather_one, norm)):
                results[s] = hs

    # Re-assemble in norm order, then drop empties — IDENTICAL ordering/merge
    # semantics to the old serial `for s in norm` loop (only wall-time differs).
    per_source: List[Tuple[str, List[Homolog]]] = [
        (s, results[s]) for s in norm if results[s]
    ]
    if not per_source:                               # nothing enabled / found
        return search_homologs(sequence, cfg, workdir, backend=backend,
                               dry_run=dry_run, remote_server=remote_server)
    if len(per_source) == 1:
        return per_source[0][1]
    return _merge_sources(per_source)


def _run_source(
    name: str, sequence: str, cfg: HomologConfig, workdir: Path, *,
    backend: Backend, dry_run: bool, remote_server: bool, msa_dir,
) -> List[Homolog]:
    """One independent homolog retriever -> tagged Homolog list ([] if it
    cannot contribute, e.g. local search without a DB)."""
    if name == "colabfold":
        if (remote_server and backend is Backend.real and not dry_run
                and msa_dir is not None):
            hs = _colabfold_homologs(sequence, msa_dir, cfg)
            for h in hs:
                h.source = "sequence"
            return hs
        return []
    if name in _SEQ_TOOLS:                       # mmseqs2 | jackhmmer | blastp
        db = _db_for(cfg, name)
        if backend is Backend.real and not dry_run and db is None:
            log.warning(
                "homolog source %r has no database (set homologs.databases.%s "
                "or homologs.database); skipping", name, name,
            )
            return []
        # Independent track: search this ONE tool against its OWN DB/format,
        # reusing the full _search_real path via a per-tool config copy.
        tcfg = cfg.model_copy(update={"method": name, "database": db})
        hs = search_homologs(sequence, tcfg, workdir / name, backend=backend,
                             dry_run=dry_run, remote_server=False)
        tag = {"mmseqs2": "mmseqs"}.get(name, name)
        for h in hs:
            h.source = "sequence"
            if not h.annotation:                 # tag the tool (keep 'synthetic')
                h.annotation = tag
        return hs
    if name == "hhblits":
        from evoliez.adapters.hhblits import search_hhblits_homologs
        return search_hhblits_homologs(sequence, cfg, workdir / "hhblits",
                                       backend=backend, dry_run=dry_run)
    if name == "local":
        if backend is Backend.real and cfg.database is None:
            log.warning(
                "homologs.sources includes a local sequence search but "
                "homologs.database is unset; skipping (no UniRef/BFD to mine). "
                "Download a DB + set homologs.database for an independent "
                "local mmseqs/jackhmmer/blastp search."
            )
            return []
        hs = search_homologs(sequence, cfg, workdir / "local", backend=backend,
                             dry_run=dry_run, remote_server=False)
        meth = {"mmseqs2": "mmseqs"}.get(cfg.method, cfg.method)
        for h in hs:
            h.source = "sequence"
            if not h.annotation:                     # tag the tool (keep 'synthetic')
                h.annotation = meth
        return hs
    if name == "foldseek":
        from evoliez.adapters.foldseek import search_structural_homologs
        return search_structural_homologs(sequence, cfg, workdir / "foldseek",
                                          backend=backend, dry_run=dry_run)
    log.warning("unknown homolog source %r; skipping", name)
    return []


def _merge_sources(per_source: List[Tuple[str, List["Homolog"]]]) -> List[Homolog]:
    """Dedupe by sequence (keep the higher-identity hit) and map each source's
    cluster ids into ONE global id space. We deliberately do NOT re-run k-mer
    clustering on the merged set: it would collapse the equal-length synthetic
    mock homologs into a single group and break the s06b subfamily holdout."""
    best: dict[str, Homolog] = {}
    order: List[str] = []
    global_of: dict[Tuple[str, int], int] = {}
    for src, homs in per_source:
        for h in homs:
            key = h.sequence
            if key in best and best[key].identity >= h.identity:
                continue
            gkey = (src, h.cluster_id)
            if gkey not in global_of:
                global_of[gkey] = len(global_of)
            # keep h.source (sequence|structure category) + h.annotation (tool)
            # as set by the retriever; `src` (retriever name) only keys the
            # global cluster space so each retriever's subfamilies stay distinct.
            h.cluster_id = global_of[gkey]
            if key not in best:
                order.append(key)
            best[key] = h
    merged = [best[k] for k in order]
    log.info(
        "merged homologs (%s) -> %d unique across %d subfamily group(s)",
        ", ".join(f"{src}={len(h)}" for src, h in per_source),
        len(merged), len({h.cluster_id for h in merged}),
    )
    return merged


def _colabfold_homologs(
    sequence: str, msa_dir: Path, cfg: HomologConfig
) -> List[Homolog]:
    """Real SEQUENCE homologs mined from the ColabFold remote MSA (no local DB).
    Each aligned row becomes a Homolog with identity computed vs the query row;
    returns [] if the MSA is unavailable (no network) so the caller falls back."""
    from evoliez.adapters.remote_msa import cached_fetch_msa

    msa = cached_fetch_msa(sequence, msa_dir)
    if not msa or len(msa) < 2:
        return []
    query_aln = msa[0][1]
    hs: List[Homolog] = []
    for i, (_, aln) in enumerate(msa[1:]):
        seq = aln.replace("-", "").replace(".", "").upper()
        if not seq:
            continue
        ident = _aligned_identity(aln, query_aln)
        if _band(cfg, ident):
            hs.append(Homolog(f"cf_{i}", seq, round(ident, 4), 1.0,
                              annotation="colabfold", source="sequence"))
    hs = hs[: cfg.max_sequences]
    log.info("ColabFold remote MSA -> %d sequence homologs", len(hs))
    return assign_clusters(hs, cfg)


def _search_real(
    sequence: str, cfg: HomologConfig, workdir: Path, *, dry_run: bool,
    remote_server: bool = False,
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
        if cfg.method == "mmseqs2" and cfg.mmseqs_gpu:
            preview = preview + ["--gpu", "1"]
        run(preview, dry_run=True)
        if cfg.database is None:
            log.warning(
                "[dry-run] homologs.database is unset - a REAL run requires "
                "it (or set msa.remote_server: true)"
            )
        return _search_mock(sequence, cfg)

    if cfg.database is None:
        if remote_server:
            # msa.remote_server: real structural MSA comes from Boltz's hosted
            # server (--use_msa_server); there is no local DB to mine homologs
            # from, so the EvoLiEZ homolog stage (and its evolutionary /
            # subfamily features) falls back to SYNTHETIC homologs. Loud so the
            # degraded evolutionary signal is never mistaken for real. Set
            # homologs.database to a local UniRef/BFD DB for real homolog mining.
            log.warning(
                "homologs.database unset + msa.remote_server: using SYNTHETIC "
                "homologs (evolutionary/subfamily features are not real); "
                "Boltz structural MSA is still real via --use_msa_server. Set "
                "homologs.database for real homolog evolutionary signal."
            )
            return _search_mock(sequence, cfg)
        raise ValueError(
            "homologs.database is required for backend=real "
            "(point it at a UniRef/BFD DB on /mnt/data2), "
            "or set msa.remote_server: true"
        )
    if cfg.method == "mmseqs2":
        require("mmseqs")
        tmp = workdir / "mmseqs_tmp"
        cmd = ["mmseqs", "easy-search", str(query), cfg.database, str(out), str(tmp),
               "--max-seqs", str(cfg.max_sequences), "-e", str(cfg.evalue_max),
               "--threads", str(cfg.search_threads),
               "--format-output",
               "query,target,fident,alnlen,evalue,qstart,qaln,taln"]
        if cfg.mmseqs_gpu:                  # GPU search needs a makepaddedseqdb DB
            cmd += ["--gpu", "1"]           # (cfg.database) + a CVD-pinned free GPU
        run(cmd, dry_run=dry_run)
    elif cfg.method == "jackhmmer":
        jh = cfg.jackhmmer_bin or "jackhmmer"
        if cfg.jackhmmer_bin is None:
            require("jackhmmer")
        # HMMER's single DB-reader thread caps a lone jackhmmer at ~2 MB/s
        # regardless of --cpu; split the DB into chunks and run one jackhmmer per
        # chunk in PARALLEL to bypass it (the homolog union is what s02 needs).
        if cfg.jackhmmer_chunks > 1 and not dry_run:
            hs = _jackhmmer_parallel(sequence, jh, cfg, workdir)
            return assign_clusters(hs, cfg)
        # -A writes the hit MSA (Stockholm); --tblout alone has NO aligned
        # sequences, so the old shared parser produced invalid homologs.
        run(
            [jh, "-N", "3", "-E", str(cfg.evalue_max),
             "--cpu", str(cfg.search_threads),
             "-A", str(out), "--tblout", str(workdir / "hits.tbl"),
             str(query), cfg.database],
            env=tool_env(jh), dry_run=dry_run,
        )
    else:  # blastp
        require("blastp")
        run(
            ["blastp", "-query", str(query), "-db", cfg.database,
             "-evalue", str(cfg.evalue_max), "-num_threads",
             str(cfg.search_threads), "-max_target_seqs",
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
    # Inverted k-mer index (kmer -> centroid ids holding it). For each new
    # sequence we TALLY shared k-mers per centroid by walking the index — no
    # `km & cen` set is ever built, so the millions of throwaway intersection
    # sets (and their GC churn that pinned the old loop at ~20% CPU) are gone.
    # `counts[c]` is exactly |km ∩ centroid_c|, so the Jaccard + lowest-index
    # tie-break are IDENTICAL to the naive O(n*centroids) scan, just CPU-bound.
    centroid_len: List[int] = []
    kmer_to_cen: dict = {}
    for i in order:
        km = kmer_cache[i]
        lkm = len(km)
        counts: dict = {}
        for kk in km:
            for c in kmer_to_cen.get(kk, ()):
                counts[c] = counts.get(c, 0) + 1
        best_c, best_j = -1, 0.0
        for c in sorted(counts):
            inter = counts[c]
            j = inter / (lkm + centroid_len[c] - inter)
            if j > best_j:
                best_c, best_j = c, j
        if best_c >= 0 and best_j >= jthr:
            cluster_of[i] = best_c
        else:
            cid = len(centroids)
            cluster_of[i] = cid
            centroids.append(km)
            centroid_len.append(lkm)
            for kk in km:
                kmer_to_cen.setdefault(kk, []).append(cid)
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


def _parse_mmseqs_m8(out: Path, cfg: HomologConfig,
                     target_len: int = 0) -> List[Homolog]:
    """Real-run format: query,target,fident,alnlen,evalue,qstart,qaln,taln
    (8 cols, carries the alignment so the hit can be target-anchored for the
    integrated MSA). Legacy 6-col output (query,target,fident,alnlen,evalue,
    tseq) is still parsed (``aligned`` stays None) so old fixtures/callers keep
    working."""
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
        if len(p) >= 8:                       # new: qstart, qaln, taln
            try:
                qstart = int(p[5])
            except ValueError:
                continue
            seq = p[7].replace("-", "").replace(".", "").upper()
            aligned = (_anchor_to_target(qstart, p[6], p[7], target_len)
                       if target_len else None)
        else:                                 # legacy: tseq, no alignment
            seq = p[5].replace("-", "").replace(".", "").upper()
            aligned = None
        if seq and _band(cfg, ident):
            hs.append(Homolog(f"mm_{i}", seq, round(ident, 4), 1.0,
                              aligned=aligned))
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


def _aligned_identity(a: str, b: str) -> float:
    """Identity between two ALIGNED (gapped, equal-column) rows: matched
    non-gap columns / columns where either row has a residue. Alignment-aware,
    so an insertion or deletion no longer frame-shifts the comparison the way
    gap-stripping-then-positional-compare did."""
    n = min(len(a), len(b))
    same = cols = 0
    for i in range(n):
        ag = a[i] in "-."
        bg = b[i] in "-."
        if ag and bg:
            continue
        cols += 1
        if not ag and not bg and a[i].upper() == b[i].upper():
            same += 1
    return same / cols if cols else 0.0


def _anchor_to_target(qstart: int, qaln: str, taln: str, target_len: int) -> str:
    """Map a local pairwise alignment (query == our target) onto ONE row in
    TARGET-column coordinates (length ``target_len``): the hit residue is placed
    in every target column the alignment covers, '-' elsewhere. Insertions in
    the hit relative to the target are dropped (a3m convention), so the row
    stacks directly onto the ColabFold a3m MSA. ``qstart`` is 1-based.

    This is what lets the STRUCTURE track (Foldseek's 3Di alignment) and an
    independent LOCAL sequence track (mmseqs) be placed by their OWN native
    alignment instead of re-aligned by MAFFT, so a remote structural homolog
    lands in the correct columns rather than injecting alignment noise."""
    row = ["-"] * target_len
    qpos = qstart - 1
    for qc, tc in zip(qaln, taln):
        if qc in "-.":                       # hit insertion vs target -> drop
            continue
        if 0 <= qpos < target_len:
            row[qpos] = tc.upper() if tc.isalpha() else "-"   # junk -> gap
        qpos += 1
    return "".join(row)


def _parse_stockholm(out: Path, cfg: HomologConfig,
                     query_seq: str, target_len: int = 0) -> List[Homolog]:
    """jackhmmer -A Stockholm MSA: aggregate aligned rows per sequence,
    identity computed vs the query.

    Identity is computed COLUMN-BY-COLUMN against the aligned query row (the
    seed, present as a row in -A output). Comparing two independently
    gap-stripped sequences positionally frame-shifts everything past any indel
    - a ~95%-identical homolog with one deletion scored ~0.20, was dropped by
    the band filter, and corrupted subfamily-representative selection. When the
    query row cannot be located we fall back to the positional estimate."""
    rows: dict[str, str] = {}
    order: List[str] = []
    for line in out.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s == "//":
            continue
        parts = s.split()
        if len(parts) != 2:
            continue
        name, aln = parts
        if name not in rows:
            order.append(name)
        rows[name] = rows.get(name, "") + aln

    q = (query_seq or "").replace("-", "").replace(".", "").upper()
    query_aln = None
    for name in order:
        if q and rows[name].replace("-", "").replace(".", "").upper() == q:
            query_aln = rows[name]
            break

    hs: List[Homolog] = []
    for i, name in enumerate(order):
        aln = rows[name]
        seq = aln.replace("-", "").replace(".", "").upper()
        if not seq:
            continue
        if query_aln is not None:
            ident = _aligned_identity(aln, query_aln)
            # Target-anchor the hit against the seed (query) row, so the
            # jackhmmer track stacks onto the SAME target-column MSA as
            # mmseqs/Foldseek instead of being re-aligned by MAFFT (user §7).
            aligned = (_anchor_to_target(1, query_aln, aln, target_len)
                       if target_len else None)
        else:
            ident = _pairwise_identity(seq, q)
            aligned = None
        if _band(cfg, ident):
            hs.append(Homolog(f"jh_{i}", seq, round(ident, 4), 1.0,
                              aligned=aligned))
    return hs[: cfg.max_sequences]


def _parse_hits(out: Path, cfg: HomologConfig,
                query_seq: str = "") -> List[Homolog]:
    if cfg.method == "mmseqs2":
        hs = _parse_mmseqs_m8(out, cfg, len(query_seq))
    elif cfg.method == "jackhmmer":
        hs = _parse_stockholm(out, cfg, query_seq, len(query_seq))
    else:
        hs = _parse_blast_m8(out, cfg)
    return assign_clusters(hs, cfg)


def _split_fasta(path: Path, outdir: Path, n: int) -> List[Path]:
    """Round-robin a FASTA into ``n`` chunk files (split on '>' record
    boundaries) so N jackhmmer processes can search them in parallel."""
    handles = [(outdir / f"c{i}.fasta").open("w") for i in range(n)]
    cur, idx = 0, -1
    try:
        with path.open() as fh:
            for line in fh:
                if line.startswith(">"):
                    idx += 1
                    cur = idx % n
                handles[cur].write(line)
    finally:
        for h in handles:
            h.close()
    return [outdir / f"c{i}.fasta" for i in range(n)]


def _jackhmmer_parallel(
    sequence: str, jh: str, cfg: HomologConfig, workdir: Path
) -> List[Homolog]:
    """Split the target DB into ``cfg.jackhmmer_chunks`` chunks (staged in
    /dev/shm to dodge a fragmented HDD) and run one jackhmmer per chunk
    concurrently, then merge + dedupe the per-chunk Stockholm hits. Sidesteps
    HMMER's single DB-reader thread (the ~2 MB/s ceiling that makes a lone
    jackhmmer ignore extra --cpu) for an ~N-fold speedup on N cores. The merged
    homolog UNION is what s02 needs; per-chunk iterative profiles differ
    slightly from a whole-DB run but still surface each chunk's homologs."""
    workdir.mkdir(parents=True, exist_ok=True)
    query = workdir / "query.fasta"
    query.write_text(f">query\n{sequence}\n")
    n = max(2, cfg.jackhmmer_chunks)
    cpu = max(1, cfg.search_threads // n)
    scratch = Path("/dev/shm") if Path("/dev/shm").is_dir() else workdir
    chunkdir = scratch / f"jh_chunks_{os.getpid()}"
    chunkdir.mkdir(parents=True, exist_ok=True)
    # The bioconda HMMER is an MPI build whose dynamic-linker library search
    # thrashes a slow/HDD conda env, and OpenBLAS/OpenMP would each spawn a
    # full thread pool per process -> N-way concurrency thrashes. Pin BLAS/OMP
    # to 1 thread and disable the hwcap subdir search so the chunks actually run
    # in parallel instead of serialising on startup/thread contention.
    env = tool_env(jh) or dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["LD_HWCAP_MASK"] = "0"
    homs: List[Homolog] = []
    try:
        chunks = _split_fasta(Path(cfg.database), chunkdir, n)
        log.info("jackhmmer: %d-way parallel search (--cpu %d each) over %s",
                 n, cpu, cfg.database)
        procs = []
        for i, ch in enumerate(chunks):
            sto = workdir / f"aln_{i}.sto"
            cmd = [str(jh), "-N", "3", "-E", str(cfg.evalue_max), "--cpu",
                   str(cpu), "-A", str(sto), "--noali", str(query), str(ch)]
            procs.append((subprocess.Popen(
                cmd, env=env, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL), sto))
        for p, sto in procs:
            p.wait()
            if sto.exists() and sto.stat().st_size:
                homs += _parse_stockholm(sto, cfg, sequence, len(sequence))
    finally:
        shutil.rmtree(chunkdir, ignore_errors=True)
    best: dict = {}
    for h in homs:                              # dedupe, keep highest identity
        if h.sequence not in best or h.identity > best[h.sequence].identity:
            best[h.sequence] = h
    merged = list(best.values())[: cfg.max_sequences]
    log.info("jackhmmer (parallel) -> %d homologs across %d chunks",
             len(merged), n)
    return merged


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


def integrate_aligned_homologs(
    msa: List[Tuple[str, str]], homologs: List[Homolog], target_len: int
) -> Tuple[List[Tuple[str, str]], int]:
    """Stack homologs that carry a TARGET-anchored ``aligned`` row (Foldseek
    STRUCTURAL hits; independent local-mmseqs SEQUENCE hits) onto the base MSA,
    so the structure + sequence tracks share ONE target-column matrix (user §7).

    Each track is placed by its OWN native alignment (structural 3Di / sequence)
    rather than re-aligned, so a remote structural homolog lands in the right
    columns instead of injecting MAFFT noise. No-op unless the base MSA is in
    target-column coordinates (ColabFold a3m / mock identity, ncol == target_len);
    a de-novo MAFFT base already contains the homologs, so we skip it. Dedupe by
    ungapped sequence (the ColabFold-mined rows are already in the a3m base).

    Returns the (possibly extended) MSA and the number of rows added."""
    if not msa:
        return msa, 0
    ncol = len(msa[0][1])
    if ncol != target_len:                   # base not in target coords -> skip
        return msa, 0
    seen = {r.replace("-", "").replace(".", "").upper() for _, r in msa}
    added = 0
    for h in homologs:
        a = getattr(h, "aligned", None)
        if not a or len(a) != ncol:
            continue
        key = a.replace("-", "").replace(".", "").upper()
        if key and key not in seen:
            msa.append((h.id, a))
            seen.add(key)
            added += 1
    return msa, added
