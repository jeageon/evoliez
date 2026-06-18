"""Homolog search + MSA (spec sections 7 & 8).

real: MMseqs2 / jackhmmer / BLASTp for search, MAFFT for alignment.
mock: deterministic synthetic homologs at controlled identity, trivially
aligned (same length), so evolutionary features have realistic structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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

    per_source: List[Tuple[str, List[Homolog]]] = []
    for s in norm:
        hs = _run_source(s, sequence, cfg, workdir, backend=backend,
                         dry_run=dry_run, remote_server=remote_server,
                         msa_dir=msa_dir)
        if hs:
            per_source.append((s, hs))
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
               "--format-output",
               "query,target,fident,alnlen,evalue,qstart,qaln,taln"]
        if cfg.mmseqs_gpu:                  # GPU search needs a makepaddedseqdb DB
            cmd += ["--gpu", "1"]           # (cfg.database) + a CVD-pinned free GPU
        run(cmd, dry_run=dry_run)
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
                     query_seq: str) -> List[Homolog]:
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
        else:
            ident = _pairwise_identity(seq, q)
        if _band(cfg, ident):
            hs.append(Homolog(f"jh_{i}", seq, round(ident, 4), 1.0))
    return hs[: cfg.max_sequences]


def _parse_hits(out: Path, cfg: HomologConfig,
                query_seq: str = "") -> List[Homolog]:
    if cfg.method == "mmseqs2":
        hs = _parse_mmseqs_m8(out, cfg, len(query_seq))
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
