"""HHblits (HH-suite) homolog retriever — the most sensitive remote-homolog
track (iterative profile–profile HMM, the twilight zone of <20% identity).

HHblits ``-oa3m`` output is ALREADY in query-column coordinates (match states
uppercase, insertions lowercase), so — exactly like the ColabFold a3m — each hit
row is target-anchored with NO re-alignment (user §7). The shared a3m reader
(``remote_msa._read_a3m``) keeps the match columns and drops lowercase
insertions, so a hit row IS its target-column alignment, ready to stack onto the
same MSA as the mmseqs / Foldseek tracks via s03's ``integrate_aligned_homologs``.

The binary lives in a dedicated conda env (off the pipeline PATH); point
``homologs.hhblits_bin`` at it. Independent track: returns [] (never raises) when
HHblits or its DB is missing, so the other four tracks still run.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from evoliez.adapters.msa_tools import (
    Homolog,
    _aligned_identity,
    _band,
    _db_for,
    _search_mock,
    assign_clusters,
)
from evoliez.config import Backend, HomologConfig
from evoliez.logging_utils import get_logger
from evoliez.utils.subprocess_utils import run, tool_env, which

log = get_logger("evoliez.hhblits")


def search_hhblits_homologs(
    sequence: str,
    cfg: HomologConfig,
    workdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
) -> List[Homolog]:
    """Independent HHblits sequence track -> target-anchored Homolog list."""
    if backend is not Backend.real:
        hs = _search_mock(sequence, cfg)
        for h in hs:
            h.source = "sequence"
            if not h.annotation:
                h.annotation = "hhblits"
        return hs

    db = _db_for(cfg, "hhblits")
    binexe = cfg.hhblits_bin or "hhblits"
    workdir.mkdir(parents=True, exist_ok=True)
    query = workdir / "query.fasta"
    query.write_text(f">query\n{sequence}\n")
    out = workdir / "hits.a3m"
    cmd = [binexe, "-i", str(query), "-d", db or "<HHBLITS_DB: set "
           "homologs.databases.hhblits>", "-oa3m", str(out),
           "-n", str(cfg.hhblits_iterations), "-e", str(cfg.evalue_max),
           "-cpu", str(cfg.search_threads), "-v", "1"]
    if dry_run:
        run(cmd, dry_run=True)
        return []
    if db is None:
        log.warning(
            "hhblits source: no database (set homologs.databases.hhblits or "
            "homologs.database); skipping the HHblits track"
        )
        return []
    if cfg.hhblits_bin is None and which("hhblits") is None:
        log.warning(
            "hhblits not on PATH (set homologs.hhblits_bin to the binary in its "
            "conda env); skipping the HHblits track"
        )
        return []
    run(cmd, env=tool_env(binexe), dry_run=dry_run)
    if not out.exists():
        log.warning("hhblits produced no a3m (%s); skipping the HHblits track", out)
        return []
    return _parse_a3m(out, cfg, sequence)


def _parse_a3m(out: Path, cfg: HomologConfig, target_seq: str) -> List[Homolog]:
    """HHblits -oa3m -> target-anchored Homologs. The shared a3m reader keeps
    match columns (uppercase + gap; lowercase insertions dropped), so every row
    is already in TARGET-column coordinates; row 0 is the query."""
    from evoliez.adapters.remote_msa import _read_a3m

    rows = _read_a3m(out)
    if len(rows) < 2:
        return []
    tlen = len(target_seq)
    query_aln = rows[0][1]
    hs: List[Homolog] = []
    for i, (_cid, row) in enumerate(rows[1:]):
        if len(row) != tlen:                 # not in target-column coords -> skip
            continue
        seq = row.replace("-", "").upper()
        if not seq:
            continue
        ident = _aligned_identity(row, query_aln)
        if _band(cfg, ident):
            hs.append(Homolog(f"hh_{i}", seq, round(ident, 4), 1.0,
                              source="sequence", annotation="hhblits",
                              aligned=row))
    hs = hs[: cfg.max_sequences]
    log.info("HHblits -> %d sequence homologs (target-anchored)", len(hs))
    return assign_clusters(hs, cfg)
