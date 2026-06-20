"""Local MSA generation via a batched mmseqs GPU search (server only).

For the s06b family ensemble: rather than fetch each representative's MSA from
the remote ColabFold server (~minutes/seq + external load), run ONE batched
mmseqs search over all representatives against a local UniRef30 DB and split the
result into per-sequence a3m files that Boltz consumes directly
(``--msa-format-mode 5`` -> A3M, ``unpackdb`` -> one file per query). The DB is
loaded once for the whole batch, so 150 sequences cost ~one search, not 150.

Content-keyed cache (sha1 of the sequence): a resume / re-run reuses prior a3m's
and only searches sequences whose a3m is missing.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from evoliez.logging_utils import get_logger
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.local_msa")


def seq_key(seq: str) -> str:
    """Stable content key for a sequence (case/space-insensitive)."""
    return hashlib.sha1(seq.strip().upper().encode()).hexdigest()[:16]


def default_msa_db(search_db: str) -> str:
    """The result2msa target DB (un-padded sequences). The GPU search needs a
    `makepaddedseqdb` DB (``*_pad``); result2msa must read the REGULAR seqdb so
    the emitted a3m has clean (un-padded) sequences."""
    return search_db[:-4] if search_db.endswith("_pad") else search_db


def ensure_local_msas(
    sequences: List[str],
    cache_dir: Path,
    *,
    mmseqs_bin: str,
    search_db: str,
    msa_db: Optional[str] = None,
    gpu: bool = True,
    max_seqs: int = 2000,
    evalue: float = 1e-3,
    threads: int = 8,
    dry_run: bool = False,
) -> Dict[str, Optional[Path]]:
    """Map each input sequence to its a3m path (None if it got no MSA).

    Only sequences whose a3m is not already cached are (batch) searched. The
    caller pins the GPU via CUDA_VISIBLE_DEVICES (inherited here)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    msa_db = msa_db or default_msa_db(search_db)

    uniq = list(dict.fromkeys(s.strip() for s in sequences))  # dedupe, keep order
    result: Dict[str, Optional[Path]] = {}
    missing: List[str] = []
    for s in uniq:
        a3m = cache_dir / f"{seq_key(s)}.a3m"
        if a3m.exists() and a3m.stat().st_size > 0:
            result[s] = a3m
        else:
            missing.append(s)

    if missing and not dry_run:
        log.info("local MSA: %d cached, searching %d via mmseqs (batched)",
                 len(uniq) - len(missing), len(missing))
        _run_batch(missing, cache_dir, mmseqs_bin=mmseqs_bin, search_db=search_db,
                   msa_db=msa_db, gpu=gpu, max_seqs=max_seqs, evalue=evalue,
                   threads=threads)
    for s in missing:
        a3m = cache_dir / f"{seq_key(s)}.a3m"
        result[s] = a3m if (a3m.exists() and a3m.stat().st_size > 0) else None
    if missing:
        got = sum(1 for s in missing if result[s] is not None)
        log.info("local MSA: %d/%d searched sequences got an a3m", got, len(missing))
    return result


def _run_batch(
    seqs: List[str], cache_dir: Path, *, mmseqs_bin: str, search_db: str,
    msa_db: str, gpu: bool, max_seqs: int, evalue: float, threads: int,
) -> None:
    """One mmseqs createdb -> search -> result2msa -> unpackdb over `seqs`,
    writing <cache_dir>/<seq_key>.a3m for every query that produced an MSA."""
    require(mmseqs_bin)
    work = Path(tempfile.mkdtemp(prefix="local_msa_", dir=str(cache_dir)))
    try:
        qfa = work / "q.fasta"
        # header = index; --shuffle 0 keeps DB keys == query order, so unpacked
        # i.a3m maps back to seqs[i] without depending on .lookup parsing.
        qfa.write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(seqs)))
        qdb, res, msa, tmp = work / "qdb", work / "res", work / "msa", work / "tmp"
        unpack = work / "a3m"

        run([mmseqs_bin, "createdb", str(qfa), str(qdb), "--shuffle", "0"])
        search = [mmseqs_bin, "search", str(qdb), search_db, str(res), str(tmp),
                  "-e", str(evalue), "--max-seqs", str(max_seqs),
                  "--threads", str(threads)]
        if gpu:
            search += ["--gpu", "1"]  # needs a makepaddedseqdb DB + a pinned GPU
        run(search, env=dict(os.environ))
        run([mmseqs_bin, "result2msa", str(qdb), msa_db, str(res), str(msa),
             "--msa-format-mode", "5", "--threads", str(threads)])  # 5 = A3M
        run([mmseqs_bin, "unpackdb", str(msa), str(unpack),
             "--unpack-suffix", ".a3m", "--unpack-name-mode", "0"])  # 0 = DB key

        for i, s in enumerate(seqs):
            src = unpack / f"{i}.a3m"
            if src.exists() and src.stat().st_size > 0:
                shutil.move(str(src), str(cache_dir / f"{seq_key(s)}.a3m"))
            else:
                log.warning("local MSA: query %d produced no a3m", i)
    finally:
        shutil.rmtree(work, ignore_errors=True)
