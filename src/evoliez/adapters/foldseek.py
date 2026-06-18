"""Foldseek structural homolog search (roadmap P1.1).

Retrieves remote STRUCTURAL homologs the way sequence search cannot: Foldseek's
3Di alphabet finds proteins with similar folds even at low sequence identity.
Modern Foldseek predicts the 3Di sequence from the query sequence with ProstT5,
so we need only the target SEQUENCE here (no structure yet at s02).

real: ``foldseek easy-search`` against a 3Di structure DB (AFDB50 / PDB100).
mock: deterministic synthetic structural homologs in a LOWER identity band than
sequence search (structural homologs are typically more remote), so the merged
homolog set in s02 has a distinct structure-sourced contribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from evoliez.adapters.msa_tools import Homolog, _band
from evoliez.config import Backend, HomologConfig
from evoliez.logging_utils import get_logger
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.foldseek")
_AA = "ACDEFGHIKLMNPQRSTVWY"


def search_structural_homologs(
    sequence: str,
    cfg: HomologConfig,
    workdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
) -> List[Homolog]:
    if backend is Backend.real and not dry_run:
        return _search_real(sequence, cfg, workdir)
    if dry_run and backend is Backend.real:
        _search_real(sequence, cfg, workdir, preview=True)
    return _search_mock(sequence, cfg)


def _search_real(
    sequence: str, cfg: HomologConfig, workdir: Path, *, preview: bool = False
) -> List[Homolog]:
    if not cfg.foldseek_database:
        log.warning(
            "homologs.sources includes 'structure' but homologs.foldseek_database "
            "is unset; skipping Foldseek (no structural homologs). Point it at a "
            "3Di DB (AFDB50/PDB100) for structural homolog mining."
        )
        return []
    workdir.mkdir(parents=True, exist_ok=True)
    query = workdir / "query.fasta"
    query.write_text(f">query\n{sequence}\n")
    out = workdir / "foldseek_hits.m8"
    tmp = workdir / "foldseek_tmp"
    # Reuse a prior search (ProstT5 + DB search is minutes). A fingerprint
    # change purges homologs/, so this only reuses within the SAME inputs.
    if not preview and out.exists() and out.stat().st_size > 0:
        log.info("reusing cached Foldseek hits (%s)", out)
        return _parse_m8(out, cfg, len(sequence))
    # easy-search on a FASTA query predicts 3Di via ProstT5 (Foldseek >= 8) and
    # searches the 3Di structure DB. format mirrors the mmseqs m8 parser.
    cmd = [
        "foldseek", "easy-search", str(query), cfg.foldseek_database,
        str(out), str(tmp), "--format-output",
        # qstart,qaln,taln carry Foldseek's STRUCTURAL alignment so each hit can
        # be anchored into the target's columns for the integrated s03 MSA.
        "query,target,fident,alnlen,evalue,qstart,qaln,taln",
        "--max-seqs", str(cfg.foldseek_max_seqs),
    ]
    # A FASTA (amino-acid) query needs ProstT5 to predict its 3Di before
    # searching the structure DB. Without the weights Foldseek treats the input
    # as structures and fails -> warn so the degraded result isn't a surprise.
    if cfg.foldseek_prostt5:
        cmd += ["--prostt5-model", cfg.foldseek_prostt5]
    elif not preview:
        log.warning(
            "Foldseek from a sequence query needs ProstT5 weights; set "
            "homologs.foldseek_prostt5 (`foldseek databases ProstT5 <dir> tmp`). "
            "Without it easy-search on a FASTA query will fail."
        )
    if preview:
        require("foldseek")
        run(cmd, dry_run=True)
        return []
    require("foldseek")
    run(cmd, dry_run=False, timeout=None)
    if not out.exists():
        log.warning("Foldseek produced no output (%s); no structural homologs", out)
        return []
    return _parse_m8(out, cfg, len(sequence))


def _parse_m8(out: Path, cfg: HomologConfig, target_len: int = 0) -> List[Homolog]:
    """Real-run format: query,target,fident,alnlen,evalue,qstart,qaln,taln
    (8 cols; carries Foldseek's STRUCTURAL alignment so each hit is anchored into
    the target's columns for the integrated MSA). Legacy 6-col output (…,tseq) is
    still parsed (``aligned`` stays None)."""
    from evoliez.adapters.msa_tools import _anchor_to_target

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
        # structural hits can be very remote; keep them down to identity_min but
        # not above identity_max (those are better served by sequence search).
        if seq and ident <= cfg.identity_max:
            hs.append(Homolog(f"fs_{i}", seq, round(max(ident, 0.0), 4), 1.0,
                              annotation="foldseek", source="structure",
                              aligned=aligned))
    return hs[: cfg.foldseek_max_seqs]


def _search_mock(sequence: str, cfg: HomologConfig) -> List[Homolog]:
    """Deterministic synthetic STRUCTURAL homologs in a remote identity band
    (~identity_min .. 0.55), distinct from the sequence-search mock so the s02
    merge has a real structure-sourced contribution and >= 1 extra subfamily."""
    L = len(sequence)
    n = min(40, max(12, cfg.foldseek_max_seqs // 80 or 16))
    hi = min(0.55, cfg.identity_max)
    lo = cfg.identity_min
    homologs: List[Homolog] = []
    for k in range(n):
        frac = k / max(1, n - 1)
        ident = hi - frac * (hi - lo)
        seq = list(sequence)
        nmut = int(round((1.0 - ident) * L))
        s = derive_seed(0xF01D, str(k))
        for _ in range(nmut):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            pos = s % L
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            seq[pos] = _AA[s % len(_AA)]
        if not _band(cfg, ident):
            ident = max(lo, min(cfg.identity_max, ident))
        homologs.append(
            Homolog(
                id=f"mock_struct_{k:03d}",
                sequence="".join(seq),
                identity=round(ident, 3),
                coverage=1.0,
                annotation="synthetic_structure",
                cluster_id=k % 3,
                source="structure",
            )
        )
    return homologs
