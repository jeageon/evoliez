"""Real-format homolog parser contracts (expert review P1).

No external tools needed: feed real tool-output fixtures to the parsers and
assert the column contracts (the original shared parser read BLAST qcovs as
identity and could not get jackhmmer sequences at all).
"""

from evoliez.adapters.msa_tools import (
    _parse_blast_m8,
    _parse_mmseqs_m8,
    _parse_stockholm,
)
from evoliez.config import HomologConfig


def test_blast_uses_pident_not_qcovs(tmp_path):
    # outfmt 6: sseqid  pident  qcovs  evalue  sseq
    # pident=80% (in band), qcovs=99% (would be out of the 0.95 cap if misread)
    f = tmp_path / "hits.m8"
    f.write_text("sp|P1|X\t80.0\t99\t1e-40\tMKTAYIAKQR\n")
    hs = _parse_blast_m8(f, HomologConfig(identity_min=0.2, identity_max=0.95))
    assert len(hs) == 1
    assert abs(hs[0].identity - 0.80) < 1e-6   # pident, NOT 0.99 qcovs
    assert hs[0].sequence == "MKTAYIAKQR"
    assert abs(hs[0].coverage - 0.99) < 1e-6


def test_mmseqs_fident_and_tseq(tmp_path):
    # query,target,fident,alnlen,evalue,tseq
    f = tmp_path / "hits.m8"
    f.write_text("q\tt1\t0.55\t120\t1e-30\tACDEFGHIKL\n")
    hs = _parse_mmseqs_m8(f, HomologConfig())
    assert len(hs) == 1 and abs(hs[0].identity - 0.55) < 1e-6
    assert hs[0].sequence == "ACDEFGHIKL"


def test_stockholm_aggregates_wrapped_rows(tmp_path):
    q = "ACDEFGHIKL"
    sto = tmp_path / "aln.sto"
    sto.write_text(
        "# STOCKHOLM 1.0\n"
        "hitA  ACDEF\n"
        "hitB  ACDEG\n"
        "\n"
        "hitA  GHIKL\n"          # wrapped continuation
        "hitB  GHIKM\n"
        "//\n"
    )
    hs = _parse_stockholm(sto, HomologConfig(identity_min=0.0,
                                             identity_max=1.0), q)
    by = {h.sequence: h for h in hs}
    assert "ACDEFGHIKL" in by                  # rows aggregated
    assert abs(by["ACDEFGHIKL"].identity - 1.0) < 1e-6
    assert "ACDEGGHIKM" in by
    assert by["ACDEGGHIKM"].identity < 1.0      # 2 mismatches vs query


def test_blast_band_filter(tmp_path):
    f = tmp_path / "hits.m8"
    f.write_text(
        "a\t99.0\t100\t0\tAAAA\n"   # 0.99 > identity_max 0.95 -> dropped
        "b\t60.0\t90\t1e-9\tCCCC\n"  # kept
    )
    hs = _parse_blast_m8(f, HomologConfig(identity_min=0.2, identity_max=0.95))
    assert [h.sequence for h in hs] == ["CCCC"]


def test_stockholm_identity_is_alignment_aware_under_indel(tmp_path):
    """jackhmmer identity must be computed against the ALIGNED query, not by
    gap-stripping both then comparing positionally. A homolog identical to the
    query except ONE deletion is ~0.95 identical; the old gap-strip method
    frame-shifts everything after the indel to ~0.21 and silently DROPS the
    homolog under any realistic identity_min (and seeds the wrong subfamily
    representative)."""
    q = "ACDEFGHIKLMNPQRSTVWY"
    sto = tmp_path / "indel.sto"
    sto.write_text(
        "# STOCKHOLM 1.0\n\n"
        "query   ACDEFGHIKLMNPQRSTVWY\n"
        "homo    ACDE-GHIKLMNPQRSTVWY\n"   # single deletion at column 5
        "//\n"
    )
    hs = _parse_stockholm(sto, HomologConfig(identity_min=0.5, identity_max=1.0), q)
    homo = [h for h in hs if h.sequence == "ACDEGHIKLMNPQRSTVWY"]
    assert homo, "near-identical homolog with one indel was dropped (frame-shift bug)"
    assert homo[0].identity > 0.9
