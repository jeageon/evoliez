"""Target-anchored integration of the structure + local tracks into the MSA.

s03 builds the sequence MSA from the ColabFold a3m (target-column coordinates);
the STRUCTURE track (Foldseek) and an independent LOCAL sequence track (mmseqs)
are then stacked onto it using each hit's OWN native alignment (qstart/qaln/taln
-> target columns), so a remote structural homolog lands in the right columns
instead of being re-aligned by MAFFT (user §7). All deterministic, no tools.
"""

from pathlib import Path

from evoliez.adapters.foldseek import _parse_m8, search_structural_homologs
from evoliez.adapters.msa_tools import (
    Homolog,
    _anchor_to_target,
    _parse_mmseqs_m8,
    integrate_aligned_homologs,
)
from evoliez.config import Backend, HomologConfig
from evoliez.features.evolutionary import compute_position_features


# ----------------------------- _anchor_to_target ----------------------------
def test_anchor_basic_full_span():
    # query/target cols 1..5, one mismatch (target C -> hit G); rest gap.
    row = _anchor_to_target(1, "ACDEF", "AGDEF", 10)
    assert row == "AGDEF-----"


def test_anchor_qstart_offset():
    # alignment starts at target residue 4 -> first 3 columns are gaps.
    row = _anchor_to_target(4, "EFG", "EXG", 10)
    assert row == "---EXG----"


def test_anchor_hit_insertion_is_dropped():
    # qaln gap == residue inserted in the hit relative to the target -> dropped,
    # and the target cursor does NOT advance (a3m convention).
    row = _anchor_to_target(1, "AC-EF", "ACGEF", 10)
    assert row == "ACEF------"          # inserted G gone; A,C,E,F in cols 1-4


def test_anchor_hit_deletion_becomes_gap():
    # hit is missing the residue at target col 3 -> that column is a gap.
    row = _anchor_to_target(1, "ACDEF", "AC-EF", 10)
    assert row == "AC-EF-----"


def test_anchor_lowercase_uppercased():
    assert _anchor_to_target(1, "ACD", "acd", 5) == "ACD--"


# ------------------------- integrate_aligned_homologs -----------------------
def _base():
    # base MSA in target-column coords: query (no gaps) + one ColabFold row.
    return [("query", "ACDEFGHIKL"), ("cf1", "ACDEFGH-KL")]   # cf1 ungap=ACDEFGHKL


def test_integrate_appends_structure_row():
    msa = _base()
    homs = [Homolog("fs0", "AGDEF", 0.2, 1.0, source="structure",
                    aligned="AGDEF-----")]
    out, n = integrate_aligned_homologs(msa, homs, 10)
    assert n == 1
    assert ("fs0", "AGDEF-----") in out
    assert len(out) == 3


def test_integrate_dedupes_against_base():
    # a homolog whose ungapped seq already appears in the base is NOT re-added
    # (the ColabFold-mined rows are already in the a3m).
    msa = _base()
    homs = [Homolog("dup", "ACDEFGHKL", 0.9, 1.0, aligned="ACDEFGH-KL")]
    _, n = integrate_aligned_homologs(msa, homs, 10)
    assert n == 0


def test_integrate_dedupes_among_homologs():
    msa = _base()
    homs = [Homolog("a", "AGDEF", 0.2, 1.0, aligned="AGDEF-----"),
            Homolog("b", "AGDEF", 0.2, 1.0, aligned="AGDEF-----")]
    _, n = integrate_aligned_homologs(msa, homs, 10)
    assert n == 1


def test_integrate_skips_wrong_length_and_none():
    msa = _base()
    homs = [Homolog("short", "AG", 0.2, 1.0, aligned="AG--"),     # len 4 != 10
            Homolog("noaln", "AGDEF", 0.2, 1.0, aligned=None)]
    _, n = integrate_aligned_homologs(msa, homs, 10)
    assert n == 0


def test_integrate_noop_when_base_not_target_coords():
    # de-novo MAFFT base has more columns than the target -> skip (homologs are
    # already aligned into it upstream).
    msa = [("query", "AC--DEFGHIKL"), ("h", "ACGGDEFGHIKL")]     # ncol 12
    homs = [Homolog("fs0", "AGDEF", 0.2, 1.0, aligned="AGDEF" + "-" * 5)]
    _, n = integrate_aligned_homologs(msa, homs, 10)              # target_len 10
    assert n == 0


def test_integrate_empty_msa():
    out, n = integrate_aligned_homologs([], [Homolog("x", "AA", 0.5, 1.0)], 10)
    assert out == [] and n == 0


# -------------------- parsers emit target-anchored `aligned` ----------------
def test_foldseek_parse_new_format_anchors(tmp_path):
    f = tmp_path / "foldseek_hits.m8"
    # query,target,fident,alnlen,evalue,qstart,qaln,taln  (remote structural hit)
    f.write_text("query\tt1\t0.18\t3\t1e-5\t3\tEFG\tEYG\n")
    hs = _parse_m8(f, HomologConfig(), target_len=10)
    assert len(hs) == 1
    h = hs[0]
    assert h.source == "structure" and h.annotation == "foldseek"
    assert h.sequence == "EYG"                 # taln, ungapped
    assert h.aligned == "--EYG-----"           # anchored at target col 3


def test_foldseek_parse_legacy_format_no_alignment(tmp_path):
    f = tmp_path / "foldseek_hits.m8"           # old 6-col (…,tseq)
    f.write_text("query\tt1\t0.30\t8\t1e-9\tACDEFGHI\n")
    hs = _parse_m8(f, HomologConfig(), target_len=10)
    assert len(hs) == 1 and hs[0].sequence == "ACDEFGHI"
    assert hs[0].aligned is None


def test_mmseqs_parse_new_format_anchors(tmp_path):
    f = tmp_path / "hits.m8"
    f.write_text("q\tt1\t0.55\t5\t1e-30\t1\tACDEF\tAGDEF\n")
    hs = _parse_mmseqs_m8(f, HomologConfig(identity_min=0.2, identity_max=0.95),
                          target_len=10)
    assert len(hs) == 1
    assert hs[0].sequence == "AGDEF"
    assert hs[0].aligned == "AGDEF-----"


def test_mmseqs_parse_legacy_still_works(tmp_path):
    # the existing 6-col contract (no target_len) -> aligned stays None.
    f = tmp_path / "hits.m8"
    f.write_text("q\tt1\t0.55\t120\t1e-30\tACDEFGHIKL\n")
    hs = _parse_mmseqs_m8(f, HomologConfig())
    assert len(hs) == 1 and hs[0].sequence == "ACDEFGHIKL"
    assert hs[0].aligned is None


# ----------------- the integrated MSA changes the conservation --------------
def test_structure_track_reaches_conservation(tmp_path):
    """The whole point: a structural homolog placed into the target columns
    actually participates in the per-position conservation (not dropped)."""
    target = "ACDEFGHIKL"
    msa = [("query", target), ("cf1", target)]          # identical -> col 1 == {A}
    # remote structural homolog: different residues in cols 1-5
    fs = tmp_path / "foldseek_hits.m8"
    fs.write_text("query\tt1\t0.20\t5\t1e-6\t1\tACDEF\tWWWWW\n")
    homs = _parse_m8(fs, HomologConfig(), target_len=len(target))
    msa, n = integrate_aligned_homologs(msa, homs, len(target))
    assert n == 1
    feats = compute_position_features(msa)
    assert len(feats) == len(target)
    c0 = feats[0]
    assert c0.target_position == 1                       # query-anchored map intact
    # col 1 now sees A,A,W -> both residues present (structure track counted)
    assert "A" in c0.amino_acid_frequencies and "W" in c0.amino_acid_frequencies


def test_search_real_passes_target_len_to_parser(tmp_path, monkeypatch):
    """Regression: BOTH foldseek return paths must pass target_len to the parser.
    The FRESH path (after running the search) once dropped it, so real-run hits
    came back with aligned=None and the structure track silently vanished from
    the integrated MSA (the cached-reuse path was fine, hiding the bug)."""
    import evoliez.adapters.foldseek as fsmod
    seq = "ACDEFGHIKLMNPQRSTVWY" * 5             # 100 aa target
    cfg = HomologConfig(foldseek_database="db", foldseek_prostt5="p5")
    monkeypatch.setattr(fsmod, "require", lambda *a, **k: None)

    def fake_run(cmd, **k):                       # emit an 8-col (qstart,qaln,taln) m8
        out = next(str(c) for c in cmd if str(c).endswith("foldseek_hits.m8"))
        Path(out).write_text("query\tt1\t0.2\t5\t1e-6\t1\tACDEF\tWYDEF\n")
    monkeypatch.setattr(fsmod, "run", fake_run)

    hs = search_structural_homologs(seq, cfg, tmp_path, backend=Backend.real)
    assert hs, "fresh foldseek path returned no homologs"
    assert hs[0].aligned is not None, "aligned None -> structure track vanishes from MSA"
    assert len(hs[0].aligned) == len(seq)         # anchored to TARGET length


def test_read_a3m_drops_corrupt_and_junk(tmp_path):
    """A malformed ColabFold a3m (a lone NUL byte as a 'sequence') must not reach
    the MSA -- else Boltz's a3m parser dies with KeyError '\\x00' and silently
    skips the structure prediction. Sanitize on read: keep aligned columns
    (uppercase + gap), drop lowercase insertions + junk, drop emptied entries."""
    from evoliez.adapters.remote_msa import _read_a3m
    p = tmp_path / "x.a3m"
    p.write_bytes(b">query\nACDEFG\n>good\nACDE-G\n>corrupt\n\x00\n>ins\nACdefGH\n")
    msa = dict(_read_a3m(p))
    assert "corrupt" not in msa                   # NUL-only entry dropped
    assert msa["query"] == "ACDEFG"
    assert msa["good"] == "ACDE-G"
    assert msa["ins"] == "ACGH"                   # lowercase insertions dropped
    assert all("\x00" not in s for s in msa.values())
