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
    _db_for,
    _parse_mmseqs_m8,
    _parse_stockholm,
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


def test_mmseqs_gpu_flag(monkeypatch, tmp_path):
    """homologs.mmseqs_gpu -> the local mmseqs easy-search runs with --gpu (against
    a makepaddedseqdb-padded database) for fast GPU search with no 16h CPU index."""
    import evoliez.adapters.msa_tools as mt
    cap = {}
    monkeypatch.setattr(mt, "require", lambda *a, **k: None)
    monkeypatch.setattr(mt, "run", lambda cmd, **k: cap.__setitem__("cmd", cmd))
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    mt._search_real(seq, HomologConfig(method="mmseqs2", database="pad_db",
                                       mmseqs_gpu=True), tmp_path, dry_run=False)
    assert "--gpu" in cap["cmd"], cap["cmd"]
    cap.clear()
    mt._search_real(seq, HomologConfig(method="mmseqs2", database="db"),
                    tmp_path, dry_run=False)
    assert "--gpu" not in cap["cmd"]               # off by default


# ===================== 5-track multi-tool homology =========================
def test_jackhmmer_stockholm_now_target_anchors(tmp_path):
    """jackhmmer's -A Stockholm hits must carry a target-anchored `aligned`
    row (against the seed/query row), or the jackhmmer track silently never
    reaches the integrated MSA the way mmseqs/Foldseek do (user §7)."""
    sto = tmp_path / "aln.sto"
    # seed (query) + one hit with a deletion at target col 3.
    sto.write_text("# STOCKHOLM 1.0\n\nquery  ACDEF\nhit1   AG-EF\n//\n")
    hs = _parse_stockholm(sto, HomologConfig(identity_min=0.2, identity_max=0.95),
                          "ACDEF", target_len=5)
    assert len(hs) == 1                            # query (ident 1.0) band-filtered
    assert hs[0].sequence == "AGEF"                # ungapped hit
    assert hs[0].aligned == "AG-EF"                # anchored to the 5 target cols
    out, n = integrate_aligned_homologs([("query", "ACDEF")], hs, 5)
    assert n == 1                                  # reaches the integrated MSA
    assert any(s == "AG-EF" for _, s in out)       # the anchored hit row is stacked


def test_jackhmmer_legacy_no_target_len_keeps_aligned_none(tmp_path):
    sto = tmp_path / "aln.sto"
    sto.write_text("# STOCKHOLM 1.0\n\nquery  ACDEF\nhit1   AG-EF\n//\n")
    hs = _parse_stockholm(sto, HomologConfig(), "ACDEF")   # no target_len
    assert hs and hs[0].aligned is None


def test_hhblits_parse_a3m_target_anchors(tmp_path):
    """HHblits -oa3m is already in query-column coords: match states uppercase,
    insertions lowercase (dropped). Each hit row IS its target-column alignment."""
    from evoliez.adapters.hhblits import _parse_a3m

    a3m = tmp_path / "hits.a3m"
    # query + h1 (mismatch) + h2 (lowercase 'g' = insertion vs query -> dropped)
    a3m.write_text(">query\nACDEFG\n>h1\nAGDEFG\n>h2\nACgDEFW\n")
    hs = _parse_a3m(a3m, HomologConfig(identity_min=0.2, identity_max=0.95),
                    "ACDEFG")
    assert {h.sequence for h in hs} == {"AGDEFG", "ACDEFW"}
    assert all(len(h.aligned) == 6 for h in hs)            # anchored to target len
    assert all(h.source == "sequence" and h.annotation == "hhblits" for h in hs)
    h2 = next(h for h in hs if h.sequence == "ACDEFW")
    assert h2.aligned == "ACDEFW"                          # insertion 'g' gone


def test_hhblits_parse_a3m_skips_query_only(tmp_path):
    from evoliez.adapters.hhblits import _parse_a3m
    a3m = tmp_path / "q.a3m"
    a3m.write_text(">query\nACDEFG\n")
    assert _parse_a3m(a3m, HomologConfig(), "ACDEFG") == []


def test_db_for_per_tool_then_fallback():
    cfg = HomologConfig(database="shared",
                        databases={"mmseqs2": "padDB", "hhblits": "hhDB"})
    assert _db_for(cfg, "mmseqs2") == "padDB"
    assert _db_for(cfg, "hhblits") == "hhDB"
    assert _db_for(cfg, "jackhmmer") == "shared"           # falls back to `database`


def test_run_source_routes_each_tool_to_its_own_db(monkeypatch, tmp_path):
    """Each concrete tool source (mmseqs2/jackhmmer/blastp) runs an INDEPENDENT
    search against databases[tool] via a per-tool config copy."""
    import evoliez.adapters.msa_tools as mt

    captured: dict[str, str] = {}

    def fake_search(seq, cfg, workdir, **k):
        captured[cfg.method] = cfg.database
        return [Homolog("x", "AAAAAA", 0.5, 1.0)]

    monkeypatch.setattr(mt, "search_homologs", fake_search)
    cfg = HomologConfig(databases={"mmseqs2": "padDB", "jackhmmer": "fastaDB"})
    for tool in ("mmseqs2", "jackhmmer"):
        hs = mt._run_source(tool, "ACDE", cfg, tmp_path, backend=Backend.real,
                            dry_run=False, remote_server=False, msa_dir=None)
        assert hs and hs[0].source == "sequence"
    assert captured["mmseqs2"] == "padDB"
    assert captured["jackhmmer"] == "fastaDB"


def test_run_source_skips_tool_without_db(monkeypatch, tmp_path):
    import evoliez.adapters.msa_tools as mt
    monkeypatch.setattr(mt, "search_homologs",
                        lambda *a, **k: [Homolog("x", "AA", 0.5, 1.0)])
    cfg = HomologConfig(databases={"mmseqs2": "padDB"})    # jackhmmer has no DB
    out = mt._run_source("jackhmmer", "ACDE", cfg, tmp_path, backend=Backend.real,
                         dry_run=False, remote_server=False, msa_dir=None)
    assert out == []                                       # no DB -> skipped


def test_homolog_cache_roundtrip(tmp_path):
    from evoliez.adapters.msa_tools import _load_homologs, _save_homologs

    homs = [Homolog("a", "ACDE", 0.5, 1.0, annotation="mmseqs", cluster_id=2,
                    source="sequence", aligned="ACDE------"),
            Homolog("b", "WYWY", 0.3, 0.9)]
    p = tmp_path / "c.json"
    _save_homologs(homs, p)
    back = _load_homologs(p)
    assert len(back) == 2
    assert back[0].aligned == "ACDE------" and back[0].cluster_id == 2
    assert back[1].sequence == "WYWY"


def test_gather_homologs_reuses_cached_track(monkeypatch, tmp_path):
    """A computed track is cached; a second gather REUSES it without re-running
    the search — finished tracks stay put while one is retried (user §7)."""
    import evoliez.adapters.msa_tools as mt

    calls = {"n": 0}

    def fake_run_source(name, seq, cfg, wd, **k):
        calls["n"] += 1
        return [Homolog(f"{name}0", "ACDEFGHIKL", 0.5, 1.0, source="sequence")]

    monkeypatch.setattr(mt, "_run_source", fake_run_source)
    cfg = HomologConfig(sources=["mmseqs2"], databases={"mmseqs2": "db"})
    h1 = mt.gather_homologs("ACDEFGHIKL" * 4, cfg, tmp_path, backend=Backend.real)
    assert calls["n"] == 1 and h1
    h2 = mt.gather_homologs("ACDEFGHIKL" * 4, cfg, tmp_path, backend=Backend.real)
    assert calls["n"] == 1                       # reused cache, did NOT re-run
    assert [x.sequence for x in h2] == [x.sequence for x in h1]


def test_persistent_cache_survives_unrelated_config_change(monkeypatch, tmp_path):
    """cache_dir -> content-keyed cache OUTSIDE the run dir: a finished track is
    reused across runs (and run-dir purges) when an UNRELATED field changes, but
    recomputed when ITS OWN input (DB) changes (user §7)."""
    import evoliez.adapters.msa_tools as mt

    cachedir = tmp_path / "persist"
    calls = {"n": 0}

    def fake_run_source(name, seq, cfg, wd, **k):
        calls["n"] += 1
        return [Homolog(f"{name}0", "ACDEFGHIKL", 0.5, 1.0, source="sequence")]

    monkeypatch.setattr(mt, "_run_source", fake_run_source)
    seq = "ACDEFGHIKL" * 4
    base = dict(sources=["mmseqs2"], databases={"mmseqs2": "db"},
                cache_dir=str(cachedir))
    mt.gather_homologs(seq, HomologConfig(**base), tmp_path / "runA",
                       backend=Backend.real)
    assert calls["n"] == 1                       # first compute
    # different run dir (old one "purged") + an UNRELATED field changed
    mt.gather_homologs(seq, HomologConfig(jackhmmer_chunks=8, **base),
                       tmp_path / "runB", backend=Backend.real)
    assert calls["n"] == 1                       # reused — survives the change
    # this track's OWN DB changes -> recompute
    mt.gather_homologs(seq, HomologConfig(sources=["mmseqs2"],
                                          databases={"mmseqs2": "db2"},
                                          cache_dir=str(cachedir)),
                       tmp_path / "runC", backend=Backend.real)
    assert calls["n"] == 2                       # different DB -> new key -> rerun


def test_assign_clusters_matches_bruteforce():
    """The inverted-index speedup must yield the IDENTICAL clustering as the
    naive O(n*centroids) scan, on deterministic synthetic homologs."""
    import copy

    from evoliez.adapters.msa_tools import _kmers, assign_clusters

    def _bruteforce(homs, cfg):
        k = 4
        p = min(0.99, max(0.05, cfg.cluster_identity))
        jthr = (p ** k) / (2.0 - p ** k)
        order = sorted(range(len(homs)), key=lambda i: homs[i].identity,
                       reverse=True)
        kc = {i: _kmers(homs[i].sequence, k) for i in order}
        cents, cof = [], {}
        for i in order:
            km = kc[i]
            bc, bj = -1, 0.0
            for c, cen in enumerate(cents):
                inter = len(km & cen)
                if not inter:
                    continue
                j = inter / (len(km) + len(cen) - inter)
                if j > bj:
                    bc, bj = c, j
            if bc >= 0 and bj >= jthr:
                cof[i] = bc
            else:
                cof[i] = len(cents)
                cents.append(km)
        return [cof[i] for i in range(len(homs))]

    AA = "ACDEFGHIKLMNPQRSTVWY"
    base = AA * 3
    homs = []
    for k in range(40):
        seq = list(base)
        s = (k * 2654435761) & 0xFFFFFFFF
        for _ in range(k % 12):
            s = (s * 1103515245 + 12345) & 0x7FFFFFFF
            seq[s % len(seq)] = AA[(s >> 8) % 20]
        homs.append(Homolog(f"h{k}", "".join(seq), round(1 - (k % 12) / 60, 3), 1.0))
    cfg = HomologConfig(cluster_identity=0.7)
    ref = _bruteforce(copy.deepcopy(homs), cfg)
    assign_clusters(homs, cfg)
    assert [h.cluster_id for h in homs] == ref


def test_split_fasta_round_robin(tmp_path):
    """jackhmmer parallel-split: records distribute round-robin across N chunks,
    every record preserved, sequences intact."""
    from evoliez.adapters.msa_tools import _split_fasta

    fa = tmp_path / "db.fasta"
    fa.write_text(">a\nAAAA\n>b\nCCCC\n>c\nDDDD\n>d\nEEEE\n>e\nFFFF\n")
    out = tmp_path / "chunks"
    out.mkdir()
    chunks = _split_fasta(fa, out, 2)
    assert len(chunks) == 2
    ids0 = [l[1:].strip() for l in chunks[0].read_text().splitlines()
            if l.startswith(">")]
    ids1 = [l[1:].strip() for l in chunks[1].read_text().splitlines()
            if l.startswith(">")]
    assert ids0 == ["a", "c", "e"] and ids1 == ["b", "d"]   # round-robin
    both = chunks[0].read_text() + chunks[1].read_text()
    assert "AAAA" in both and "FFFF" in both                # sequences intact
    assert sorted(ids0 + ids1) == ["a", "b", "c", "d", "e"]  # nothing lost


def test_jackhmmer_chunks_routes_to_parallel(monkeypatch, tmp_path):
    """homologs.jackhmmer_chunks>1 -> _search_real dispatches to the parallel
    split path (bypassing HMMER's single-thread DB-reader bottleneck)."""
    import evoliez.adapters.msa_tools as mt

    seen = {}

    def fake_parallel(seq, jh, cfg, wd):
        seen["chunks"] = cfg.jackhmmer_chunks
        return [Homolog("p", "AAAACCCCDD", 0.5, 1.0)]

    monkeypatch.setattr(mt, "require", lambda *a, **k: None)
    monkeypatch.setattr(mt, "_jackhmmer_parallel", fake_parallel)
    hs = mt._search_real("ACDEFGHIKL" * 4,
                         HomologConfig(method="jackhmmer", database="db.fa",
                                       jackhmmer_chunks=8),
                         tmp_path, dry_run=False)
    assert seen.get("chunks") == 8
    assert hs and hs[0].id == "p"                # result flows through (clustered)


def test_jackhmmer_chunks_one_is_single_process(monkeypatch, tmp_path):
    """Default jackhmmer_chunks=1 keeps the original single-jackhmmer path."""
    import evoliez.adapters.msa_tools as mt

    cap = {"parallel": False}
    monkeypatch.setattr(mt, "require", lambda *a, **k: None)
    monkeypatch.setattr(mt, "run", lambda cmd, **k: cap.__setitem__("cmd", cmd))
    monkeypatch.setattr(mt, "_jackhmmer_parallel",
                        lambda *a, **k: cap.__setitem__("parallel", True) or [])
    # no aln.sto produced -> falls back to mock, but the dispatch is what matters
    mt._search_real("ACDEFGHIKL" * 4,
                    HomologConfig(method="jackhmmer", database="db.fa"),
                    tmp_path, dry_run=False)
    assert cap["parallel"] is False
    assert "-A" in [str(c) for c in cap["cmd"]]


def test_tool_env_prepends_conda_env_lib(tmp_path):
    """jackhmmer/hhblits living in another conda env need that env's lib on
    LD_LIBRARY_PATH (their libopenblas etc.), or they die with 'error while
    loading shared libraries' when called from the pipeline's env."""
    from evoliez.utils.subprocess_utils import tool_env

    (tmp_path / "bin").mkdir()
    (tmp_path / "lib").mkdir()
    env = tool_env(str(tmp_path / "bin" / "jackhmmer"))
    assert env is not None
    assert env["LD_LIBRARY_PATH"].split(":")[0] == str(tmp_path / "lib")
    assert tool_env("jackhmmer") is None                   # bare name -> on PATH
    nolib = tmp_path / "other" / "bin"
    nolib.mkdir(parents=True)
    assert tool_env(str(nolib / "tool")) is None           # no sibling lib dir
