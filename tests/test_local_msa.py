"""Local-MSA path for the s06b family ensemble (rep_msa = local | single | server).

mmseqs is server-only, so these test the Python wiring — content-keyed caching,
dedupe, order-preserving map, and the s06b mode selection — with the batch step
stubbed. The real mmseqs recipe is validated on the server.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evoliez.adapters import local_msa
from evoliez.config import (
    ComplexPredictionConfig,
    HomologConfig,
    InteractionModelConfig,
)
from evoliez.stages.s06b_interaction_model import _prepare_rep_msas


# --- local_msa helper ------------------------------------------------------ #
def test_default_msa_db_strips_pad():
    assert local_msa.default_msa_db("/x/uniref30_seqdb_pad") == "/x/uniref30_seqdb"
    assert local_msa.default_msa_db("/x/uniref30_seqdb") == "/x/uniref30_seqdb"


def test_ensure_local_msas_dedupes_caches_and_maps(tmp_path, monkeypatch):
    batches = []

    def fake_batch(seqs, cache_dir, **kw):
        batches.append(list(seqs))
        for s in seqs:
            (cache_dir / f"{local_msa.seq_key(s)}.a3m").write_text(f">q\n{s}\n")

    monkeypatch.setattr(local_msa, "_run_batch", fake_batch)
    seqs = ["AAAA", "BBBB", "AAAA"]  # duplicate AAAA

    res = local_msa.ensure_local_msas(
        seqs, tmp_path, mmseqs_bin="mmseqs", search_db="/db/uniref_pad"
    )
    assert batches == [["AAAA", "BBBB"]]          # deduped -> one batch
    assert res["AAAA"].exists() and res["BBBB"].exists()

    # all cached now -> no second batch
    res2 = local_msa.ensure_local_msas(
        seqs, tmp_path, mmseqs_bin="mmseqs", search_db="/db/uniref_pad"
    )
    assert batches == [["AAAA", "BBBB"]]
    assert res2["AAAA"] == res["AAAA"]


def test_ensure_local_msas_missing_a3m_is_none(tmp_path, monkeypatch):
    # a query that produces no a3m (no hits) maps to None, not a crash.
    monkeypatch.setattr(local_msa, "_run_batch", lambda seqs, cache_dir, **kw: None)
    res = local_msa.ensure_local_msas(
        ["ZZZZ"], tmp_path, mmseqs_bin="mmseqs", search_db="/db/uniref_pad"
    )
    assert res["ZZZZ"] is None


def test_ensure_local_msas_dry_run_skips_search(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(local_msa, "_run_batch", lambda *a, **k: called.append(1))
    res = local_msa.ensure_local_msas(
        ["AAAA"], tmp_path, mmseqs_bin="mmseqs", search_db="/db/uniref_pad",
        dry_run=True,
    )
    assert called == [] and res["AAAA"] is None


# --- s06b mode selection --------------------------------------------------- #
def _reps(n=2):
    return [SimpleNamespace(sequence=f"SEQ{i}AAAA") for i in range(n)]


def test_prepare_server_keeps_msa_server():
    cfg = InteractionModelConfig(rep_msa="server")
    cp_out, msas = _prepare_rep_msas(
        SimpleNamespace(), cfg, _reps(), ComplexPredictionConfig(use_msa_server=True)
    )
    assert cp_out.use_msa_server is True and msas == {}


def test_prepare_single_disables_server():
    cfg = InteractionModelConfig(rep_msa="single")
    cp_out, msas = _prepare_rep_msas(
        SimpleNamespace(), cfg, _reps(), ComplexPredictionConfig(use_msa_server=True)
    )
    assert cp_out.use_msa_server is False and msas == {}


def test_prepare_local_batches_and_disables_server(tmp_path, monkeypatch):
    captured = {}

    def fake_ensure(seqs, cache, **kw):
        captured["seqs"] = list(seqs)
        captured["kw"] = kw
        return {s.strip(): tmp_path / "x.a3m" for s in seqs}

    monkeypatch.setattr(local_msa, "ensure_local_msas", fake_ensure)
    cfg = InteractionModelConfig(rep_msa="local", rep_msa_max_seqs=1500)
    ctx = SimpleNamespace(
        config=SimpleNamespace(
            homologs=HomologConfig(
                database="/db/uniref30_seqdb_pad", mmseqs_gpu=True,
                search_threads=6, mmseqs_bin="/envs/evoliez/bin/mmseqs",
            )
        ),
        paths=SimpleNamespace(structures=tmp_path),
        dry_run=False,
    )
    cp_out, msas = _prepare_rep_msas(ctx, cfg, _reps(3), ComplexPredictionConfig())
    assert cp_out.use_msa_server is False
    assert len(captured["seqs"]) == 3
    assert captured["kw"]["max_seqs"] == 1500
    assert captured["kw"]["search_db"] == "/db/uniref30_seqdb_pad"
    assert captured["kw"]["mmseqs_bin"] == "/envs/evoliez/bin/mmseqs"


def test_prepare_local_requires_db():
    cfg = InteractionModelConfig(rep_msa="local")
    ctx = SimpleNamespace(
        config=SimpleNamespace(homologs=HomologConfig()),  # no database
        paths=SimpleNamespace(structures=None), dry_run=False,
    )
    with pytest.raises(ValueError, match="UniRef mmseqs DB"):
        _prepare_rep_msas(ctx, cfg, _reps(), ComplexPredictionConfig())


# --- config ---------------------------------------------------------------- #
def test_rep_msa_parses_and_rejects_bad():
    assert InteractionModelConfig(rep_msa="local").rep_msa == "local"
    with pytest.raises(Exception):
        InteractionModelConfig(rep_msa="cloud")
