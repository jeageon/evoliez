"""Integrated multi-source homology + ESM prior (roadmap P1.1).

s02 merges SEQUENCE (mmseqs/ColabFold) and STRUCTURE (Foldseek) homologs into
one set with a unified subfamily cluster space; s03 attaches an MSA-free ESM2
per-position variability prior. All exercised on the mock backend (the real
tools are server-only) following the codebase's mock/real adapter pattern.
"""

from pathlib import Path

from evoliez.adapters.esm import esm_position_priors
from evoliez.adapters.foldseek import search_structural_homologs
from evoliez.adapters.msa_tools import gather_homologs, search_homologs
from evoliez.config import Backend, HomologConfig, load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]
_SEQ = ("MKVLAYCVRPDEEQHGNWFKSTAARNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSAL"
        "LSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL")


# ---------------------------- Foldseek (structure) --------------------------
def test_foldseek_mock_structural_homologs(tmp_path):
    cfg = HomologConfig()
    hs = search_structural_homologs(_SEQ, cfg, tmp_path, backend=Backend.mock)
    assert hs and all(h.source == "structure" for h in hs)
    assert all(h.id.startswith("mock_struct") for h in hs)
    assert all(cfg.identity_min <= h.identity <= cfg.identity_max for h in hs)


# ------------------------------- ESM2 prior ---------------------------------
def test_esm_mock_prior_shape_and_determinism():
    a = esm_position_priors(_SEQ, model="esm2_t33_650M_UR50D", backend=Backend.mock)
    b = esm_position_priors(_SEQ, model="esm2_t33_650M_UR50D", backend=Backend.mock)
    assert len(a) == len(_SEQ)
    assert all(0.0 <= v <= 1.0 for v in a)
    assert a == b                                  # deterministic
    assert len(set(a)) > 1                         # not flat


# --------------------------- gather_homologs merge --------------------------
def test_gather_merges_sources_with_unified_clusters(tmp_path):
    cfg = HomologConfig(sources=["sequence", "structure"])
    merged = gather_homologs(_SEQ, cfg, tmp_path, backend=Backend.mock)
    srcs = {h.source for h in merged}
    assert srcs == {"sequence", "structure"}       # both contribute
    seqs = [h.sequence for h in merged]
    assert len(seqs) == len(set(seqs))             # deduped by sequence
    assert len({h.cluster_id for h in merged}) >= 3  # s06b holdout needs groups


def test_gather_single_source_is_sequence_only(tmp_path):
    cfg = HomologConfig()                           # default sources == [sequence]
    out = gather_homologs(_SEQ, cfg, tmp_path, backend=Backend.mock)
    assert out and all(h.source == "sequence" for h in out)
    # same population as the bare sequence search (backward compatible)
    base = search_homologs(_SEQ, cfg, tmp_path, backend=Backend.mock)
    assert len(out) == len(base)


def test_use_foldseek_legacy_alias_adds_structure(tmp_path):
    cfg = HomologConfig(use_foldseek=True)          # legacy flag, no 'sources'
    merged = gather_homologs(_SEQ, cfg, tmp_path, backend=Backend.mock)
    assert any(h.source == "structure" for h in merged)


# --------------------------- s02/s03 integration ----------------------------
def _cfg(tmp_path, **over):
    base = {"project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "mutation_generation": {"methods": ["chemistry_rules"],
                                    "max_candidates": 15},
            "gnn": {"build_dataset": False}}
    base.update(over)
    return load_config(ROOT / "configs" / "example_fdh_nadp.yaml", base)


def test_pipeline_multisource_and_esm(tmp_path):
    cfg = _cfg(tmp_path,
               homologs={"sources": ["sequence", "structure"]},
               msa={"esm_enabled": True})
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s03_msa")

    src = ctx.meta("homolog_sources", {})
    assert src.get("sequence", 0) > 0 and src.get("structure", 0) > 0
    feats = ctx.get("position_features", [])
    esm = [f.esm_variability for f in feats if f.target_position is not None]
    assert any(v > 0 for v in esm)                 # ESM prior attached
    assert ctx.meta("mean_esm_variability", 0.0) > 0.0


def test_pipeline_default_is_unchanged(tmp_path):
    # default config: single sequence source, ESM off -> no structure homologs,
    # esm_variability stays 0 (backward compatible).
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s03_msa")
    assert ctx.meta("homolog_sources", {}).get("structure", 0) == 0
    feats = ctx.get("position_features", [])
    assert all(f.esm_variability == 0.0 for f in feats)
