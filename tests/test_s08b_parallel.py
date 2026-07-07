"""s08b mutant-Boltz multi-GPU fan-out.

Re-predicting the top-N mutant complexes with Boltz is the same independent
per-item GPU work as the s06b ensemble, so s08b fans out across the pinned
CUDA_VISIBLE_DEVICES with the same GIL-free ProcessPool (each mutant scoped to
its own outdir; Δ computed serially afterward). Real Boltz is server-only, so
these run the mock predictor and assert every top-N mutant is predicted, the Δ
is attached, and the parallel result matches the serial one (reproducible).
"""

from __future__ import annotations

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.stages.s08b_mutant_boltz import MutantBoltzStage
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _to_s08b(tmp_path, cvd, monkeypatch):
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "mutation_generation": {"methods": ["chemistry_rules"],
                                    "max_candidates": 20, "multipoint": False},
            "reranking": {"top_for_redocking": 6, "top_for_md": 3,
                          "mutant_boltz_top_n": 5},
            "gnn": {"build_dataset": False},
        },
    )
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s08_reranker")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    MutantBoltzStage().run(ctx)
    return ctx


def _scores(ctx):
    return {c.candidate_id: {k: round(c.scores.get(k, 0.0), 4)
                             for k in ("d_ligand_iptm", "d_complex_iplddt")}
            for c in ctx.get("redock_candidates")
            if c.details.get("boltz_delta_source")}


def test_mutant_boltz_fans_out_and_predicts_all(tmp_path, monkeypatch):
    ctx = _to_s08b(tmp_path, "0,1,2", monkeypatch)         # >1 GPU -> ProcessPool
    evaluated = [c for c in ctx.get("redock_candidates")
                 if c.details.get("boltz_delta_source")]
    assert evaluated, "no mutants evaluated"
    assert len(ctx.get("mutant_complexes") or {}) >= 5     # every top-N predicted
    assert all("delta" in c.details for c in evaluated)


def test_parallel_matches_serial(tmp_path, monkeypatch):
    par = _scores(_to_s08b(tmp_path / "par", "0,1,2", monkeypatch))
    ser = _scores(_to_s08b(tmp_path / "ser", "2", monkeypatch))   # 1 GPU -> serial
    assert par and par == ser                              # deterministic mock


def test_single_gpu_serial(tmp_path, monkeypatch):
    ctx = _to_s08b(tmp_path, "2", monkeypatch)
    assert ctx.get("mutant_complexes")
    assert any(c.details.get("boltz_delta_source") for c in ctx.get("redock_candidates"))
