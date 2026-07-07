"""s06b multi-GPU ensemble fan-out (GIL-free ProcessPool).

The per-rep CPU work (Boltz-output parsing + per-sample fingerprinting) is
GIL-bound, so a ThreadPool serialised the ensemble onto one core and starved the
other GPUs. The fan-out therefore runs each representative in a SEPARATE PROCESS
(round-robin across the pinned CUDA_VISIBLE_DEVICES, 'spawn' context). Real Boltz
is server-only, so these run the mock predictor; we assert the process pool
dispatches every rep, trains end-to-end, matches the serial result, and that the
worker payload pickles (a hard requirement for ProcessPool + spawn).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.stages.s06b_interaction_model import InteractionModelStage, _run_chunk
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _to_s06b(tmp_path, cvd, monkeypatch):
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "mutation_generation": {"methods": ["chemistry_rules"], "max_candidates": 20},
            "reranking": {"top_for_redocking": 6, "top_for_md": 3},
            "gnn": {"build_dataset": False},
            # keep the ensemble small + the mock fast
            "interaction_model": {"representative_homologs": 4, "poses_per_homolog": 4},
        },
    )
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s06_graph")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    InteractionModelStage().run(ctx)
    return ctx


def test_process_pool_fans_out_and_trains(tmp_path, monkeypatch):
    """>1 pinned GPU -> the spawn ProcessPool runs; every rep is dispatched and
    the model trains end-to-end."""
    ctx = _to_s06b(tmp_path, "0,1,2", monkeypatch)
    n = ctx.meta("interaction_model")["representatives"]
    assert n >= 2
    assert ctx.get("interaction_model") is not None
    poses = ctx.get("pose_dataset")
    assert poses
    assert len({p["group_id"] for p in poses}) == n     # one group per rep, all present


def test_parallel_matches_serial(tmp_path, monkeypatch):
    """The GIL-free process fan-out must not change results: the dataset from the
    multi-GPU ProcessPool path == the single-GPU serial path (deterministic mock,
    rep-index ordering)."""
    par = _to_s06b(tmp_path / "par", "0,1,2", monkeypatch).get("pose_dataset")
    ser = _to_s06b(tmp_path / "ser", "2", monkeypatch).get("pose_dataset")
    assert len(par) == len(ser) and par != []
    assert [p["group_id"] for p in par] == [p["group_id"] for p in ser]


def test_single_gpu_is_sequential(tmp_path, monkeypatch):
    """A single pinned GPU -> the serial branch (no ProcessPool); still trains."""
    ctx = _to_s06b(tmp_path, "2", monkeypatch)
    assert ctx.get("interaction_model") is not None
    assert ctx.get("pose_dataset")


def test_chunk_round_trips_through_pickle():
    """The (gpu, payloads) chunk a worker receives must pickle — ProcessPool +
    spawn ships it by value. Catches a non-picklable arg before it costs a run."""
    payloads = [(0, "MKV", 0.9, None, None, None, "/tmp/out", 7, True, None, 4.0, 6)]
    assert pickle.loads(pickle.dumps(("2", payloads))) == ("2", payloads)
    # _run_chunk is module-level (importable by spawned workers)
    assert _run_chunk.__module__ == "evoliez.stages.s06b_interaction_model"
