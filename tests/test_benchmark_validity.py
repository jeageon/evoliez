"""Benchmark overlap/validity guard + example benchmark consistency
(expert review: example benchmark was invalid vs the target sequence)."""

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.ml.benchmark import (
    load_benchmark,
    run_benchmark,
    validate_benchmark,
)
from evoliez.pipeline import Pipeline
from evoliez.types import Candidate, Mutation
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def test_validate_benchmark_flags_wt_mismatch():
    seq = "ACDEF"  # pos1=A
    w = validate_benchmark([{"mutation": "A1K"}, {"mutation": "K2L"}], seq)
    assert not any("A1K" in x for x in w)         # A1K matches
    assert any("K2L" in x and "WT" in x for x in w)  # pos2 is C, not K


def test_example_benchmark_is_sequence_consistent():
    seq = "".join(l.strip() for l in
                   open(ROOT / "examples" / "fdh" / "target.fasta")
                   if not l.startswith(">"))
    bench = load_benchmark(ROOT / "examples" / "fdh" / "benchmark.csv")
    warns = validate_benchmark(bench, seq)
    assert not any("WT '" in w for w in warns), warns   # all WT letters match


def test_run_benchmark_overlap_and_validity():
    ranked = [Candidate("c", [Mutation("E", 85, "S")], "g")]
    ranked[0].scores["final_score"] = 3.0
    ok = run_benchmark(
        ranked, [{"mutation": "E85S", "label": "beneficial",
                  "activity": 2.0}], k=1)
    assert ok["overlap"] == 1 and ok["valid"] is True

    bad = run_benchmark(
        ranked, [{"mutation": "Q999W", "label": "beneficial",
                  "activity": 2.0}], k=1)
    assert bad["overlap"] == 0 and bad["valid"] is False
    assert any("overlap" in w for w in bad["warnings"])


def test_example_benchmark_overlaps_mock_run(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {"project.output_dir": str(tmp_path / "run"),
         "input.target_fasta": str(ROOT / "examples" / "fdh"
                                   / "target.fasta"),
         "mutation_generation": {"methods": ["chemistry_rules"],
                                 "max_candidates": 40,
                                 "design_radius_angstrom": 9.0},
         "reranking": {"top_for_redocking": 20, "top_for_md": 4},
         "validation": {"md": {"top_candidates": 4}},
         "gnn": {"build_dataset": False}},
    )
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)
    r = run_benchmark(
        ctx.get("ranked_candidates", []),
        load_benchmark(ROOT / "examples" / "fdh" / "benchmark.csv"),
        k=20, target_sequence=ctx.get("target_sequence", ""),
    )
    assert r["overlap"] >= 1 and r["valid"] is True   # was 0 before the fix
    assert r["beneficial_recall_at_k"] > 0.0
