"""s08b per-mutant Boltz re-evaluation (expert review #2)."""

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.stages.s08b_mutant_boltz import _mutant_sequence
from evoliez.types import Candidate, Mutation
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def test_mutant_sequence_applies_substitutions():
    wt = "ACDEFGHIK"
    c = Candidate("c", [Mutation("C", 2, "W"), Mutation("K", 9, "R")], "g")
    assert _mutant_sequence(wt, c) == "AWDEFGHIR"


def _run(tmp_path, **ov):
    base = {
        "project.output_dir": str(tmp_path / "run"),
        "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
        "mutation_generation": {"methods": ["chemistry_rules"],
                                "max_candidates": 30,
                                "design_radius_angstrom": 9.0},
        "reranking": {"top_for_redocking": 10, "top_for_md": 3},
        "validation": {"md": {"top_candidates": 3}},
        "gnn": {"build_dataset": False},
    }
    base.update(ov)
    cfg = load_config(ROOT / "configs" / "example_fdh_nadp.yaml", base)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)
    return ctx


def test_top_n_get_real_delta_rest_proxy(tmp_path):
    ctx = _run(tmp_path, reranking={
        "top_for_redocking": 10, "top_for_md": 3,
        "mutant_boltz_enabled": True, "mutant_boltz_top_n": 3,
    })
    # PIPELINE-ORDER fix: s08b now runs AFTER s09, so the 3 candidates
    # that get real per-mutant Boltz are the SAME 3 that reach s10_md
    # (md_candidates) - not the s08-rerank-top-3 of redock_candidates.
    # This is the architectural property the fix establishes: every md
    # candidate has a real mutant structure, no s08b/s10 set mismatch.
    assert ctx.meta("n_mutant_boltz_evaluated") == 3
    md_cands = ctx.get("md_candidates", [])
    assert len(md_cands) == 3
    md_srcs = [c.details.get("boltz_delta_source") for c in md_cands]
    assert md_srcs == ["mock", "mock", "mock"], (
        "every md_candidate must carry real (mock-in-this-test) ΔBoltz"
    )
    # Candidates that reached redocking but NOT MD keep proxy ΔBoltz.
    redock = ctx.get("redock_candidates", [])
    md_ids = {c.candidate_id for c in md_cands}
    non_md = [c for c in redock if c.candidate_id not in md_ids]
    assert non_md, "redock pool should be larger than md pool"
    assert all(c.details.get("boltz_delta_source") == "proxy"
               for c in non_md), (
        "non-MD candidates must keep proxy ΔBoltz (s08b only updates MD set)"
    )


def test_disabled_keeps_proxy(tmp_path):
    ctx = _run(tmp_path, reranking={
        "top_for_redocking": 8, "top_for_md": 3,
        "mutant_boltz_enabled": False,
    })
    assert ctx.meta("n_mutant_boltz_evaluated") is None
    cands = ctx.get("redock_candidates", [])
    assert all(c.details.get("boltz_delta_source") == "proxy"
               for c in cands)
