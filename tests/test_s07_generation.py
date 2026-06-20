"""s07 multi-strategy generation: per-generator quotas, FuncLib multipoint,
and the designable-only / catalytic-safe guarantee."""

from __future__ import annotations

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _to_s07(tmp_path, **mg):
    over = {
        "project.output_dir": str(tmp_path / "run"),
        "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
        "input.catalytic_residues": ["H120"],
        "mutation_generation": {
            "methods": ["chemistry_rules", "msa_sampler", "ligandmpnn"],
            "max_candidates": 80, **mg},
        "gnn": {"build_dataset": False},
    }
    cfg = load_config(ROOT / "configs" / "example_fdh_nadp.yaml", over)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s07_mutation_gen")
    return ctx


def test_quotas_prevent_generator_monopoly(tmp_path):
    ctx = _to_s07(tmp_path)
    cands = ctx.get("candidates")
    assert cands
    gens: dict = {}
    for c in cands:
        gens[c.generator] = gens.get(c.generator, 0) + 1
    # several generators represented — chemistry_rules did NOT fill the small cap
    assert len(gens) >= 2, gens
    # no single generator owns the whole library
    assert max(gens.values()) < len(cands), gens


def test_funclib_multipoint_candidates(tmp_path):
    ctx = _to_s07(tmp_path)
    cands = ctx.get("candidates")
    multi = [c for c in cands if len(c.mutations) >= 2]
    assert multi, "no multi-point combinations generated"
    assert any(c.generator == "multipoint" for c in cands)
    # multipoint stays within the configured order
    assert all(len(c.mutations) <= 3 for c in cands if c.generator == "multipoint")


def test_never_mutates_catalytic_or_fixed(tmp_path):
    ctx = _to_s07(tmp_path)
    cands = ctx.get("candidates")
    cat = set(ctx.get("catalytic_positions") or [])
    fixed = set(ctx.get("fixed_positions") or [])
    for c in cands:
        for m in c.mutations:
            assert m.position not in cat and m.position not in fixed


def test_disabling_multipoint(tmp_path):
    ctx = _to_s07(tmp_path, multipoint=False)
    cands = ctx.get("candidates")
    assert cands
    assert not any(c.generator == "multipoint" for c in cands)
