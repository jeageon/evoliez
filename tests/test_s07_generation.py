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


# --------------------------------------------------------------------------- #
# paper-grade provenance: per-mutation distances + forbidden-check columns
# --------------------------------------------------------------------------- #
import csv  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402

from evoliez.stages.s07_mutation_gen import (  # noqa: E402
    MutationGenStage, _is_radical_substitution)
from evoliez.types import Candidate, Mutation  # noqa: E402


def _to_s07_extras(tmp_path, extras=None, **mg):
    """Like _to_s07 but keeps the config's real catalytic/fixed numbering and can
    co-model extra ligands (to exercise the dynamic distance_to_<id> columns)."""
    over = {
        "project.output_dir": str(tmp_path / "run"),
        "mutation_generation": {
            "methods": ["chemistry_rules", "msa_sampler", "ligandmpnn"],
            "max_candidates": 80, **mg},
        "gnn": {"build_dataset": False},
    }
    if extras is not None:
        over["input.extra_ligands"] = extras
    cfg = load_config(ROOT / "configs" / "example_fdh_nadp.yaml", over)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s07_mutation_gen")
    return ctx


def _gen_rows(ctx):
    prov = ctx.paths.reports / "provenance"
    return json.loads((prov / "generated_candidates.json").read_text())


def _csv_header(path):
    with Path(path).open() as fh:
        return next(csv.reader(fh))


def test_provenance_distance_and_forbidden_columns(tmp_path):
    ctx = _to_s07_extras(
        tmp_path,
        extras=[{"id": "formate", "type": "smiles", "value": "[O-]C=O"}])
    prov = ctx.paths.reports / "provenance"
    hdr = _csv_header(prov / "generated_candidates.csv")
    # the design-ligand + nearest-catalytic distance columns + forbidden_check
    for col in ("distance_to_design_ligand", "distance_to_nearest_catalytic",
                "forbidden_check", "risk_flag", "disallowed_reason"):
        assert col in hdr, f"missing column {col}: {hdr}"
    # dynamic per-extra-ligand column derived from the config ligand id (generic)
    assert "distance_to_formate" in hdr, hdr
    # no column appears twice
    assert len(hdr) == len(set(hdr)), [h for h in hdr if hdr.count(h) > 1]

    rows = _gen_rows(ctx)
    assert rows
    # every row carries the static distance/forbidden fields; forbidden==ok since
    # the generators exclude protected positions
    for r in rows:
        assert r["forbidden_check"] == "ok"
        assert "distance_to_design_ligand" in r
        assert "distance_to_formate" in r
    # at least one candidate has a numeric design-ligand distance
    assert any(isinstance(r["distance_to_design_ligand"], (int, float))
               for r in rows)
    # a multi-mutation candidate reports a per-position distance list
    multi = [r for r in rows if r["n_mutations"] >= 2]
    if multi:
        assert any(r.get("distance_to_design_ligand_per_pos") for r in multi)


# --------------------------------------------------------------------------- #
# post-hoc protected assert (cross-generator verification)
# --------------------------------------------------------------------------- #
def test_protected_assert_passes_on_real_run(tmp_path):
    # a normal run must not raise (the generators already exclude protected)
    ctx = _to_s07_extras(tmp_path)
    cands = ctx.get("candidates")
    cat = set(ctx.get("catalytic_positions") or [])
    fixed = set(ctx.get("fixed_positions") or [])
    for c in cands:
        for m in c.mutations:
            assert m.position not in cat and m.position not in fixed


def test_protected_assert_raises_on_violation():
    # feed the assert a candidate that DOES touch a protected position -> raise
    stage = MutationGenStage()
    good = Candidate("mut_0", [Mutation("A", 50, "K")], "test")
    bad = Candidate("mut_1", [Mutation("H", 154, "K")], "test")
    with pytest.raises(RuntimeError, match="PROTECTED-POSITION VIOLATION"):
        stage._assert_no_protected([good, bad], {154, 285}, {15, 33, 154, 285})
    # and it must NOT raise when everything is designable
    stage._assert_no_protected([good], {154, 285}, {15, 33, 154, 285})


# --------------------------------------------------------------------------- #
# risk flags + configurable filter-vs-flag
# --------------------------------------------------------------------------- #
def test_radical_substitution_classifier():
    assert _is_radical_substitution("D", "K")     # neg -> pos (class flip)
    assert _is_radical_substitution("A", "W")     # tiny -> huge (volume jump)
    assert not _is_radical_substitution("L", "I")  # conservative hydrophobic
    assert not _is_radical_substitution("D", "E")  # both negative, small delta


def test_risk_flags_recorded_default_flag_only(tmp_path):
    ctx = _to_s07_extras(tmp_path)  # risk_filter defaults False
    rows = _gen_rows(ctx)
    # risk_flag present on every row; reason non-empty exactly when flagged
    for r in rows:
        assert "risk_flag" in r and "disallowed_reason" in r
        if r["risk_flag"]:
            assert r["disallowed_reason"]
    # flag-only: a flagged Gly/Pro/Cys-introducing or radical candidate is KEPT
    # in ctx (not dropped) -> the full pool is shipped
    assert len(ctx.get("candidates")) == ctx.meta("n_candidates_generated")
    # Pro/Gly/Cys introductions are flagged by reason text when present
    flagged = [r for r in rows if r["risk_flag"]]
    if flagged:
        assert any("introduces_" in r["disallowed_reason"]
                   or "radical_2shell" in r["disallowed_reason"]
                   for r in flagged)


def test_risk_filter_drops_flagged(tmp_path):
    flag_ctx = _to_s07_extras(tmp_path)
    flagged_ids = {r["candidate_id"] for r in _gen_rows(flag_ctx)
                   if r["risk_flag"]}
    if not flagged_ids:
        pytest.skip("no risk-flagged candidates in this deterministic run")
    drop_ctx = _to_s07_extras(tmp_path, risk_filter=True)
    shipped = {c.candidate_id for c in drop_ctx.get("candidates")}
    # none of the flagged candidates survive into ctx when filtering is on
    assert not (shipped & flagged_ids)
    # but the provenance table STILL records the full pre-filter pool (audit)
    prov_ids = {r["candidate_id"] for r in _gen_rows(drop_ctx)}
    assert flagged_ids <= prov_ids


# --------------------------------------------------------------------------- #
# budget tiers (single / multipoint / risky) + config fields
# --------------------------------------------------------------------------- #
def test_budget_tier_files_and_caps(tmp_path):
    ctx = _to_s07_extras(
        tmp_path, tier_single_top=8, tier_multipoint_top=4, tier_risky_top=3)
    prov = ctx.paths.reports / "provenance"
    # the combined file is still written alongside the three tiers
    assert (prov / "generated_candidates.csv").exists()
    for tier, cap in (("single", 8), ("multipoint", 4), ("risky", 3)):
        jf = prov / f"generated_candidates_{tier}.json"
        cf = prov / f"generated_candidates_{tier}.csv"
        assert jf.exists() and cf.exists(), f"missing tier file {tier}"
        rows = json.loads(jf.read_text())
        assert len(rows) <= cap, f"{tier} exceeded its cap {cap}"
    # single tier holds only single-mutation, non-risky candidates
    single = json.loads(
        (prov / "generated_candidates_single.json").read_text())
    assert all(r["n_mutations"] == 1 and not r["risk_flag"] for r in single)
    multi = json.loads(
        (prov / "generated_candidates_multipoint.json").read_text())
    assert all(r["n_mutations"] >= 2 and not r["risk_flag"] for r in multi)
    risky = json.loads(
        (prov / "generated_candidates_risky.json").read_text())
    assert all(r["risk_flag"] for r in risky)
    # ctx candidates remains the FULL set (tiers are a view, not a prune)
    assert len(ctx.get("candidates")) == ctx.meta("n_candidates_generated")


def test_tier_config_field_defaults():
    from evoliez.config import MutationGenConfig
    c = MutationGenConfig()
    assert c.tier_single_top == 100
    assert c.tier_multipoint_top == 30
    assert c.tier_risky_top == 20
    assert c.risk_filter is False
