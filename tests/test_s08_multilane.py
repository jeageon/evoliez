"""ROADMAP_V3 B1/B2 regression: the s08 redock queue must be a multi-lane UNION
(not a single ml_score top-N cut) when selection_lanes is enabled — so an ML
false-negative (a low-ml candidate) can still reach s08b/s09 instead of being
silently dropped at s08 and never getting a real structure or MD.

Runs the pipeline to s08_reranker with the mock backend (same pattern as
test_s08b_parallel) and contrasts lanes-off vs lanes-on redock sets.
"""
from __future__ import annotations

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _run_to_s08(tmp_path, *, lanes: bool):
    overrides = {
        "project.output_dir": str(tmp_path / "run"),
        "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
        "mutation_generation": {"methods": ["chemistry_rules"],
                                "max_candidates": 30, "multipoint": False},
        "reranking": {"top_for_redocking": 5, "top_for_md": 3,
                      "mutant_boltz_top_n": 5},
        "gnn": {"build_dataset": False},
    }
    if lanes:
        overrides["selection_lanes"] = {
            "enabled": True, "from_ml_high": 4, "from_stability_high": 0,
            "from_geometry_high": 0, "from_diversity": 2, "low_ml_controls": 3,
        }
    cfg = load_config(ROOT / "configs" / "example_fdh_nadp.yaml", overrides)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s08_reranker")
    return ctx


def test_lanes_off_is_scalar_top_n(tmp_path):
    ctx = _run_to_s08(tmp_path, lanes=False)
    redock = ctx.get("redock_candidates")
    cands = ctx.get("candidates")
    assert redock, "expected a redock queue"
    # scalar path: exactly top_for_redocking by ml_score, no lane tag
    assert len(redock) == 5
    top5 = sorted(cands, key=lambda c: -c.scores.get("ml_score", 0.0))[:5]
    assert {c.candidate_id for c in redock} == {c.candidate_id for c in top5}
    assert all(c.details.get("selection_lane") is None for c in redock)


def test_lanes_on_keeps_a_low_ml_candidate(tmp_path):
    ctx = _run_to_s08(tmp_path, lanes=True)
    redock = ctx.get("redock_candidates")
    cands = ctx.get("candidates")
    assert redock, "expected a redock queue"

    # every promoted candidate carries the lane that surfaced it
    assert all(c.details.get("selection_lane") for c in redock)
    lanes = {c.details["selection_lane"] for c in redock}
    assert "low_ml_control" in lanes, f"no false-negative probe lane; got {lanes}"

    # the low_ml_control candidate is provably one the scalar top-N would have cut:
    # its ml_score is below the 5th-highest ml_score.
    by_ml = sorted(cands, key=lambda c: -c.scores.get("ml_score", 0.0))
    cutoff = by_ml[4].scores.get("ml_score", 0.0)  # top_for_redocking == 5
    low = [c for c in redock if c.details["selection_lane"] == "low_ml_control"]
    assert low
    assert any(c.scores.get("ml_score", 0.0) <= cutoff for c in low), (
        "low_ml_control lane did not actually rescue a below-cut candidate")


def test_lanes_augment_never_shrink_below_scalar_baseline(tmp_path):
    """Regression: lanes-on must NOT collapse the redock funnel below the scalar
    top_for_redocking (the bug where from_ml_high<<top_for_redocking starved s08b/s09
    and the wet-lab plate)."""
    ctx_off = _run_to_s08(tmp_path / "off", lanes=False)
    ctx_on = _run_to_s08(tmp_path / "on", lanes=True)
    n_off = len(ctx_off.get("redock_candidates"))
    n_on = len(ctx_on.get("redock_candidates"))
    # ml_high lane == top_for_redocking, so lanes-on is the scalar set PLUS probes
    assert n_on >= n_off, f"lanes-on ({n_on}) shrank the redock funnel below scalar ({n_off})"
