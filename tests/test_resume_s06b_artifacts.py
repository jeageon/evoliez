"""s06b --resume must restore the FULL bus state, not just the model.

Regression: InteractionModelStage.load() used to rebuild only interaction_model,
while run() also puts ensemble_contacts / pose_dataset / edge_dataset. On a
--resume that skipped s06b, s08 and s11 then read those via ctx.get(key, []) and
silently got [] -> truncated ML datasets and missing edge features. load() now
persists and restores all four (or returns False to force a clean re-run).
"""

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.stages.s06b_interaction_model import InteractionModelStage
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path):
    return load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "mutation_generation": {"methods": ["chemistry_rules"],
                                    "max_candidates": 25},
            "reranking": {"top_for_redocking": 8, "top_for_md": 3},
            "gnn": {"build_dataset": False},
        },
    )


def test_s06b_load_restores_all_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s06b_interaction")

    # produced during the run
    assert ctx.get("interaction_model") is not None
    assert ctx.get("ensemble_contacts")           # non-empty list
    assert ctx.get("pose_dataset")
    assert ctx.get("edge_dataset")

    # fresh process/context, same dir+config -> resume. load() must restore
    # EVERY artifact run() put, not just the model.
    ctx2 = RunContext(_cfg(tmp_path), allow_small_disk=True).setup()
    assert not ctx2.invalidated
    assert ctx2.is_stage_done("s06b_interaction")
    st = InteractionModelStage()
    assert st.load(ctx2) is True

    assert ctx2.get("interaction_model") is not None
    assert ctx2.get("ensemble_contacts"), "ensemble_contacts NOT restored on resume"
    assert ctx2.get("pose_dataset"), "pose_dataset NOT restored on resume"
    assert ctx2.get("edge_dataset"), "edge_dataset NOT restored on resume"
    # restored content matches what the run produced
    assert len(ctx2.get("ensemble_contacts")) == len(ctx.get("ensemble_contacts"))
    assert len(ctx2.get("edge_dataset")) == len(ctx.get("edge_dataset"))


def test_s06b_load_returns_false_when_artifacts_missing(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s06b_interaction")

    # simulate a partial/corrupt checkpoint: drop the artifacts file
    art = ctx.paths.interaction_graphs / "s06b_artifacts.json"
    assert art.exists()
    art.unlink()

    ctx2 = RunContext(_cfg(tmp_path), allow_small_disk=True).setup()
    st = InteractionModelStage()
    # must force a re-run rather than hand downstream empty data
    assert st.load(ctx2) is False
