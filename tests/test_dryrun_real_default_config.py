"""`evoliez run --backend real --dry-run` must work on the DEFAULT example
config (no homologs.database). Regression for the reported dry-run UX break:
the missing-DB hard check ran before the dry-run command preview.
"""

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_real_dry_run_on_example_config_without_database(tmp_path):
    # exactly the user's CLI scenario: example config, backend real, dry-run,
    # NO homologs.database override
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "dry"),
            "input.target_fasta": str(ROOT / "examples" / "fdh"
                                      / "target.fasta"),
            "backend": "real",
            "mutation_generation": {"methods": ["chemistry_rules"],
                                    "max_candidates": 20},
            "reranking": {"top_for_redocking": 8, "top_for_md": 3},
            "validation": {"md": {"top_candidates": 3}},
            "gnn": {"build_dataset": False},
        },
    )
    assert cfg.homologs.database is None       # the triggering condition
    ctx = RunContext(cfg, allow_small_disk=True)
    ctx.dry_run = True
    ctx.setup()
    Pipeline().run(ctx)                         # must NOT raise
    assert (ctx.paths.reports / "final_report.md").exists()
