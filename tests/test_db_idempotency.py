"""Server-confirmed bug (expert audit): `--resume` does NOT actually skip
most stages because only s01/s06b have a `load()` implementation - the
others (s02..s11) always re-run. The combination of "re-runs" + "INSERT
without idempotency check" silently accumulates DockingPose and
MDSimulation rows on every resume cycle (1 -> 2 -> 4 -> ...).

Fix: s05_docking and s10_md now `delete(synchronize_session=False)` any
prior rows keyed on (project_id, candidate_id) before re-inserting fresh
results. Same intent as s04_complex's `if not s.query(...).first()`
pre-check, but delete-then-add so the latest run's results always win.

These tests lock the invariant: running the mock pipeline twice in the
same project produces the SAME number of DockingPose/MDSimulation rows
as one run - not double.
"""

from __future__ import annotations

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.db.schema import DockingPose, MDSimulation
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _row_counts(ctx: RunContext) -> tuple[int, int]:
    with ctx.store.session() as s:
        n_dp = s.query(DockingPose).filter_by(project_id=ctx.project_id).count()
        n_md = s.query(MDSimulation).filter_by(project_id=ctx.project_id).count()
    return n_dp, n_md


def _cfg(tmp_path: Path):
    return load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "mutation_generation": {"methods": ["chemistry_rules"],
                                    "max_candidates": 16,
                                    "design_radius_angstrom": 9.0},
            "reranking": {"top_for_redocking": 6, "top_for_md": 3,
                          "mutant_boltz_enabled": False},
            "validation": {"md": {"top_candidates": 3}},
            "gnn": {"build_dataset": False},
        },
    )


def test_resume_does_not_duplicate_docking_or_md_rows(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)

    # First full run
    ctx1 = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx1)
    n_dp_1, n_md_1 = _row_counts(ctx1)
    assert n_dp_1 > 0, "first run produced no DockingPose rows; test setup wrong"
    assert n_md_1 > 0, "first run produced no MDSimulation rows; test setup wrong"

    # Second run on the SAME project (the resume-without-load path):
    # s05 and s10 re-execute because they have no load(), but the
    # idempotency delete-then-add must keep row counts stable.
    ctx2 = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx2, resume=True)
    n_dp_2, n_md_2 = _row_counts(ctx2)

    assert n_dp_2 == n_dp_1, (
        f"DockingPose rows duplicated on resume: {n_dp_1} -> {n_dp_2}. "
        "s05_docking idempotency guard is missing or broken."
    )
    assert n_md_2 == n_md_1, (
        f"MDSimulation rows duplicated on resume: {n_md_1} -> {n_md_2}. "
        "s10_md idempotency guard is missing or broken."
    )


def test_third_resume_still_stable(tmp_path):
    # Catches accidental growth that only shows up after >= 2 resumes.
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)

    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)
    n0 = _row_counts(ctx)

    for i in range(2):
        ctx = RunContext(cfg, allow_small_disk=True).setup()
        Pipeline().run(ctx, resume=True)
        ni = _row_counts(ctx)
        assert ni == n0, (
            f"row counts drifted on resume #{i+2}: {n0} -> {ni}"
        )
