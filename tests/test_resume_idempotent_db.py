"""Re-running with --resume must not duplicate DB rows (expert review P2)."""

from pathlib import Path

from sqlalchemy import func

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.db.schema import (
    ComplexPrediction,
    DockingPose,
    MDSimulation,
    MutationCandidate,
    Sequence,
)
from evoliez.db.store import Store
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path):
    return load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh"
                                      / "target.fasta"),
            "mutation_generation": {"methods": ["chemistry_rules"],
                                    "max_candidates": 25,
                                    "design_radius_angstrom": 9.0},
            "reranking": {"top_for_redocking": 8, "top_for_md": 3},
            "validation": {"md": {"top_candidates": 3}},
            "gnn": {"build_dataset": False},
        },
    )


def _counts(db):
    s = Store(db)
    with s.session() as ses:
        return (
            ses.query(func.count(Sequence.sequence_id))
            .filter_by(source="homolog").scalar(),
            ses.query(func.count(ComplexPrediction.complex_id)).scalar(),
            ses.query(func.count(MutationCandidate.candidate_id)).scalar(),
            ses.query(func.count(DockingPose.pose_id)).scalar(),
            ses.query(func.count(MDSimulation.md_id)).scalar(),
        )


def test_resume_does_not_duplicate_db_rows(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)
    first = _counts(ctx.paths.db_path)
    # homologs, complexes, candidates, AND docking poses + MD rows all present
    assert all(v >= 1 for v in first), first

    # fresh context, same dir + config (fingerprint unchanged -> NOT
    # invalidated); stages re-run but inserts must be idempotent
    ctx2 = RunContext(_cfg(tmp_path), allow_small_disk=True).setup()
    assert not ctx2.invalidated
    Pipeline().run(ctx2, resume=True)
    second = _counts(ctx2.paths.db_path)
    assert second == first, f"DB rows duplicated on resume: {first} -> {second}"
