"""Full mock pipeline end-to-end: the core local verification (plan §Verification)."""

from pathlib import Path

from sqlalchemy import func

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.db.schema import (
    InteractionEdge,
    MDSimulation,
    MSAPosition,
    MutationCandidate,
    Sequence,
)
from evoliez.db.store import Store
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path: Path):
    return load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            # keep the test fast but exercise every stage
            "mutation_generation": {
                "methods": ["chemistry_rules", "msa_sampler", "ligandmpnn"],
                "max_candidates": 120,
                "design_radius_angstrom": 9.0,
                "ligandmpnn_samples": 8,
            },
            "reranking": {"top_for_redocking": 40, "top_for_md": 6,
                          "model": "xgboost", "use_experimental_labels": False},
            "validation": {"md": {"enabled": True, "protocol_level": 1,
                                   "top_candidates": 6}},
        },
    )


def test_full_mock_pipeline(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)

    paths = ctx.paths
    # spec 17.2 directory tree
    for d in (paths.inputs, paths.homologs, paths.msa, paths.complexes,
              paths.docking, paths.interaction_graphs, paths.mutations,
              paths.validation, paths.md, paths.reports):
        assert d.exists(), f"missing dir {d}"

    # spec 4.3 outputs
    assert (paths.reports / "final_report.md").exists()
    assert (paths.reports / "final_candidates.csv").exists()
    assert (paths.reports / "focused_library.csv").exists()
    assert (paths.reports / "session.pml").exists()

    # s06b family interaction-geometry model trained + persisted
    assert (paths.interaction_graphs / "interaction_model.json").exists()
    imeta = ctx.meta("interaction_model")
    assert imeta and imeta["train_rows"] > 0 and imeta["n_consensus"] > 0
    assert ctx.get("interaction_model") is not None

    # multi-level ML datasets + data-role policy (Boltz != label)
    dsd = paths.root / "ml_datasets"
    for name in ("pose_level", "edge_level", "residue_level",
                 "mutation_level", "variant_level"):
        assert (dsd / f"{name}.csv").exists(), f"missing dataset {name}"
    import csv as _csv
    import json as _json
    with (dsd / "edge_level.csv").open() as fh:
        cols = next(_csv.reader(fh))
    assert "contact_frequency" in cols  # priority-1 feature present
    roles = _json.loads((dsd / "roles.json").read_text())
    # the ONLY supervised-label column is the experimental one
    for tbl, colroles in roles["column_roles"].items():
        for col, role in colroles.items():
            if "supervised_label" in role:
                assert col == "experimental_label", (
                    f"{tbl}.{col} must not be a supervised label"
                )

    ranked = ctx.get("ranked_candidates")
    assert ranked and len(ranked) >= 1
    # scores are monotonically non-increasing (sorted)
    s = [c.scores["final_score"] for c in ranked]
    assert s == sorted(s, reverse=True)
    # every ranked candidate has the full score decomposition
    for c in ranked:
        assert "score_breakdown" in c.details
        for k in ("ml_score", "md_lite_score", "final_score",
                  "family_interaction_score"):
            assert k in c.scores
        assert 0.0 <= c.scores["family_interaction_score"] <= 1.0
        assert "family_interaction" in c.details["score_breakdown"]["contributions"]

    # spec 17.1 DB populated
    store = Store(paths.db_path)
    with store.session() as ses:
        assert ses.query(func.count(Sequence.sequence_id)).scalar() > 1
        assert ses.query(func.count(MSAPosition.id)).scalar() > 0
        assert ses.query(func.count(InteractionEdge.edge_id)).scalar() > 0
        assert ses.query(func.count(MutationCandidate.candidate_id)).scalar() >= 1
        assert ses.query(func.count(MDSimulation.md_id)).scalar() >= 1


def test_resume_is_idempotent(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)
    first = [c.scores["final_score"] for c in ctx.get("ranked_candidates")]

    # fresh context, same dir, resume: completed stages skipped, result stable
    ctx2 = RunContext(cfg, allow_small_disk=True).setup()
    assert ctx2.is_stage_done("s11_final")
    Pipeline().run(ctx2, resume=True)
    second = [c.scores["final_score"] for c in (ctx2.get("ranked_candidates") or [])]
    if second:  # s11 reran (no load impl) -> deterministic identical result
        assert first == second


def test_dry_run_real_backend_builds_commands(tmp_path):
    """backend=real + --dry-run must not raise even without GPU tools."""
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "dry"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "backend": "real",
            "msa": {"remote_server": False},
            "mutation_generation": {"methods": ["chemistry_rules"],
                                     "max_candidates": 20},
            "reranking": {"top_for_redocking": 10, "top_for_md": 3},
            "validation": {"md": {"top_candidates": 3}},
            "homologs": {"database": "/tmp/nonexistent_db"},
        },
    )
    ctx = RunContext(cfg, allow_small_disk=True)
    ctx.dry_run = True
    ctx.setup()
    Pipeline().run(ctx)
    assert (ctx.paths.reports / "final_report.md").exists()
