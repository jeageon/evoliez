"""Resume-checkpoint invalidation (expert review priority #1).

A change to config / input ligand / backend / software version must wipe the
resume state so stale artifacts are never silently reused.
"""

from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, **ov):
    base = {
        "project.output_dir": str(tmp_path / "run"),
        "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
        "mutation_generation": {"methods": ["chemistry_rules"],
                                "max_candidates": 30,
                                "design_radius_angstrom": 9.0},
        "reranking": {"top_for_redocking": 12, "top_for_md": 4},
        "validation": {"md": {"enabled": True, "protocol_level": 1,
                              "top_candidates": 4}},
        "gnn": {"build_dataset": False},
    }
    base.update(ov)
    return load_config(ROOT / "configs" / "example_fdh_nadp.yaml", base)


def _setup(cfg, tmp_path):
    return RunContext(cfg, allow_small_disk=True).setup()


def test_fingerprint_deterministic(tmp_path):
    cfg = _cfg(tmp_path)
    a = RunContext(cfg, allow_small_disk=True).run_fingerprint()
    b = RunContext(cfg, allow_small_disk=True).run_fingerprint()
    assert a == b
    assert set(a) == {"evoliez_version", "ranking_formula_version",
                      "backend", "dry_run", "input_sha1", "config_sha1"}


def test_unchanged_config_keeps_resume(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = _setup(cfg, tmp_path)
    Pipeline().run(ctx)
    assert ctx.is_stage_done("s11_final")
    assert not ctx.invalidated
    sp = ctx.paths.state_path
    assert "fingerprint" in __import__("json").loads(sp.read_text())

    # same config, fresh context, same dir -> resume preserved
    ctx2 = _setup(_cfg(tmp_path), tmp_path)
    assert not ctx2.invalidated
    assert ctx2.is_stage_done("s11_final")


def test_config_change_invalidates(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = _setup(cfg, tmp_path)
    Pipeline().run(ctx)
    assert ctx.is_stage_done("s06_graph")

    ctx2 = _setup(_cfg(tmp_path, **{"seed": 99}), tmp_path)
    assert ctx2.invalidated
    assert not ctx2.is_stage_done("s06_graph")
    assert ctx2._state["completed_stages"] == []


def test_ligand_change_invalidates(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = _setup(cfg, tmp_path)
    Pipeline().run(ctx)

    ctx2 = _setup(
        _cfg(tmp_path, **{"input.ligand.value": "CCO"}), tmp_path
    )
    assert ctx2.invalidated
    assert not ctx2.is_stage_done("s11_final")


def test_backend_change_invalidates(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = _setup(cfg, tmp_path)
    Pipeline().run(ctx)

    ctx2 = _setup(_cfg(tmp_path, **{"backend": "real"}), tmp_path)
    assert ctx2.invalidated
    assert not ctx2.is_stage_done("s11_final")
