"""ROADMAP_V3 B5+B6 regression:
  * B5 — the optional V4 reference-ensemble stage (s04x) is inserted into the pipeline
    after s04 when config.reference_ensemble.enabled, and absent by default;
  * B6 — the generic SeedManifest schema accepts non-FDH manifests (the FDH lead/control
    invariants are gated on target_id), and the manifest path is configurable.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.experimental.seed_manifest import SeedManifest, load_seed_manifest
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]
FDH = ROOT / "configs" / "example_fdh_nadp.yaml"


def _stage_names(enabled: bool):
    cfg = load_config(FDH, {"reference_ensemble": {"enabled": enabled}})
    return [s.name for s in Pipeline()._effective_stages(types.SimpleNamespace(config=cfg))]


def test_b5_stage_absent_by_default():
    assert "s04x_reference_ensemble" not in _stage_names(False)


def test_b5_stage_inserted_after_s04_when_enabled():
    names = _stage_names(True)
    assert "s04x_reference_ensemble" in names
    assert names[names.index("s04x_reference_ensemble") - 1] == "s04_complex"


def test_b5_config_field_default_off():
    c = load_config(FDH)
    assert c.reference_ensemble.enabled is False


def test_b6_fdh_default_manifest_still_loads():
    m = load_seed_manifest()
    assert m.target_id == "fdh_nadp"


def test_b6_generic_manifest_accepts_non_fdh():
    non_fdh = dict(
        schema_version=4.0, target_id="tem1_beta_lactamase",
        mechanism="nucleophilic_acyl_substitution", claim_level_default="L0_uncalibrated",
        source_runs={"r0": dict(date="2026-01-01", path="/x", reports=[], hash="abcd")},
        candidates=[dict(candidate_id="c1", mutation="E104K", role="hypothesis",
                         evidence_source=["v3"], claim_level="L0_uncalibrated")],
        controls=dict(), selection_policy=dict())
    sm = SeedManifest(**non_fdh)  # previously rejected for lacking mut_00479/Q382R
    assert sm.target_id == "tem1_beta_lactamase"


def test_b6_env_override(monkeypatch, tmp_path):
    # env path is honoured by load_seed_manifest(path=None)
    import shutil
    dst = tmp_path / "seed.yaml"
    shutil.copy(ROOT / "data" / "v4_seed_evidence" / "fdh_v1_v2_v3_manifest.yaml", dst)
    monkeypatch.setenv("EVOLIEZ_SEED_MANIFEST", str(dst))
    assert load_seed_manifest().target_id == "fdh_nadp"


def test_b5_integration_runs_s04x(tmp_path):
    cfg = load_config(FDH, {
        "project.output_dir": str(tmp_path / "run"),
        "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
        "gnn": {"build_dataset": False},
        "reference_ensemble": {"enabled": True},
    })
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s04x_reference_ensemble")
    assert ctx.get("reference_ensemble") is not None
    assert (ctx.paths.reports / "provenance" / "v4_reference_ensemble.json").exists()
