"""ROADMAP_V3 review: every V4 MVP artifact (hardcoded _PROFILES fixtures) carries an
explicit data-provenance marker so it can never be mistaken for computed data, and the
pipeline-wired s04x reference-ensemble artifact is stamped seed-fixture in a real run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "_v4build", ROOT / "scripts" / "build_v4_mvp_artifacts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mark_synthetic_json_md_csv(tmp_path):
    mod = _load_builder()
    j = tmp_path / "a.json"
    j.write_text(json.dumps({"score": 0.722}))
    jl = tmp_path / "list.json"
    jl.write_text(json.dumps([{"id": "x"}]))
    md = tmp_path / "a.md"
    md.write_text("# Geometry\nmut_00479 0.722\n")
    csv = tmp_path / "a.csv"
    csv.write_text("mutation,score\nmut_00479,0.722\n")

    mod._mark_synthetic([j, jl, md, csv])

    jd = json.loads(j.read_text())
    assert jd["_data_provenance"]["data_provenance"] == "synthetic_fixture"
    assert jd["score"] == 0.722                       # original preserved
    jld = json.loads(jl.read_text())
    assert jld["_data_provenance"]["computed"] is False
    assert jld["items"] == [{"id": "x"}]              # list wrapped, not lost
    assert md.read_text().startswith("> ⚠ **SYNTHETIC")
    assert csv.read_text().splitlines()[0].startswith("# data_provenance=synthetic_fixture")


def test_s04x_artifact_is_stamped_seed_fixture(tmp_path):
    from evoliez.config import load_config
    from evoliez.context import RunContext
    from evoliez.pipeline import Pipeline
    from evoliez.utils.seeds import seed_everything
    cfg = load_config(ROOT / "configs" / "example_fdh_nadp.yaml", {
        "project.output_dir": str(tmp_path / "run"),
        "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
        "gnn": {"build_dataset": False},
        "reference_ensemble": {"enabled": True},
    })
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s04x_reference_ensemble")
    art = json.loads((ctx.paths.reports / "provenance" / "v4_reference_ensemble.json").read_text())
    assert art["_data_provenance"]["data_provenance"] == "seed_fixture_v0"
    assert art["_data_provenance"]["computed_from_this_run"] is False
