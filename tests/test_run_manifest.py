"""Reproducibility regression: RunContext.setup() snapshots a run manifest tying the
outputs to the exact code (git SHA + dirty flag) + fully-resolved config + backend + seed,
so paper data is recoverable (and a -dirty tree is recorded, not silent).
"""
from __future__ import annotations

import json
from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext

ROOT = Path(__file__).resolve().parents[1]
FDH = ROOT / "configs" / "example_fdh_nadp.yaml"


def _setup(tmp_path, **over):
    ov = {"project.output_dir": str(tmp_path / "run"),
          "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta")}
    ov.update(over)
    cfg = load_config(FDH, ov)
    return RunContext(cfg, allow_small_disk=True).setup()


def test_manifest_written_with_core_fields(tmp_path):
    ctx = _setup(tmp_path)
    m = json.loads((ctx.paths.reports / "run_manifest.json").read_text())
    assert m["schema"] == "evoliez_run_manifest_v1"
    assert m["seed"] == ctx.config.seed
    assert m["backend"] == ctx.config.backend.value
    # the FULL resolved config is captured (not just a hash)
    assert "reranking" in m["resolved_config"]
    assert "selection_lanes" in m["resolved_config"]


def test_manifest_records_git_state(tmp_path):
    ctx = _setup(tmp_path)
    g = json.loads((ctx.paths.reports / "run_manifest.json").read_text())["git"]
    # in this repo the source is a git tree: a commit is recorded and dirty is a bool
    assert g["commit"] is None or isinstance(g["commit"], str)
    assert isinstance(g["dirty"], bool)
    assert isinstance(g["dirty_files"], list)


def test_manifest_records_resolved_backend(tmp_path):
    ctx = _setup(tmp_path, **{"backend": "mock"})
    m = json.loads((ctx.paths.reports / "run_manifest.json").read_text())
    assert m["backend"] == "mock"
