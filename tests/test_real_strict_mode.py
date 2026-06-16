"""Audit P0 #3 + #6 — a real run must not silently proceed on mock artifacts.

#3  adapters/ligand HARD-FAIL (RealToolError / ValueError) instead of degrading
    to a mock pose / synthetic structure, unless allow_mock_fallback is set.
#6  `doctor` reports a tool/dep the real config will INVOKE as BLOCK (a gate),
    not a mere MISSING.
"""

from pathlib import Path

import pytest

from evoliez.adapters import boltz
from evoliez.adapters.base import RealToolError
from evoliez.config import Backend, ComplexPredictionConfig, load_config
from evoliez.context import RunContext
from evoliez.diagnostics import BLOCK, MISSING, _required_under_real, collect
from evoliez.features.ligand import parse_ligand
from evoliez.pipeline import Pipeline
from evoliez.types import Ligand, LigandAtom

ROOT = Path(__file__).resolve().parents[1]


def _ligand():
    return Ligand(id="L", smiles="CCO",
                  atoms=[LigandAtom(id="C1", element="C", coord=(0.0, 0.0, 0.0))],
                  source="rdkit")


# --------------------------- #3: Boltz hard-fail ----------------------------
def test_boltz_hardfail_no_prediction(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLIEZ_DRY_RUN", "1")        # require() tolerant; run() no-op
    # real path, dry_run=False -> the (no-op) run leaves no boltz_results -> the
    # strict default refuses to substitute a mock complex.
    with pytest.raises(RealToolError):
        boltz.predict_complex("wt", "MKVLAYCVR", _ligand(),
                              ComplexPredictionConfig(), tmp_path,
                              backend=Backend.real, dry_run=False, seed=1)


def test_boltz_mock_fallback_when_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("EVOLIEZ_DRY_RUN", "1")
    monkeypatch.setenv("EVOLIEZ_ALLOW_MOCK_FALLBACK", "1")
    cx = boltz.predict_complex("wt", "MKVLAYCVR", _ligand(),
                               ComplexPredictionConfig(), tmp_path,
                               backend=Backend.real, dry_run=False, seed=1)
    assert cx is not None                              # degraded to mock, no raise


# --------------------------- #3: ligand hard-fail ---------------------------
def _cfg(tmp_path, backend, **over):
    ov = {"project.output_dir": str(tmp_path / "run"),
          "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
          "backend": backend}
    ov.update(over)
    return load_config(ROOT / "configs" / "example_fdh_nadp.yaml", ov)


def test_s01_hardfail_synthetic_ligand_under_real(tmp_path):
    bad = "this_is_not_a_valid_smiles_(((="
    assert parse_ligand(  # sanity: this really does fall back to synthetic
        type("L", (), {"id": "x", "type": "smiles", "value": bad})()
    ).source != "rdkit"
    cfg = _cfg(tmp_path, "real", **{"input.ligand": {"id": "x", "type": "smiles",
                                                     "value": bad}})
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    with pytest.raises(ValueError, match="rdkit"):
        Pipeline().run(ctx, to_stage="s01_input")


# --------------------------- #6: doctor is a gate ---------------------------
def test_required_set_real_vs_mock(tmp_path):
    real = _required_under_real(_cfg(tmp_path, "real"))
    tools, deps = real
    assert "boltz" in tools and "rdkit" in deps        # central real needs
    # a mock config requires nothing (everything is synthetic anyway)
    assert _required_under_real(_cfg(tmp_path, "mock")) == (set(), set())


def test_doctor_blocks_missing_required_tool(tmp_path):
    # boltz is not installed in the dev env; under a real config it must BLOCK
    # (a gate), but under a mock config the same absence is only MISSING.
    real_rep = collect(str(_write_cfg(tmp_path, "real")))
    mock_rep = collect(str(_write_cfg(tmp_path, "mock")))
    real_boltz = next(c for c in real_rep.checks if c.name == "tool:boltz")
    mock_boltz = next(c for c in mock_rep.checks if c.name == "tool:boltz")
    assert real_boltz.status == BLOCK
    assert mock_boltz.status == MISSING
    assert real_rep.n_block >= 1 and mock_rep.n_block == 0


def _write_cfg(tmp_path, backend):
    # collect() needs a path; materialise a minimal valid config.
    cfg = _cfg(tmp_path, backend)
    p = tmp_path / f"cfg_{backend}.yaml"
    import yaml
    p.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    return p
