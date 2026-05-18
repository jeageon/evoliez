from pathlib import Path

import pytest

from evoliez.config import Backend, Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads():
    cfg = load_config(ROOT / "configs" / "example_fdh_nadp.yaml")
    assert cfg.project.objective == "cofactor_switching"
    assert cfg.backend is Backend.mock
    assert "ligandmpnn" in cfg.mutation_generation.methods
    assert cfg.input.ligand.type == "smiles"


def test_default_config_loads():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.validation.md.protocol_level == 1
    assert cfg.backend_for("s04_complex") is Backend.mock


def test_overrides_and_per_stage_backend():
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {"backend": "real", "backends": {"s10_md": "mock"}},
    )
    assert cfg.backend is Backend.real
    assert cfg.backend_for("s10_md") is Backend.mock
    assert cfg.backend_for("s04_complex") is Backend.real


def test_unknown_key_rejected():
    with pytest.raises(Exception):
        Config.model_validate({"input": {"ligand": {"value": "C"}}, "bogus": 1})


def test_input_requires_sequence_source():
    with pytest.raises(Exception):
        Config.model_validate({"input": {"ligand": {"value": "C"}}})
