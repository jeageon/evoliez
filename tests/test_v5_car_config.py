"""ROADMAP_V5 — CAR V5 config loads with the adenylation mechanism, legacy FDH configs still
load in legacy mode, and the geometry source-of-truth is recorded honestly."""
from pathlib import Path

from evoliez.config import load_config
from evoliez.mechanism.mode import (
    SRC_LEGACY_FROM_MECHANISM, mechanism_mode, resolve_geometry_source)

ROOT = Path(__file__).resolve().parents[1]


def test_car_v5_mechanism_config_loads():
    cfg = load_config(str(ROOT / "configs" / "car_srcar_3hp_v5.yaml"))
    assert cfg.mechanism is not None
    assert cfg.mechanism.reaction.cls == "adenylation_phosphoryl_transfer"
    assert cfg.mechanism.reaction_state.cofactor_state == "ATP"
    assert "Mg" in (cfg.mechanism.reaction_state.metal_state or "")
    assert mechanism_mode(cfg) == "mechanism_spec"
    # the mechanism emits the O->P geometry terms
    labels = [g.label for g in cfg.mechanism.to_geometry_terms()]
    assert "nuc_O_to_alphaP" in labels and "inline_attack_Onuc_Pa_Oleaving" in labels


def test_car_v5_geometry_source_is_legacy_from_mechanism():
    # legacy reactive_geometry runs (it carries the V5-1 O->P fix) while mechanism is declared
    cfg = load_config(str(ROOT / "configs" / "car_srcar_3hp_v5.yaml"))
    assert cfg.validation.md.reactive_geometry.enabled is True
    assert cfg.validation.md.reactive_geometry.transfer_is_h is False
    src = resolve_geometry_source(cfg, cfg.mechanism.to_geometry_terms(), nac_enabled=True)
    assert src == SRC_LEGACY_FROM_MECHANISM


def test_legacy_fdh_config_still_loads_in_legacy_mode():
    for name in ("prod_fdh_nadp.yaml", "example_fdh_nadp.yaml"):
        cfg = load_config(str(ROOT / "configs" / name))
        assert cfg.mechanism is None
        assert mechanism_mode(cfg) == "legacy"
