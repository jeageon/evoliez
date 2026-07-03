"""ROADMAP_V3 B4 regression: MechanismSpec is wired into Config + the pipeline.

Before: MechanismSpec existed but was not a Config field and no stage published it —
ctx "mechanism" came only from the legacy features.mechanism heuristic. Now:
  * Config has an optional ``mechanism`` field, hard-gate validated at config-load;
  * s06 (s06_graph) publishes ``mechanism_spec`` + runtime ``geometry_terms`` to ctx;
  * a hydride config lifts to the SAME geometry the legacy NAC config used.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.mechanism.spec import ReactionState, mechanism_from_reactive_geometry
from evoliez.pipeline import Pipeline
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]
FDH = ROOT / "configs" / "example_fdh_nadp.yaml"
MECH_OK = {
    "reaction": {"class": "hydride_transfer"},
    "reaction_state": {"cofactor_redox_state": "NADP+",
                       "conformational_state": "closed_ternary"},
}


def test_config_has_mechanism_field_default_none():
    c = load_config(FDH)
    assert "mechanism" in type(c).model_fields
    assert c.mechanism is None  # opt-in: absent by default


def test_valid_mechanism_loads_and_fills_template_geometry():
    c = load_config(FDH, {"mechanism": MECH_OK})
    assert c.mechanism is not None
    assert c.mechanism.reaction.cls == "hydride_transfer"
    # template supplied default geometry terms (distance + angle)
    assert len(c.mechanism.geometry_terms) == 2
    assert c.mechanism.to_geometry_terms()  # converts to runtime GeometryTerm objects


def test_missing_required_reaction_state_aborts_config_load():
    # hydride_transfer requires cofactor_redox_state + conformational_state
    with pytest.raises(ValidationError):
        load_config(FDH, {"mechanism": {"reaction": {"class": "hydride_transfer"}}})


def test_unknown_reaction_class_aborts_config_load():
    with pytest.raises(ValidationError):
        load_config(FDH, {"mechanism": {"reaction": {"class": "not_a_real_class"},
                                        "reaction_state": {"pH": 7.0}}})


def test_legacy_nac_lifts_to_same_geometry():
    prod = load_config(ROOT / "configs" / "prod_fdh_nadp.yaml")
    rg = prod.validation.md.reactive_geometry
    st = ReactionState(cofactor_redox_state="NADP+", conformational_state="closed_ternary")
    terms = mechanism_from_reactive_geometry(rg, st).to_geometry_terms()
    assert terms[0].a_smarts == rg.donor_smarts
    assert terms[0].b_smarts == rg.acceptor_smarts
    assert terms[0].distance_max == rg.distance_max
    assert terms[1].angle_min == rg.angle_min


def _run_to_s06(tmp_path, *, mechanism):
    overrides = {
        "project.output_dir": str(tmp_path / "run"),
        "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
        "gnn": {"build_dataset": False},
    }
    if mechanism is not None:
        overrides["mechanism"] = mechanism
    cfg = load_config(FDH, overrides)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s06_graph")
    return ctx


def test_s06_publishes_mechanism_spec_and_geometry_terms(tmp_path):
    ctx = _run_to_s06(tmp_path, mechanism=MECH_OK)
    spec = ctx.get("mechanism_spec")
    assert spec is not None and spec.reaction.cls == "hydride_transfer"
    terms = ctx.get("geometry_terms")
    assert terms and len(terms) == 2


def test_s06_no_mechanism_leaves_ctx_unset(tmp_path):
    ctx = _run_to_s06(tmp_path, mechanism=None)
    assert ctx.get("mechanism_spec") is None
