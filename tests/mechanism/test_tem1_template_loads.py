"""V4-10: the TEM-1 non-FDH smoke config loads config-only (ROADMAP_V4 §7.11)."""

import copy
from pathlib import Path

import pytest
import yaml

from evoliez.mechanism.spec import MechanismSpec

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs" / "benchmarks" / "tem1_v4_smoke.yaml"


def _load():
    return yaml.safe_load(CFG.read_text())


def test_tem1_mechanism_loads_config_only():
    spec = MechanismSpec(**_load()["mechanism"])
    assert spec.reaction.cls == "nucleophilic_acyl_substitution"
    # geometry terms are filled from the registered template
    assert len(spec.geometry_terms) == 2
    selectors = {(t.a_residue, t.a_atom) for t in spec.geometry_terms}
    assert ("SER", "OG") in selectors


def test_missing_required_reaction_state_hard_fails():
    data = copy.deepcopy(_load())
    data["mechanism"]["reaction_state"].pop("protonation_model")
    with pytest.raises(ValueError):
        MechanismSpec(**data["mechanism"])
