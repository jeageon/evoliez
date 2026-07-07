"""ROADMAP_V5 V5-C — the config-template surface (configs/templates/mechanisms/*.yaml) must
stay CONSISTENT with the code registry, so the documented "config-only" mechanism declaration
never drifts from what MechanismSpec actually resolves."""
from pathlib import Path

import pytest
import yaml

from evoliez.mechanism.templates import get_template

TEMPL_DIR = Path(__file__).resolve().parents[1] / "configs" / "templates" / "mechanisms"


def _template_yaml_files():
    return sorted(TEMPL_DIR.glob("*.yaml"))


def test_there_is_a_yaml_for_every_real_geometry_template():
    yaml_keys = {yaml.safe_load(f.read_text())["key"] for f in _template_yaml_files()}
    real = {k for k in ("hydride_transfer", "nucleophilic_acyl_substitution",
                        "glycosidic_bond_cleavage", "adenylation_phosphoryl_transfer")}
    missing = real - yaml_keys
    assert not missing, f"missing config-template YAML for: {missing}"


@pytest.mark.parametrize("f", _template_yaml_files(), ids=lambda f: f.stem)
def test_yaml_template_matches_code_registry(f):
    doc = yaml.safe_load(f.read_text())
    code = get_template(doc["key"])
    assert code is not None, f"{doc['key']} not in code registry"
    # required reaction-state fields must match exactly (the hard-gate contract)
    assert set(doc["required_reaction_state"]) == set(code["required_reaction_state"])
    # geometry-term labels + kinds must match (the geometry the spec will emit)
    y_terms = [(t["label"], t["kind"]) for t in doc.get("default_geometry_terms", [])]
    c_terms = [(t["label"], t["kind"]) for t in code.get("default_geometry_terms", [])]
    assert y_terms == c_terms, f"{doc['key']}: yaml {y_terms} != code {c_terms}"
