"""ROADMAP_V5 V5-3 — adenylation_phosphoryl_transfer mechanism template (CAR / NRPS A-domain).

The ANL-superfamily adenylation half-reaction: carboxylate O⁻ attacks ATP ALPHA-P (acyl-adenylate),
distinct from the generic phosphoryl_transfer (Pγ / kinase) stub.
"""
import pytest

from evoliez.mechanism.spec import MechanismSpec
from evoliez.mechanism.templates import get_template


def test_registered_with_real_geometry_distinct_from_stub():
    t = get_template("adenylation_phosphoryl_transfer")
    assert t is not None
    labels = [g["label"] for g in t["default_geometry_terms"]]
    assert labels == ["nuc_O_to_alphaP", "inline_attack_Onuc_Pa_Oleaving"]
    # the generic phosphoryl_transfer (Pγ) stub must stay a stub — different chemistry
    assert get_template("phosphoryl_transfer")["default_geometry_terms"] == []
    assert "alpha" in t["notes"].lower() or "Pα" in t["notes"]


def test_mechanismspec_validates_and_emits_terms():
    spec = MechanismSpec.model_validate({
        "reaction": {"cls": "adenylation_phosphoryl_transfer"},
        "reaction_state": {"substrate_state": "3HP_carboxylate", "cofactor_state": "ATP_Mg",
                           "metal_state": "Mg2+_bridged", "conformational_state": "A_domain_closed"},
    })
    gts = spec.to_geometry_terms()
    by_label = {g.label: g for g in gts}
    assert "nuc_O_to_alphaP" in by_label
    ang = by_label["inline_attack_Onuc_Pa_Oleaving"]
    assert ang.kind == "angle" and ang.angle_min == 150.0 and ang.angle_max == 180.0
    dist = by_label["nuc_O_to_alphaP"]
    assert dist.kind == "distance" and dist.distance_max == 3.6


def test_missing_required_reaction_state_aborts():
    # metal_state is required for adenylation (Mg2+ state-defining) -> hard abort at load
    with pytest.raises(Exception):
        MechanismSpec.model_validate({
            "reaction": {"cls": "adenylation_phosphoryl_transfer"},
            "reaction_state": {"substrate_state": "3HP", "conformational_state": "closed"}})


def test_cofactor_state_field_accepts_non_redox_cofactor():
    # ATP (non-redox) uses the new cofactor_state field, distinct from cofactor_redox_state
    spec = MechanismSpec.model_validate({
        "reaction": {"cls": "adenylation_phosphoryl_transfer"},
        "reaction_state": {"substrate_state": "3HP_carboxylate", "cofactor_state": "ATP",
                           "metal_state": "Mg2+_bridged", "conformational_state": "A_domain_closed"},
    })
    assert spec.reaction_state.cofactor_state == "ATP"
    assert spec.reaction_state.cofactor_redox_state is None
