"""ROADMAP_V5 Phase V5-C (generality proof) — the mechanism framework is CONFIG-ONLY across
every real-geometry template: declaring a reaction.class selects the template, emits its
geometry terms, and hard-gates its required reaction_state — no per-mechanism code.

This is the enzyme-agnostic contract (FDH hydride · CAR adenylation · TEM-1 acyl · glycosidase)
proven without needing per-target run data. Mg²⁺ MD, the CAR rerun, and TEM-1/glycosidase
end-to-end smokes remain server/data-gated.
"""
import pytest

from evoliez.mechanism.spec import MechanismSpec
from evoliez.mechanism.templates import get_template

# (reaction.class, a reaction_state that satisfies its required fields, expected geometry labels)
REAL_TEMPLATES = [
    ("hydride_transfer",
     {"cofactor_redox_state": "NADPH", "conformational_state": "closed_ternary"},
     ["donor_acceptor", "hydride_axis"]),
    ("nucleophilic_acyl_substitution",
     {"protonation_model": "pH7", "conformational_state": "acyl_enzyme"},
     ["nucleophile_carbonyl", "burgi_dunitz"]),
    ("glycosidic_bond_cleavage",
     {"protonation_model": "pH5", "conformational_state": "michaelis"},
     ["nucleophile_anomeric_C", "anomeric_inline_attack"]),
    ("adenylation_phosphoryl_transfer",
     {"substrate_state": "carboxylate", "metal_state": "Mg2+_bridged",
      "conformational_state": "A_domain_closed"},
     ["nuc_O_to_alphaP", "inline_attack_Onuc_Pa_Oleaving"]),
]


@pytest.mark.parametrize("cls,state,labels", REAL_TEMPLATES)
def test_template_is_config_declarable_and_emits_geometry(cls, state, labels):
    spec = MechanismSpec.model_validate(
        {"reaction": {"class": cls}, "reaction_state": state})
    assert spec.reaction.cls == cls
    got = [g.label for g in spec.to_geometry_terms()]
    assert got == labels, f"{cls}: {got}"


@pytest.mark.parametrize("cls,state,labels", REAL_TEMPLATES)
def test_missing_required_reaction_state_aborts_for_every_template(cls, state, labels):
    # drop one required field -> MechanismSpec must refuse to run (hard reference-gate)
    required = get_template(cls)["required_reaction_state"]
    for drop in required:
        partial = {k: v for k, v in state.items() if k != drop}
        with pytest.raises(Exception):
            MechanismSpec.model_validate(
                {"reaction": {"class": cls}, "reaction_state": partial})


def test_all_four_share_one_geometry_term_schema():
    # every emitted term is the SAME GeometryTerm type with kind in {distance, angle}
    for cls, state, _ in REAL_TEMPLATES:
        spec = MechanismSpec.model_validate(
            {"reaction": {"class": cls}, "reaction_state": state})
        for g in spec.to_geometry_terms():
            assert g.kind in ("distance", "angle")
            assert hasattr(g, "label") and hasattr(g, "weight")
