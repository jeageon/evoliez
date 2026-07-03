"""ROADMAP_V3 B7: the 3rd benchmark mechanism (glycosidase) now carries real default
geometry, so a non-FDH benchmark contributes geometry evidence, not just a load test.
"""
from __future__ import annotations

from evoliez.mechanism.spec import MechanismSpec, ReactionInfo, ReactionState
from evoliez.mechanism.templates import TEMPLATES, get_template


def test_at_least_three_templates_have_geometry():
    with_geom = [k for k, t in TEMPLATES.items() if t["default_geometry_terms"]]
    assert set(with_geom) >= {
        "hydride_transfer", "nucleophilic_acyl_substitution",
        "glycosidic_bond_cleavage"}


def test_glycosidic_template_geometry_shape():
    t = get_template("glycosidic_bond_cleavage")
    assert len(t["default_geometry_terms"]) == 2
    kinds = [g["kind"] for g in t["default_geometry_terms"]]
    assert kinds == ["distance", "angle"]


def test_glycosidic_mechanism_builds_and_converts():
    spec = MechanismSpec(
        reaction=ReactionInfo(**{"class": "glycosidic_bond_cleavage"}),
        reaction_state=ReactionState(protonation_model="acid_base_pair",
                                     conformational_state="michaelis_complex"),
    )
    assert len(spec.geometry_terms) == 2
    terms = spec.to_geometry_terms()  # runtime GeometryTerm objects (protein + SMARTS mix)
    assert terms[0].a_residue == "ASP" and terms[0].a_atom == "OD2"
    assert terms[0].b_smarts and terms[1].kind == "angle"
