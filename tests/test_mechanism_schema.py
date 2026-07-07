"""V3-0 schema contract freeze (ROADMAP_V3 §9).

Every v3 card constructs from a valid example AND rejects malformed input (the strict
schema layer ClaimGuard relies on). Pure pydantic — runs in the light env.
"""
import pytest
from pydantic import ValidationError

from evoliez.mechanism import (
    ActiveStateReferenceEnsemble, BenchmarkCard, EvidenceCard, GeometryCalibration,
    MechanismSpec, ReferenceConfidenceCard, ReferenceMember, SimulationSetupCard,
    WetLabLabelSpec, list_template_keys,
)
from evoliez.mechanism.cards import EvidenceAxis, LigandParam, TermDistribution
from evoliez.mechanism.spec import ReactionInfo, ReactionState
from evoliez.mechanism import vocab


# --- MechanismSpec + templates -------------------------------------------------------
def _fdh_spec(**rs):
    base = dict(cofactor_redox_state="NADP+", conformational_state="closed_ternary_complex")
    base.update(rs)
    return MechanismSpec(
        mechanism_spec_id="fdh_hydride_transfer_v1",
        reaction=ReactionInfo(**{"class": "hydride_transfer"}),
        reaction_state=ReactionState(**base),
    )


def test_template_keys_present():
    keys = list_template_keys()
    assert "hydride_transfer" in keys
    assert "nucleophilic_acyl_substitution" in keys


def test_hydride_template_fills_default_geometry():
    spec = _fdh_spec()
    # the FDH NAC geometry comes from the template, not FDH-specific code
    labels = [t.label for t in spec.geometry_terms]
    assert "donor_acceptor" in labels and "hydride_axis" in labels
    # convertible to runtime terms is exercised separately (needs md env)


def test_unknown_reaction_class_aborts():
    with pytest.raises(ValidationError):
        MechanismSpec(reaction=ReactionInfo(**{"class": "teleportation"}))


def test_missing_required_reaction_state_aborts():
    # hydride_transfer requires cofactor_redox_state + conformational_state
    with pytest.raises(ValidationError):
        MechanismSpec(
            reaction=ReactionInfo(**{"class": "hydride_transfer"}),
            reaction_state=ReactionState(conformational_state="closed_ternary_complex"),
        )


def test_geometry_calibration_caps_claim():
    spec = _fdh_spec()
    spec_g5 = MechanismSpec(
        reaction=ReactionInfo(**{"class": "hydride_transfer"}),
        reaction_state=ReactionState(cofactor_redox_state="NADP+",
                                     conformational_state="closed"),
        geometry_calibration=GeometryCalibration(source_tier="G4"),
    )
    assert spec.claim_ceiling == vocab.STRONG_SCREENING  # default G2
    assert spec_g5.claim_ceiling == vocab.HYPOTHESIS_GRADE


def test_bad_geometry_tier_rejected():
    with pytest.raises(ValidationError):
        GeometryCalibration(source_tier="G9")


def test_legacy_reactive_geometry_lift():
    from evoliez.config import ReactiveGeometryConfig
    from evoliez.mechanism.spec import mechanism_from_reactive_geometry
    rg = ReactiveGeometryConfig(enabled=True, donor_smarts="[CX3H1](=O)[O-]",
                                acceptor_smarts="[cH1]([cH0]C(=O)[NX3])[cH1]",
                                distance_max=3.5, angle_min=150.0, label="fdh")
    spec = mechanism_from_reactive_geometry(
        rg, ReactionState(cofactor_redox_state="NADP+", conformational_state="closed"))
    assert [t.label for t in spec.geometry_terms] == ["fdh_donor_acceptor", "fdh_axis"]
    # the lift still hard-gates a missing reaction-state field
    with pytest.raises(ValidationError):
        mechanism_from_reactive_geometry(rg, ReactionState(cofactor_redox_state="NADP+"))


def test_typo_field_rejected():
    with pytest.raises(ValidationError):
        ReactionState(redox_state="NADP+")  # field is cofactor_redox_state


# --- ReferenceConfidenceCard ---------------------------------------------------------
def test_reference_card_derives_claim_strength():
    a = ReferenceConfidenceCard(tier="A", substrate_is_real=True,
                                known_active_controls_available=True,
                                functional_atom_mapping_verified=True)
    assert a.claim_strength == vocab.STRONG_SCREENING
    # tier C + analog + no controls -> uncalibrated (weakest)
    c = ReferenceConfidenceCard(tier="C", substrate_is_real=False)
    assert c.claim_strength == vocab.UNCALIBRATED


def test_reference_bad_tier_rejected():
    with pytest.raises(ValidationError):
        ReferenceConfidenceCard(tier="Z")


# --- ActiveStateReferenceEnsemble ----------------------------------------------------
def test_ensemble_ceiling_from_best_member_then_disagreement():
    ens = ActiveStateReferenceEnsemble(
        ensemble_id="e1",
        references=[
            ReferenceMember(reference_id="a1", tier="A"),
            ReferenceMember(reference_id="c1", tier="C"),
        ],
        distance_terms={"donor_acceptor": TermDistribution(median=3.0, iqr=0.4)},
    )
    # best member is tier A -> strong, low disagreement keeps it
    assert ens.claim_ceiling == vocab.STRONG_SCREENING

    ens_noisy = ActiveStateReferenceEnsemble(
        ensemble_id="e2",
        references=[ReferenceMember(reference_id="a1", tier="A")],
        disagreement={"geometry_variance": 0.3},
    )
    assert ens_noisy.claim_ceiling == vocab.HYPOTHESIS_GRADE


def test_ensemble_requires_a_reference():
    with pytest.raises(ValidationError):
        ActiveStateReferenceEnsemble(ensemble_id="empty", references=[])


# --- EvidenceCard --------------------------------------------------------------------
def test_evidence_card_score_and_confidence_independent():
    card = EvidenceCard(
        variant_id="A123G",
        reaction_geometry_accommodation=EvidenceAxis(
            score=0.82, confidence="low",
            reason="Tier-C reference, no controls"),
        uncertainty=EvidenceAxis(score=0.72, confidence="high"),
    )
    # high score yet low confidence is representable
    assert card.reaction_geometry_accommodation.score == 0.82
    assert card.reaction_geometry_accommodation.confidence == "low"


def test_evidence_bad_confidence_rejected():
    with pytest.raises(ValidationError):
        EvidenceAxis(score=0.5, confidence="pretty_sure")


# --- SimulationSetupCard -------------------------------------------------------------
def test_sim_card_forbids_angle_restraints():
    with pytest.raises(ValidationError):
        SimulationSetupCard(angle_restraints=True)


def test_sim_card_param_penalty():
    card = SimulationSetupCard(ligand_parameters={
        "NADP": LigandParam(redox_state="NADP+", parameter_validated=True),
        "formate": LigandParam(protonation_state="anion", parameter_validated=False),
    })
    assert card.md_confidence_penalty() is True


# --- BenchmarkCard + WetLabLabelSpec -------------------------------------------------
def test_benchmark_card():
    card = BenchmarkCard(
        target_id="TEM1", mechanism_template="nucleophilic_acyl_substitution",
        label_primary="antibiotic_fitness", is_direct_kcat=False,
        prohibited_claims=["direct kcat prediction"])
    assert "direct kcat prediction" in card.prohibited_claims


def test_wetlab_replicated_gate():
    spec = WetLabLabelSpec(assay_id="fdh_340", endpoint_primary="initial_rate",
                           replicates_biological=3, wt_replicates=6)
    assert spec.replicated() is True
    weak = WetLabLabelSpec(assay_id="x", endpoint_primary="rate",
                           replicates_biological=1, wt_replicates=0)
    assert weak.replicated() is False


# --- vocab fail-safe combinator ------------------------------------------------------
def test_weakest_claim_is_fail_safe():
    assert vocab.weakest_claim("strong_screening", "hypothesis_grade") == "hypothesis_grade"
    assert vocab.weakest_claim() == vocab.UNCALIBRATED
    assert vocab.weakest_claim("nonsense_key") == vocab.UNCALIBRATED  # unknown -> weakest
