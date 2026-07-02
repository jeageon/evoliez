"""V3-5: EvidenceCard builder (ROADMAP_V3 D4)."""
from evoliez.ranking.evidence_card import build_evidence_card
from evoliez.mechanism.cards import ReferenceConfidenceCard, SimulationSetupCard, LigandParam


def _rec(**kw):
    base = dict(candidate_id="A123G")
    base.update(kw)
    return base


def test_axes_populated_from_provenance():
    rec = _rec(stability_score=1.5, conservation=0.2, msa_permissiveness=0.7,
               hbond_occupancy=0.8, catalytic_distance_mean=3.2, nac_occupancy=0.6,
               docking_uncertainty=0.3)
    card = build_evidence_card(rec)
    assert card.variant_id == "A123G"
    assert card.structural_viability.score > 0.5
    assert card.reaction_geometry_accommodation.score > 0.0
    assert "nac_occupancy" in card.reaction_geometry_accommodation.evidence
    # uncertainty axis present
    assert card.uncertainty.score >= 0.0


def test_score_high_confidence_low_is_representable():
    # high geometry score but Tier-C reference + sparse WT geometry -> low confidence
    rec = _rec(nac_occupancy=0.9)
    weak_ref = ReferenceConfidenceCard(tier="C", substrate_is_real=False)
    card = build_evidence_card(rec, reference_card=weak_ref, wt_geometry_sparse=True)
    axis = card.reaction_geometry_accommodation
    assert axis.score >= 0.8
    assert axis.confidence in ("low", "low_to_medium")
    assert "diagnostic" in axis.reason


def test_strong_reference_raises_confidence():
    rec = _rec(nac_occupancy=0.9, hbond_occupancy=0.9, stability_score=1.0)
    strong = ReferenceConfidenceCard(tier="A", substrate_is_real=True,
                                     known_active_controls_available=True,
                                     functional_atom_mapping_verified=True)
    card = build_evidence_card(rec, reference_card=strong)
    assert card.reaction_geometry_accommodation.confidence in ("medium", "high", "low_to_medium")


def test_sim_param_penalty_lowers_md_confidence():
    rec = _rec(nac_occupancy=0.8, hbond_occupancy=0.8)
    sim_bad = SimulationSetupCard(ligand_parameters={
        "NADP": LigandParam(parameter_validated=False)})
    sim_ok = SimulationSetupCard(ligand_parameters={
        "NADP": LigandParam(parameter_validated=True)})
    c_bad = build_evidence_card(rec, sim_card=sim_bad)
    c_ok = build_evidence_card(rec, sim_card=sim_ok)
    # md-parameterization confidence component is lower in the unvalidated case
    assert c_bad.confidence_model.components["md_parameterization"] < \
        c_ok.confidence_model.components["md_parameterization"]


def test_delta_nac_drives_reaction_geometry_over_absolute_occupancy():
    # the real fdh_5track case: lead has small ABSOLUTE occupancy (0.12) but a POSITIVE
    # ΔNAC vs WT — it must out-score a neutral candidate with the SAME absolute occupancy.
    lead = build_evidence_card(_rec(nac_occupancy=0.12, nac_delta_vs_wt=0.12))
    neutral = build_evidence_card(_rec(nac_occupancy=0.12, nac_delta_vs_wt=0.0))
    worse = build_evidence_card(_rec(nac_occupancy=0.12, nac_delta_vs_wt=-0.1))
    assert lead.reaction_geometry_accommodation.score >= 0.6      # surfaced
    assert lead.reaction_geometry_accommodation.score > neutral.reaction_geometry_accommodation.score
    assert neutral.reaction_geometry_accommodation.score > worse.reaction_geometry_accommodation.score
    assert abs(neutral.reaction_geometry_accommodation.score - 0.5) < 0.12  # neutral ~ 0.5


def test_geometry_penalty_does_not_fabricate_positive_evidence():
    # a candidate with NO NAC data but a (contaminated) Boltz-pose penalty must NOT be
    # scored as geometry-supported — it has 0.0 score and the penalty only cuts confidence.
    no_nac = build_evidence_card(_rec(catalytic_geometry_penalty=1.4))
    assert no_nac.reaction_geometry_accommodation.score == 0.0
    # a real ΔNAC gainer outscores it even when the gainer also carries a strain penalty
    lead = build_evidence_card(_rec(nac_occupancy=0.12, nac_delta_vs_wt=0.12,
                                    catalytic_geometry_penalty=1.07))
    assert lead.reaction_geometry_accommodation.score > no_nac.reaction_geometry_accommodation.score
    assert "boltz_pose_strain" in lead.reaction_geometry_accommodation.evidence


def test_missing_fields_are_honest_low_confidence_not_crash():
    card = build_evidence_card(_rec())  # almost nothing provided
    assert card.structural_viability.score == 0.0
    assert card.reaction_geometry_accommodation.score == 0.0
    # no signals -> low confidence, no exception
    assert card.reaction_geometry_accommodation.confidence in ("low", "low_to_medium")
