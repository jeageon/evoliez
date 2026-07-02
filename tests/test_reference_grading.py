"""V3-3: grade_reference + build_reference_ensemble (ROADMAP_V3 D2/D6)."""
from evoliez.reference_state import (
    ReferenceGates, build_reference_ensemble, grade_reference,
)
from evoliez.mechanism import vocab


def _gates(passed=True, **checks):
    base = dict(catalytic_residues_present=True, reference_atom_map_verified=True,
                charge_protonation_provenance=True)
    base.update(checks)
    return ReferenceGates(passed=passed, checks=base, reasons=[])


def test_grade_strong_reference():
    card = grade_reference(_gates(), tier="A", substrate_is_real=True,
                           known_active_controls_available=True)
    assert card.functional_atom_mapping_verified is True
    assert card.claim_strength == vocab.STRONG_SCREENING


def test_grade_weak_reference_downgrades():
    # tier C + analog + no controls -> uncalibrated, regardless of good gates
    card = grade_reference(_gates(), tier="C", substrate_is_real=False,
                           analog_identity="formate_proxy")
    assert card.claim_strength == vocab.UNCALIBRATED


def test_grade_reads_gate_checks():
    card = grade_reference(_gates(reference_atom_map_verified=False),
                           tier="B", known_active_controls_available=True)
    assert card.functional_atom_mapping_verified is False
    # unverified atom mapping caps at hypothesis_grade
    assert card.claim_strength == vocab.HYPOTHESIS_GRADE


def test_build_ensemble_distribution_and_disagreement():
    members = [
        {"reference_id": "ref_A1", "tier": "A", "weight": 1.0},
        {"reference_id": "ref_B1", "tier": "B", "weight": 0.7},
        {"reference_id": "ref_C1", "tier": "C", "weight": 0.4},
    ]
    ens = build_reference_ensemble(
        "fdh_active_v1", members,
        distance_measurements={"donor_acceptor": {"ref_A1": 3.0, "ref_B1": 3.1, "ref_C1": 3.2}},
    )
    td = ens.distance_terms["donor_acceptor"]
    assert abs(td.median - 3.1) < 1e-9
    assert td.iqr > 0
    # tight agreement -> low disagreement -> ceiling stays strong (best member tier A)
    assert ens.disagreement.geometry_variance < 0.1
    assert ens.claim_ceiling == vocab.STRONG_SCREENING


def test_build_ensemble_high_disagreement_caps_claim():
    members = [{"reference_id": "a1", "tier": "A"}]
    ens = build_reference_ensemble(
        "noisy", members,
        distance_measurements={"donor_acceptor": {"a1": 3.0, "x": 3.0, "y": 9.0}},
    )
    # wide spread -> high coefficient of dispersion -> claim ceiling drops below strong
    assert ens.disagreement.geometry_variance >= 0.1
    assert ens.claim_ceiling != vocab.STRONG_SCREENING
