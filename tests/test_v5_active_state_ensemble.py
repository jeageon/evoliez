"""ROADMAP_V5 Day-6 (E1 CONDITIONAL -> active-state ensemble): build the ReferenceEnsemble from REAL
measured MD geometry (the E1 explicit WT trajectory retained the co-substrate, retention 1.0) instead of
build_reference_ensemble_v0's hardcoded synthetic offsets. A candidate is then scored against a real
active-state ensemble — an ensemble-vs-ensemble comparison that is robust to the single-pose noise that
gave discriminates=false at 0.3 ns. Claim stays L0_uncalibrated (no wet-lab)."""
import pytest

from evoliez.guided.ensemble_builder import build_reference_ensemble_from_observations


def test_real_ensemble_from_measured_frames():
    obs = [(4.5 + 0.4 * (i % 5) / 5.0, 120.0 + 50.0 * (i % 7) / 7.0) for i in range(50)]
    e = build_reference_ensemble_from_observations("srcar", obs, source_label="e1_explicit_wt")
    assert len(e.conformers) == 50
    assert e.uncertainty.evidence_density == "measured_md"
    assert e.conformers[0].provenance["measured"] == "true"          # NOT a synthetic fixture
    assert e.claim_level == "L0_uncalibrated"                         # no wet-lab calibration
    assert e.uncertainty.geometry_variance >= 0.0
    assert e.insufficiency_reason is None


def test_small_frame_set_flags_insufficiency_not_crash():
    e = build_reference_ensemble_from_observations("srcar", [(4.7, 169.0), (4.9, 150.0), (5.1, 120.0)])
    assert len(e.conformers) == 3
    assert e.insufficiency_reason and "<8" in e.insufficiency_reason
    assert e.uncertainty.evidence_density == "limited"


def test_empty_observations_raise():
    with pytest.raises(ValueError, match="1 real frame"):
        build_reference_ensemble_from_observations("srcar", [])


def test_contact_and_strain_optional_default_clean():
    e = build_reference_ensemble_from_observations("srcar", [(4.7, 169.0, 0.9, 0.1)] * 8)
    c0 = e.conformers[0].geometry
    assert c0.distance_A == 4.7 and c0.angle_deg == 169.0
    assert c0.contact_score == 0.9 and c0.strain_penalty == 0.1
