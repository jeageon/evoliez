"""Regression tests for the anti-leakage / label-integrity guards (Fable safety review).

Three fail-open holes: (1) the `distance_to_` prefix defaulted cheap so a post-pose/MD distance
leaked in as a feature; (2) FORBIDDEN_HEAD_TOKENS missed activity synonyms; (3) the round-2 gate
denylisted one exact string, so any other claim_level ran AL on uncalibrated data.
"""
from __future__ import annotations

import pytest

from evoliez.experimental.active_learning import require_calibrated_for_round2
from evoliez.experimental.calibration import CalibrationResult
from evoliez.ml.evidence_prior import HeadSpec, assert_heads_are_priors
from evoliez.ml.learnability import assert_no_leakage, is_cheap


# --- distance_to_ prefix no longer blanket-cheap --------------------------------------
def test_static_distances_stay_cheap():
    assert is_cheap("distance_to_design_ligand")
    assert is_cheap("distance_to_nearest_catalytic_per_pos")


@pytest.mark.parametrize("field", [
    "distance_to_catalytic_after_md", "distance_to_ligand_post_dock",
    "distance_to_reactive_atom_nac", "distance_to_pocket_rmsd",
])
def test_post_pose_distances_are_blocked(field):
    assert not is_cheap(field)
    with pytest.raises(ValueError):
        assert_no_leakage([field])


# --- head-name activity synonyms ------------------------------------------------------
@pytest.mark.parametrize("name", [
    "P(turnover_number)", "P(vmax)", "P(v_max)", "P(productivity)", "P(reaction_rate)",
    "P(catalytic_conversion)", "P(yield)", "P(fitness)", "P(growth_fitness)",
    "P(specific_activity)", "P(selectivity)",
])
def test_activity_synonym_head_names_are_rejected(name):
    with pytest.raises(ValueError):
        assert_heads_are_priors([HeadSpec(name, "md_pass")])


def test_surrogate_head_names_are_allowed():
    assert_heads_are_priors([HeadSpec("P(structural_viable)", "md_pass"),
                             HeadSpec("P(reference_pose_accommodation)", "reference_like")])


# --- round-2 gate: allowlist, not one denylisted string -------------------------------
def test_round2_blocked_for_uncalibrated_and_unknown():
    with pytest.raises(ValueError):
        require_calibrated_for_round2(CalibrationResult(claim_level="L0_uncalibrated"))
    # an unrecognised claim_level can no longer even be CONSTRUCTED (Literal-constrained)
    with pytest.raises(Exception):
        CalibrationResult(claim_level="L1_prelim")
    with pytest.raises(Exception):
        CalibrationResult(claim_level="")


def test_round2_allowed_for_calibrated_levels():
    require_calibrated_for_round2(CalibrationResult(claim_level="L1_screened"))
    require_calibrated_for_round2(CalibrationResult(claim_level="L2_calibrated"))


@pytest.mark.parametrize("field", [
    "distance_to_catalytic_mean", "distance_to_ligand_avg", "distance_to_reactive_final",
    "distance_to_catalytic_production", "distance_to_site_ns",
])
def test_trajectory_averaged_distances_are_blocked(field):
    from evoliez.ml.learnability import is_cheap
    assert not is_cheap(field)


@pytest.mark.parametrize("name", [
    "P(catalysis)", "P(potency)", "P(efficacy)", "P(kobs)", "P(activation_energy)",
    "P(product_formation)", "P(gain_of_function)", "P(rate_constant)",
])
def test_more_activity_synonym_heads_rejected(name):
    with pytest.raises(ValueError):
        assert_heads_are_priors([HeadSpec(name, "md_pass")])
