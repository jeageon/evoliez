"""V4-3: per-term geometry sensitivity (ROADMAP_V4 §7.4 build target)."""

from evoliez.guided.geometry_sensitivity import dominant_term, geometry_sensitivity


def test_sensitivity_has_all_terms_and_finite():
    sens = geometry_sensitivity(
        distance_A=3.4, angle_deg=150.0, contact_score=0.9, strain_penalty=0.15
    )
    assert set(sens) == {"distance_A", "angle_deg", "contact_score", "strain_penalty"}
    for v in sens.values():
        assert isinstance(v, float)


def test_sensitivity_signs_are_physically_sensible():
    # operating point chosen so the score is healthy (~0.6) and every partial is measurable
    sens = geometry_sensitivity(
        distance_A=3.4, angle_deg=150.0, contact_score=0.9, strain_penalty=0.15
    )
    assert sens["distance_A"] < 0      # past the optimal distance -> further hurts
    assert sens["angle_deg"] > 0       # below the optimal angle -> larger helps
    assert sens["contact_score"] > 0   # more contact retention -> better
    assert sens["strain_penalty"] < 0  # more strain -> worse


def test_dominant_term_is_an_input():
    sens = geometry_sensitivity(
        distance_A=3.4, angle_deg=150.0, contact_score=0.9, strain_penalty=0.15
    )
    assert dominant_term(sens) in sens
