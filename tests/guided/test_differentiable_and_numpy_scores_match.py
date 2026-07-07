from evoliez.guided.losses.catalytic_geom import catalytic_geometry_loss
from evoliez.guided.soft_kernels import finite_difference_distance_gradient, soft_reaction_geometry_score


def test_differentiable_and_numpy_scores_match():
    kwargs = dict(distance_A=3.25, angle_deg=165.0, contact_score=1.0, strain_penalty=0.0)
    score = soft_reaction_geometry_score(**kwargs)
    assert abs((1.0 - catalytic_geometry_loss(**kwargs)) - score) < 1e-9
    assert abs(finite_difference_distance_gradient(3.0, angle_deg=165.0, contact_score=1.0, strain_penalty=0.0)) > 0
