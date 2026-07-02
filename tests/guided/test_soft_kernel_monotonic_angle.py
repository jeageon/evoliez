from evoliez.guided.soft_kernels import soft_reaction_geometry_score


def test_soft_kernel_monotonic_angle():
    common = dict(distance_A=3.25, contact_score=1.0, strain_penalty=0.0)
    good = soft_reaction_geometry_score(angle_deg=165.0, **common)
    poor = soft_reaction_geometry_score(angle_deg=80.0, **common)
    assert good > poor
