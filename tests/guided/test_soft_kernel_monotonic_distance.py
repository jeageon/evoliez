from evoliez.guided.soft_kernels import soft_reaction_geometry_score


def test_soft_kernel_monotonic_distance():
    common = dict(angle_deg=165.0, contact_score=1.0, strain_penalty=0.0)
    near = soft_reaction_geometry_score(distance_A=3.25, **common)
    far = soft_reaction_geometry_score(distance_A=5.0, **common)
    assert near > far
