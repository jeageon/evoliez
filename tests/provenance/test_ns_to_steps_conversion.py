from evoliez.utils.trajectory_qc import ns_to_steps, simulated_time_from_steps


def test_ns_to_steps_conversion():
    assert ns_to_steps(2.0, 2.0) == 1_000_000
    assert simulated_time_from_steps(1_000_000, 2.0) == 2.0
