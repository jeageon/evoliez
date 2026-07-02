from evoliez.guided.feasibility import run_feasibility_gate


def test_guided_feasibility_requires_atom_mapping():
    result = run_feasibility_gate(differentiable_sampler_available=True, synthetic_gradient_ok=True)
    assert result.decision == "NO_GO"
    assert "validated atom mapping is required" in result.reasons
