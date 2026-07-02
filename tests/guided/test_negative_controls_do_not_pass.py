from evoliez.guided.feasibility import run_feasibility_gate
from evoliez.provenance.atom_map import ReactiveAtomMap


def test_negative_controls_do_not_pass():
    atom_map = ReactiveAtomMap(
        cofactor_atoms={"C4": 1},
        substrate_atoms={"H": 2},
        catalytic_residue_atoms={"D221": "OD1"},
        validated=True,
    )
    result = run_feasibility_gate(
        atom_map=atom_map,
        differentiable_sampler_available=True,
        synthetic_gradient_ok=True,
        negative_controls_passed=False,
    )
    assert result.decision == "NO_GO"
    assert "negative controls did not pass" in result.reasons
