"""Unit tests for the gate-stack verdict — encodes the S340G lesson."""
from evoliez.md.gate_stack import (
    ALTERNATIVE_POSE, CANDIDATE_IMPROVED, CANDIDATE_NO_GAIN,
    CONFIRMED_COMPUTATIONAL, REJECTED, evaluate_gate_stack,
)

_REF = {"design_ligand": {"status": "reference_like"}}


def test_rejected_unstable():
    r = evaluate_gate_stack({"md_instability": 1.0, "passed": False})
    assert r.verdict == REJECTED and r.deciding_gate == "structural_stability"


def test_rejected_high_ddg():
    assert evaluate_gate_stack({"ddg_fold": 5.0}).verdict == REJECTED


def test_alternative_pose_boltz_s340g():
    # fresh-Boltz S340G: NADP 8 A off WT (displaced) even though its NAC looked +0.12
    r = evaluate_gate_stack({"pose_gate": {"design_ligand": {"status": "displaced"}},
                             "nac_delta_vs_wt": 0.12})
    assert r.verdict == ALTERNATIVE_POSE and r.deciding_gate == "design_pose"


def test_candidate_no_gain_anchored_s340g():
    # WT-anchored S340G: WT-like pose but ΔNAC 0 -> NOT a catalytic lead
    r = evaluate_gate_stack({"pose_gate": _REF,
                             "nac_status": "valid_restrained_retention_screen",
                             "nac_delta_vs_wt": 0.0})
    assert r.verdict == CANDIDATE_NO_GAIN and r.deciding_gate == "functional_geometry"


def test_candidate_improved_without_rbfe():
    r = evaluate_gate_stack({"pose_gate": _REF, "nac_status": "valid",
                             "nac_delta_vs_wt": 0.1})
    assert r.verdict == CANDIDATE_IMPROVED


def test_confirmed_with_favourable_rbfe():
    r = evaluate_gate_stack({"pose_gate": _REF, "nac_status": "valid",
                             "nac_delta_vs_wt": 0.1, "rbfe_ddg_bind": -2.8,
                             "rbfe_mode": "additive_x2"})
    assert r.verdict == CONFIRMED_COMPUTATIONAL and r.gates["energetic"] == "pass"


def test_rbfe_nan_stays_candidate_improved():
    r = evaluate_gate_stack({"pose_gate": _REF, "nac_status": "valid",
                             "nac_delta_vs_wt": 0.1, "rbfe_ddg_bind": None,
                             "rbfe_mode": "failed_softcore_ti_nan"})
    assert r.verdict == CANDIDATE_IMPROVED


def test_experimental_always_pending():
    r = evaluate_gate_stack({"pose_gate": _REF, "nac_status": "valid",
                             "nac_delta_vs_wt": 0.1, "rbfe_ddg_bind": -2.8})
    assert r.experimental == "pending"
