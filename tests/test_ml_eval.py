"""v2 Phase F — ML re-evaluation on functional-state preservation."""
from evoliez.ranking.ml_eval import _auc, is_functional, ml_functional_eval

_REF = {"design_ligand": {"status": "reference_like"}}
_DISP = {"design_ligand": {"status": "displaced"}}


def test_is_functional():
    assert is_functional({"pose_gate": _REF, "nac_delta_vs_wt": 0.1})
    assert is_functional({"pose_gate": _REF})                       # pose-only (no NAC) ok
    assert not is_functional({"pose_gate": _REF, "nac_delta_vs_wt": -0.2})  # lost reactivity
    assert not is_functional({"pose_gate": _DISP})                  # not reference-like


def test_auc():
    assert _auc([(0.9, True), (0.1, False)]) == 1.0
    assert _auc([(0.1, True), (0.9, False)]) == 0.0
    assert _auc([(0.5, True)]) is None                             # no negatives


def test_eval_and_control_false_negative():
    recs = [
        {"candidate_id": "a", "pose_gate": _REF, "nac_delta_vs_wt": 0.1, "passed": True},
        {"candidate_id": "b", "pose_gate": _REF, "nac_delta_vs_wt": 0.05, "passed": True},
        {"candidate_id": "c", "pose_gate": _DISP, "passed": False},
    ]
    ml = {"a": 0.9, "b": 0.2, "c": 0.8}        # b is functional but LOW ml
    lane = {"a": "ml_high", "b": "low_ml_control", "c": "ml_high"}
    out = ml_functional_eval(recs, ml_score_by_id=ml, lane_by_id=lane)
    assert out["n_functional"] == 2
    assert out["functional_in_control"] == 1   # b: a winner the ML cut would have dropped
    assert out["ml_false_negative"] is True
    assert out["auc_ml_to_functional"] is not None
