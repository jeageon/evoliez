"""Unit tests for the evidence-class library (v2 Phase E)."""
from evoliez.md.gate_stack import (
    ALTERNATIVE_POSE, CONFIRMED_COMPUTATIONAL, REJECTED,
)
from evoliez.ranking.evidence import (
    build_evidence_library, is_paper_grade, pareto_front, verdict_of,
)

_REF = {"design_ligand": {"status": "reference_like"}}


def test_verdict_embedded_then_computed():
    assert verdict_of({"gate_stack": {"verdict": "x"}}) == "x"        # embedded wins
    r = {"pose_gate": _REF, "nac_status": "valid", "nac_delta_vs_wt": 0.1,
         "rbfe_ddg_bind": -2.8, "rbfe_mode": "additive_x2"}
    assert verdict_of(r) == CONFIRMED_COMPUTATIONAL                   # computed on the fly


def test_paper_grade_requires_real_anchored():
    ok = {"validation_structure": "wt_anchored", "pose_gate": _REF,
          "gate_stack": {"verdict": "x"}}
    assert is_paper_grade(ok)
    assert not is_paper_grade({**ok, "validation_structure": "boltz"})
    assert not is_paper_grade({**ok, "pose_gate": {"design_ligand": {"status": "displaced"}}})
    assert not is_paper_grade({k: v for k, v in ok.items() if k != "gate_stack"})


def test_pareto_front_orthogonal_axes():
    a = {"candidate_id": "a", "nac_delta_vs_wt": 0.2, "md_instability": 0.0, "rbfe_ddg_bind": -3}
    b = {"candidate_id": "b", "nac_delta_vs_wt": 0.1, "md_instability": 0.0, "rbfe_ddg_bind": 0}
    c = {"candidate_id": "c", "nac_delta_vs_wt": 0.0, "md_instability": 0.0, "rbfe_ddg_bind": -5}
    axes = [("nac_delta_vs_wt", True), ("md_instability", False), ("rbfe_ddg_bind", False)]
    front = {r["candidate_id"] for r in pareto_front([a, b, c], axes)}
    assert "a" in front and "c" in front and "b" not in front        # b dominated by a


def test_library_groups_counts_and_paper_gate():
    recs = [{"candidate_id": "c1", "gate_stack": {"verdict": REJECTED}},
            {"candidate_id": "c2", "gate_stack": {"verdict": ALTERNATIVE_POSE}},
            {"candidate_id": "c3", "gate_stack": {"verdict": ALTERNATIVE_POSE}}]
    lib = build_evidence_library(recs)
    assert lib.counts.get(ALTERNATIVE_POSE) == 2 and lib.counts.get(REJECTED) == 1
    assert lib.paper_grade == []          # none anchored -> none paper-grade
    assert "c2" in lib.to_json()["classes"][ALTERNATIVE_POSE]
