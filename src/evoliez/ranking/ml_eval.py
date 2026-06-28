"""ML re-evaluation on FUNCTIONAL-STATE preservation (ROADMAP_V2 Phase F).

ML enriches binding-validity, NOT catalysis (the FDH run: AUC ml->MD-pass 0.83 vs ml->NAC
0.36). The v2 question is whether ml_score predicts a FUNCTIONALLY-PRESERVING mutant — an
anchored reference-like pose with a non-negative ΔNAC — and how many functional mutants the ML
cut would have DROPPED (the false-negative rate, read off the low-ML control lane). This is the
measurement framework; the actual retrain on anchored features runs on a real anchored run's
provenance (run-gated). Pure / unit-testable on s10 md_candidates records.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def _auc(scored: Sequence) -> Optional[float]:
    """ROC-AUC via Mann-Whitney (ties = 0.5). scored = [(score, label_bool)]."""
    pos = [s for s, y in scored if y]
    neg = [s for s, y in scored if not y]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return round(wins / (len(pos) * len(neg)), 3)


def is_functional(rec: dict) -> bool:
    """A functionally-preserving mutant: anchored design pose stayed reference_like AND ΔNAC is
    non-negative (didn't lose reactivity). Absent ΔNAC counts as non-negative (pose-only)."""
    pg = (rec.get("pose_gate") or {}).get("design_ligand") or {}
    if pg.get("status") != "reference_like":
        return False
    dnac = rec.get("nac_delta_vs_wt")
    return dnac is None or (isinstance(dnac, (int, float)) and dnac >= 0)


def is_md_pass(rec: dict) -> bool:
    return bool(rec.get("passed"))


def ml_functional_eval(records: List[dict],
                       ml_score_by_id: Optional[Dict[str, float]] = None,
                       lane_by_id: Optional[Dict[str, str]] = None) -> dict:
    """Compare ml_score's enrichment of FUNCTIONAL preservation vs mere MD-pass, and count the
    functional mutants the ML cut would have dropped (low-ML control lane)."""
    def ml(rec):
        if ml_score_by_id is not None:
            return ml_score_by_id.get(rec.get("candidate_id"))
        return rec.get("ml_score")

    scored_func = [(ml(r), is_functional(r)) for r in records
                   if isinstance(ml(r), (int, float))]
    scored_pass = [(ml(r), is_md_pass(r)) for r in records
                   if isinstance(ml(r), (int, float))]
    out: Dict[str, object] = {
        "n": len(records),
        "n_functional": sum(1 for r in records if is_functional(r)),
        "auc_ml_to_functional": _auc(scored_func),
        "auc_ml_to_mdpass": _auc(scored_pass),
    }
    if lane_by_id:
        ctrl = [r for r in records
                if lane_by_id.get(r.get("candidate_id")) == "low_ml_control"]
        fn = sum(1 for r in ctrl if is_functional(r))
        out["n_low_ml_control"] = len(ctrl)
        out["functional_in_control"] = fn      # >0 => the ML hard cut WOULD have dropped a winner
        out["ml_false_negative"] = fn > 0
    return out
