"""Evidence-class library (ROADMAP_V2 Phase E) — replace 'one scalar rank' with an evidence
stack: group candidates by their gate-stack verdict + a Pareto front, and mark which are
*paper-grade* (carry a real anchored validation). The final claim is an evidence class, not a
position in a sorted list (which silently mixes binding, stability, and catalysis signals).

Works on s10 md_candidates records (dicts). If a record has no embedded `gate_stack` (e.g. a
pre-v2 run) the verdict is computed on the fly via `md.gate_stack.evaluate_gate_stack`, so the
library is reproducible from any provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from evoliez.md.gate_stack import (
    ALTERNATIVE_POSE, CANDIDATE_IMPROVED, CANDIDATE_NO_GAIN,
    CONFIRMED_COMPUTATIONAL, REJECTED, evaluate_gate_stack,
)

# evidence classes, strongest-first
EVIDENCE_ORDER: Tuple[str, ...] = (
    CONFIRMED_COMPUTATIONAL, CANDIDATE_IMPROVED, CANDIDATE_NO_GAIN,
    ALTERNATIVE_POSE, REJECTED,
)


def verdict_of(rec: dict) -> str:
    gs = rec.get("gate_stack")
    if isinstance(gs, dict) and gs.get("verdict"):
        return gs["verdict"]
    return evaluate_gate_stack(rec).verdict


def is_paper_grade(rec: dict) -> bool:
    """Paper-grade requires a REAL anchored validation (Phase E gate): the candidate was
    validated WT-anchored, its design pose stayed reference_like, and a gate-stack verdict
    exists. A pre-v2 / Boltz-pose record can never be paper-grade."""
    if rec.get("validation_structure") != "wt_anchored":
        return False
    pg = (rec.get("pose_gate") or {}).get("design_ligand") or {}
    if pg.get("status") != "reference_like":
        return False
    return isinstance(rec.get("gate_stack"), dict)


def _val(rec: dict, key: str, default: float = 0.0) -> float:
    v = rec.get(key)
    return float(v) if isinstance(v, (int, float)) else default


def pareto_front(recs: Sequence[dict], axes: Sequence[Tuple[str, bool]]) -> List[dict]:
    """Non-dominated set. axes = [(key, maximize)]. A record dominates another if it is no
    worse on every axis and strictly better on at least one."""
    def better_eq(a, b, k, mx):
        return (_val(a, k) >= _val(b, k)) if mx else (_val(a, k) <= _val(b, k))

    def strictly(a, b, k, mx):
        return (_val(a, k) > _val(b, k)) if mx else (_val(a, k) < _val(b, k))

    front: List[dict] = []
    for r in recs:
        dominated = any(
            r is not o
            and all(better_eq(o, r, k, mx) for k, mx in axes)
            and any(strictly(o, r, k, mx) for k, mx in axes)
            for o in recs
        )
        if not dominated:
            front.append(r)
    return front


@dataclass
class EvidenceLibrary:
    classes: Dict[str, List[dict]]   # verdict -> records (strongest-first keys)
    paper_grade: List[dict]          # records passing the anchored paper-grade gate
    pareto: List[dict]               # non-dominated on (ΔNAC↑, instability↓, ΔΔG_bind↓)
    counts: Dict[str, int]

    def to_json(self) -> dict:
        def _ids(rs):
            return [r.get("candidate_id") for r in rs]
        return {
            "counts": self.counts,
            "paper_grade": _ids(self.paper_grade),
            "pareto": _ids(self.pareto),
            "classes": {v: _ids(rs) for v, rs in self.classes.items() if rs},
        }


def build_evidence_library(records: List[dict]) -> EvidenceLibrary:
    classes: Dict[str, List[dict]] = {v: [] for v in EVIDENCE_ORDER}
    for r in records:
        classes.setdefault(verdict_of(r), []).append(r)
    paper = [r for r in records if is_paper_grade(r)]
    # Pareto over the orthogonal evidence axes (never collapse them into one scalar):
    #   functional geometry (ΔNAC, maximize) · structural stability (md_instability, minimize)
    #   · energetic binding (ΔΔG_bind, minimize). Paper-grade set if any, else all.
    pareto = pareto_front(
        paper or records,
        [("nac_delta_vs_wt", True), ("md_instability", False), ("rbfe_ddg_bind", False)],
    )
    counts = {v: len(c) for v, c in classes.items() if c}
    return EvidenceLibrary(classes=classes, paper_grade=paper, pareto=pareto, counts=counts)
