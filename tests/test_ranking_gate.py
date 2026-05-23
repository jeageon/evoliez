"""P0a regression: ranking gate must demote Reject / invalid pose /
failed MD candidates out of top recommendations.

## Production incident
PseFDH run: top-ranked S335T had `evidence_class=Reject` AND
`pose_validity_status=invalid`. A clearly broken signal landing at
rank 1 would mislead every downstream consumer (paper figures, wet-lab
prioritisation, retrospective benchmark recall@K).

## Fix scope (this file pins it against future regressions)
- `s11_final_ranking` computes per-candidate `is_blocked` + `block_reason`
  AFTER evidence_class assignment.
- Accepted candidates occupy rank 1..M; blocked occupy M+1..N.
- `focused_library.csv` writer hard-filters blocked.
- `final_candidates.csv` has new `is_blocked` + `block_reason` columns
  so the user can grep / filter without re-running.
- `ctx.persist_meta("top_candidate", ...)` reflects top ACCEPTED (not
  raw top-scoring).
"""

from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from evoliez.types import Candidate, Mutation


def _mk_cand(cid: str, score: float, *, evidence: str = "Promising",
             pose_valid: str = "valid", md_status: str = "ok",
             generator: str = "chemistry_rules") -> Candidate:
    c = Candidate(
        candidate_id=cid,
        mutations=[Mutation(position=1, wt="A", mut="V")],
        generator=generator,
        scores={
            "final_score": score,
            "ml_score": 0.5,
            "evidence_class": evidence,
            "pose_validity_status": pose_valid,
            "md_status": md_status,
        },
        details={
            "evidence_class": evidence,
            "pose_validity_status": pose_valid,
        },
    )
    return c


# --------------------------------------------------------------------- #
# 1. Source guards
# --------------------------------------------------------------------- #
def test_s11_run_defines_blocked_helper():
    """The `_blocked_with_reason` helper must exist in s11.run and check
    all three hard-gate conditions (evidence_class, pose_validity,
    md_status)."""
    from evoliez.stages import s11_final_ranking
    src = inspect.getsource(s11_final_ranking.FinalRankingStage.run)
    assert "_blocked_with_reason" in src
    assert "evidence_class" in src
    assert "pose_validity_status" in src
    assert "md_status" in src
    assert "Reject" in src
    assert "invalid" in src
    assert "failed" in src


def test_s11_run_splits_into_accepted_and_blocked():
    """After scoring, s11 must explicitly partition into `accepted`
    and `blocked` lists. Anyone reverting to a single sorted list
    fails this guard."""
    from evoliez.stages import s11_final_ranking
    src = inspect.getsource(s11_final_ranking.FinalRankingStage.run)
    assert "accepted: List[Candidate]" in src or "accepted = []" in src or "accepted: List" in src
    assert "blocked: List[Candidate]" in src or "blocked = []" in src or "blocked: List" in src
    # The recombined order MUST be accepted-then-blocked (not interleaved).
    assert "accepted + blocked" in src


def test_s11_persists_top_accepted_not_raw_top():
    """ctx.persist_meta('top_candidate', ...) must use ACCEPTED, not the
    raw top-scoring candidate. Otherwise the meta field still lies
    when a Reject candidate has the highest final_score."""
    from evoliez.stages import s11_final_ranking
    src = inspect.getsource(s11_final_ranking.FinalRankingStage.run)
    assert "accepted[0]" in src, (
        "top_candidate meta must come from accepted[0], not ranked[0] / "
        "validated[0] - otherwise a blocked candidate at rank 1 leaks "
        "into downstream meta consumers"
    )


# --------------------------------------------------------------------- #
# 2. Blocked logic per condition
# --------------------------------------------------------------------- #
def _blocked_with_reason_inline(c: Candidate):
    """Mirror of the s11 helper for test-level invocation."""
    ec = c.scores.get("evidence_class") or c.details.get("evidence_class")
    if ec == "Reject":
        return True, "evidence_class=Reject"
    pose = c.scores.get("pose_validity_status") or c.details.get("pose_validity_status")
    if pose == "invalid":
        return True, "pose_validity_status=invalid"
    md_status = c.scores.get("md_status") or c.details.get("md_status")
    if md_status == "failed":
        return True, "md_status=failed"
    return False, ""


def test_evidence_reject_is_blocked():
    c = _mk_cand("c1", 5.0, evidence="Reject")
    blocked, reason = _blocked_with_reason_inline(c)
    assert blocked
    assert reason == "evidence_class=Reject"


def test_pose_invalid_is_blocked_even_with_strong_evidence():
    """The S335T production case: evidence=Reject (because of invalid pose)
    OR evidence=Strong but pose=invalid - both must block."""
    c = _mk_cand("c1", 5.0, evidence="Strong", pose_valid="invalid")
    blocked, reason = _blocked_with_reason_inline(c)
    assert blocked
    assert reason == "pose_validity_status=invalid"


def test_md_failed_is_blocked():
    c = _mk_cand("c1", 5.0, evidence="Promising", md_status="failed")
    blocked, reason = _blocked_with_reason_inline(c)
    assert blocked
    assert reason == "md_status=failed"


def test_clean_candidate_is_not_blocked():
    c = _mk_cand("c1", 5.0, evidence="Strong", pose_valid="valid", md_status="ok")
    blocked, reason = _blocked_with_reason_inline(c)
    assert not blocked
    assert reason == ""


# --------------------------------------------------------------------- #
# 3. Production-shape reproducer: D222N should be top of accepted
# --------------------------------------------------------------------- #
def test_production_shape_d222n_climbs_to_accepted_top():
    """Reproduce the PseFDH production case (without re-running the
    pipeline): the raw top-scoring candidate is a Reject; after gate,
    a clean Strong-evidence D222N should be the top of `accepted`.

    Mirrors the production csv:
      rank 1 mut_00053 S335T  score=3.07  Reject/invalid  -> BLOCKED
      rank 2 mut_00030 T221N  score=3.04  Strong/valid    -> ACCEPTED #1
      ...
      rank 6 mut_00037 D222N  score=2.74  Strong/valid    -> still in accepted
    """
    cands = [
        _mk_cand("mut_00053", 3.0752, evidence="Reject", pose_valid="invalid"),
        _mk_cand("mut_00030", 3.0401, evidence="Strong", pose_valid="valid"),
        _mk_cand("mut_00054", 2.9131, evidence="Promising", pose_valid="invalid"),
        _mk_cand("mut_00029", 2.8934, evidence="Strong", pose_valid="valid"),
        _mk_cand("mut_00065", 2.8590, evidence="Strong", pose_valid="valid"),
        _mk_cand("mut_00037", 2.7408, evidence="Strong", pose_valid="valid"),
    ]
    # Apply the same logic the stage uses
    accepted = []
    blocked = []
    for c in cands:
        is_blocked, reason = _blocked_with_reason_inline(c)
        c.details["is_blocked"] = is_blocked
        c.details["block_reason"] = reason
        (blocked if is_blocked else accepted).append(c)
    accepted.sort(key=lambda c: -c.scores["final_score"])
    blocked.sort(key=lambda c: -c.scores["final_score"])

    # mut_00053 (Reject/invalid) and mut_00054 (Promising/invalid) blocked
    assert {c.candidate_id for c in blocked} == {"mut_00053", "mut_00054"}
    # Top accepted is no longer the raw top-scoring
    assert accepted[0].candidate_id == "mut_00030", (
        f"Expected mut_00030 as top accepted, got {accepted[0].candidate_id}"
    )
    # D222N (mut_00037) was rank 6 raw; should still be in accepted (rank 4 of 4)
    assert "mut_00037" in {c.candidate_id for c in accepted}


# --------------------------------------------------------------------- #
# 4. Report writer: new columns + library filters blocked
# --------------------------------------------------------------------- #
def test_final_candidates_csv_has_gate_columns(tmp_path):
    """final_candidates.csv must carry `is_blocked` + `block_reason`
    columns immediately after the identifier columns, so a user
    scanning the file sees the gate first."""
    pytest.importorskip("pandas", reason="not strictly required; csv reader works")
    from evoliez.io.report import _write_candidates_csv
    from evoliez.io.paths import ProjectPaths

    paths = ProjectPaths(root=tmp_path)
    paths.reports.mkdir(parents=True, exist_ok=True)
    cands = [
        _mk_cand("c1", 5.0, evidence="Strong", pose_valid="valid"),
        _mk_cand("c2", 4.0, evidence="Reject", pose_valid="invalid"),
    ]
    # Annotate as if the stage already gated them
    for c in cands:
        is_blocked, reason = _blocked_with_reason_inline(c)
        c.details["is_blocked"] = is_blocked
        c.details["block_reason"] = reason
        c.scores["is_blocked"] = int(is_blocked)
        c.scores["block_reason"] = reason
    csv_path = _write_candidates_csv(paths, cands)
    rows = list(csv.DictReader(csv_path.open()))
    assert rows[0]["candidate_id"] == "c1"
    assert rows[0]["is_blocked"] == "0"
    assert rows[0]["block_reason"] == ""
    assert rows[1]["candidate_id"] == "c2"
    assert rows[1]["is_blocked"] == "1"
    assert "Reject" in rows[1]["block_reason"]


def test_focused_library_csv_excludes_blocked(tmp_path):
    """The library is the wet-lab order. Blocked candidates MUST NOT
    appear, even when their final_score is high."""
    from evoliez.config import load_config
    from evoliez.io.report import _write_library_csv
    from evoliez.io.paths import ProjectPaths

    # Real config from example fixture (avoids fragile inline Config())
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "example_fdh_nadp.yaml",
        {"project.output_dir": str(tmp_path / "out"),
         "output.final_library_size": 10},
    )
    paths = ProjectPaths(root=tmp_path)
    paths.reports.mkdir(parents=True, exist_ok=True)

    cands = [
        _mk_cand("blocked_high", 9.9, evidence="Reject"),
        _mk_cand("accepted_1", 5.0, evidence="Strong"),
        _mk_cand("blocked_invalid", 8.5, evidence="Strong", pose_valid="invalid"),
        _mk_cand("accepted_2", 4.0, evidence="Promising"),
    ]
    for c in cands:
        is_blocked, reason = _blocked_with_reason_inline(c)
        c.details["is_blocked"] = is_blocked
        c.scores["is_blocked"] = int(is_blocked)
        c.scores["block_reason"] = reason
    csv_path = _write_library_csv(cfg, paths, cands)
    rows = list(csv.DictReader(csv_path.open()))
    ids = [r["candidate_id"] for r in rows]
    assert ids == ["accepted_1", "accepted_2"], (
        f"library must contain ONLY accepted candidates; got {ids}"
    )
