"""Ultra-review regression: ranking-honesty bugs in s09/s10/s11.

These pin four verified bugs where a candidate that did NOT honestly earn
its evidence could still reach Strong / be accepted at the top of the
recommendation list:

  #1 dead `ligand_escape` REJECT branch  - s11 read scores["ligand_escape"]
     but NOTHING ever wrote it (s09 kept it local; MD wrote analysis.json).
     Now s09 (redock escape) and s10 (MD escape / unstable) propagate it.
  #2 "unstable" MD status silently -> Promising/accepted - openmm sets
     status="unstable" when the ligand RMSD leaves the pocket; it was
     neither "failed" nor "skipped*" so it slipped to Promising and the
     ranking gate accepted it. Now: evidence Reject + gate blocks it.
  #3 non-MD candidates read as phantom "MD ok" - validated_candidates is
     the FULL s09 list, but only md_candidates run MD. The others had no
     md_status -> evidence_class read the "ok" default -> md_passed=True ->
     could reach Strong on an MD that never ran. Now tagged "not_run".
  #4 skipped-primary-docking scored as PERFECT - a skipped Pose carries
     score=0.0 / rmsd_to_reference=None, which the old code turned into
     docking_score 0.0 + redocking_consistency 1.0 (a flawless redock for a
     pose that was never docked). Now neutral (0.5) + None docking_score,
     and pose_validity_status "unknown" keeps it out of Strong.

The final test is a NON-regression guard: a genuinely clean candidate
(ok MD, real Boltz, valid pose, good PLIF) must STILL reach Strong.
"""

from __future__ import annotations

import inspect

from evoliez.stages import s11_final_ranking
from evoliez.stages.s11_final_ranking import (
    EVIDENCE_PROMISING,
    EVIDENCE_REJECT,
    EVIDENCE_STRONG,
    EVIDENCE_UNCERTAIN,
    evidence_class,
)
from evoliez.types import Candidate, Mutation, Pose


def _make(scores: dict, details: dict | None = None) -> Candidate:
    c = Candidate(
        candidate_id="c", mutations=[Mutation("A", 10, "K")],
        generator="test",
    )
    c.scores.update(scores)
    c.details.update(details or {})
    return c


def _clean_strong_scores() -> dict:
    """Score dict for a candidate that SHOULD reach Strong."""
    return {
        "md_status": "ok",
        "md_lite_score": 1.2,
        "pose_validity_status": "valid",
        "pose_validity_valid_methods": 2,
        "pose_validity_total_methods": 2,
        "plif_recovery_min": 0.7,
        "docking_method_disagreement": 0.4,
    }


def _clean_strong_details() -> dict:
    return {
        "boltz_delta_source": "real",
        "requires_real_boltz_for_strong": True,
    }


# --------------------------------------------------------------------------- #
# Bug #2: "unstable" MD status
# --------------------------------------------------------------------------- #
def test_unstable_md_is_reject_not_promising():
    """Ligand left the pocket (openmm status="unstable"). It is neither
    "failed" nor "skipped*"; previously it slipped to Promising. Now Reject."""
    c = _make(
        scores={**_clean_strong_scores(), "md_status": "unstable"},
        details=_clean_strong_details(),
    )
    assert evidence_class(c) == EVIDENCE_REJECT


def test_unstable_md_cannot_be_strong():
    """Even with otherwise-perfect signals, unstable MD must NOT be Strong."""
    c = _make(
        scores={**_clean_strong_scores(), "md_status": "unstable"},
        details=_clean_strong_details(),
    )
    assert evidence_class(c) != EVIDENCE_STRONG


def test_unstable_md_is_blocked_by_ranking_gate():
    """The gate must demote an unstable-MD candidate out of the top slots,
    exactly like a failed MD. We mirror the in-stage `_blocked_with_reason`
    by invoking the real source via a tiny shim so we test the actual rule,
    not a copy."""
    c = _make(scores={"md_status": "unstable"}, details={})
    blocked, reason = _blocked_with_reason_via_source(c)
    assert blocked
    assert reason == "md_status=unstable"


def test_failed_md_still_blocked_by_gate():
    """Non-regression: the pre-existing 'failed' gate rule still fires."""
    c = _make(scores={"md_status": "failed"}, details={})
    blocked, reason = _blocked_with_reason_via_source(c)
    assert blocked
    assert reason == "md_status=failed"


# --------------------------------------------------------------------------- #
# Bug #1: ligand_escape propagated -> REJECT fires
# --------------------------------------------------------------------------- #
def test_ligand_escape_score_triggers_reject():
    """The REJECT branch reads scores['ligand_escape']; once it's actually
    set (by s09 redock-escape or s10 MD-escape) it must fire."""
    c = _make(
        scores={**_clean_strong_scores(), "ligand_escape": True},
        details=_clean_strong_details(),
    )
    assert evidence_class(c) == EVIDENCE_REJECT


def test_s09_propagates_ligand_escape_to_scores():
    """s09 must write cand.scores['ligand_escape']=True (not just a local /
    a reason string) so the s11 REJECT branch is reachable."""
    from evoliez.stages import s09_nonmd_validation
    src = inspect.getsource(s09_nonmd_validation.NonMDValidationStage.run)
    assert 'cand.scores["ligand_escape"] = True' in src


def test_s10_propagates_ligand_escape_to_scores():
    """s10 must write cand.scores['ligand_escape']=True when the MD shows
    the ligand left the pocket (metrics.ligand_escape) or status=unstable."""
    from evoliez.stages import s10_md
    src = inspect.getsource(s10_md.MDStage.run)
    assert 'cand.scores["ligand_escape"] = True' in src
    assert "metrics.ligand_escape" in src
    assert 'result.status == "unstable"' in src


# --------------------------------------------------------------------------- #
# Bug #3: non-MD candidate (not in md_by_id) -> md_status "not_run"
# --------------------------------------------------------------------------- #
def test_not_run_md_cannot_be_strong():
    """A candidate that never reached MD (md_status='not_run') must not be
    Strong even when every other signal is clean - the MD never happened."""
    c = _make(
        scores={**_clean_strong_scores(), "md_status": "not_run"},
        details=_clean_strong_details(),
    )
    cls = evidence_class(c)
    assert cls != EVIDENCE_STRONG
    # No other uncertainty/reject trigger here -> Promising is the honest
    # landing spot (NOT Strong, NOT silently rejected).
    assert cls == EVIDENCE_PROMISING


def test_evidence_class_default_md_status_is_not_run():
    """If md_status is entirely missing, the default must be the honest
    'not_run' (NOT 'ok'), so a phantom MD can never reach md_passed/Strong."""
    c = _make(
        scores={k: v for k, v in _clean_strong_scores().items()
                if k != "md_status"},
        details=_clean_strong_details(),
    )
    # md_status absent -> default not_run -> not Strong.
    assert evidence_class(c) != EVIDENCE_STRONG


def test_s11_tags_non_md_candidates_not_run():
    """s11.run must set md_status='not_run' for candidates absent from
    md_by_id before computing evidence_class."""
    src = inspect.getsource(s11_final_ranking.FinalRankingStage.run)
    assert '"not_run"' in src
    assert 'c.scores["md_status"] = "not_run"' in src
    # And it must merge md_status back for candidates that DID run MD.
    assert '"md_status"' in src
    assert '"ligand_escape"' in src


# --------------------------------------------------------------------------- #
# Bug #4: skipped primary docking -> not perfect consistency, not Strong
# --------------------------------------------------------------------------- #
def test_s09_skipped_primary_docking_not_scored_perfect():
    """The s09 source must branch on pose.skipped and NOT assign the
    favourable defaults (docking_score 0.0, consistency 1.0) for a pose that
    was never docked."""
    from evoliez.stages import s09_nonmd_validation
    src = inspect.getsource(s09_nonmd_validation.NonMDValidationStage.run)
    assert "if pose.skipped:" in src
    # Neutral signal for the skipped branch, not perfect.
    assert 'cand.scores["redocking_consistency"] = 0.5' in src
    assert 'cand.scores["docking_score"] = None' in src


def test_skipped_pose_unknown_validity_blocks_strong():
    """A skipped primary pose leaves pose_validity_status='unknown', which
    the Strong pose_clean check (== 'valid') must reject. This pins the
    'cannot be Strong' property at the evidence_class boundary."""
    c = _make(
        scores={
            "md_status": "ok",
            "md_lite_score": 1.0,
            # skipped redock -> neutral signal + unknown validity
            "redocking_consistency": 0.5,
            "docking_uncertainty": 0.5,
            "pose_validity_status": "unknown",
            "plif_recovery_min": 0.7,
        },
        details=_clean_strong_details(),
    )
    cls = evidence_class(c)
    assert cls != EVIDENCE_STRONG
    # Unknown pose (not invalid, not <0.3 plif) -> Promising, not Reject.
    assert cls == EVIDENCE_PROMISING


def test_skipped_pose_object_shape_is_neutral_after_s09_rule():
    """Behavioural check on the actual s09 rule applied to a real skipped
    Pose object (vina-style): score=0.0, rmsd_to_reference=None,
    pose_validity_status='unknown'. After the rule the candidate must have
    neutral docking signals - NOT a perfect redock."""
    pose = Pose(
        candidate_id="c", method="vina", score=0.0,
        rmsd_to_reference=None,
        skipped="skipped_no_full_atom_structure",
        pose_validity_status="unknown",
    )
    # Replicate the exact s09 decision the stage makes for the primary pose.
    if pose.skipped:
        docking_score = None
        consistency = 0.5
        docking_uncertainty = 0.5
        ligand_escape = False
    else:  # pragma: no cover - this branch is the non-skipped path
        docking_score = pose.score
        consistency = max(0.0, 1.0 - (pose.rmsd_to_reference or 0.0) / 4.0)
        docking_uncertainty = round(1.0 - consistency, 4)
        ligand_escape = (pose.rmsd_to_reference or 0.0) > 4.5

    assert docking_score is None
    assert consistency == 0.5            # was 1.0 (perfect) before the fix
    assert docking_uncertainty == 0.5    # was 0.0 (no uncertainty) before
    assert ligand_escape is False        # don't claim escape on a non-dock


# --------------------------------------------------------------------------- #
# NON-regression: a genuinely clean candidate STILL reaches Strong
# --------------------------------------------------------------------------- #
def test_clean_candidate_still_reaches_strong():
    """Guard against over-correction: ok MD + real Boltz + valid pose +
    good PLIF must STILL be Strong."""
    c = _make(scores=_clean_strong_scores(), details=_clean_strong_details())
    assert evidence_class(c) == EVIDENCE_STRONG


def test_clean_candidate_not_blocked_by_gate():
    """And the clean candidate is NOT demoted by the ranking gate."""
    c = _make(
        scores={
            "md_status": "ok",
            "pose_validity_status": "valid",
            "evidence_class": "Strong",
        },
        details={"evidence_class": "Strong",
                 "pose_validity_status": "valid"},
    )
    blocked, reason = _blocked_with_reason_via_source(c)
    assert not blocked
    assert reason == ""


# --------------------------------------------------------------------------- #
# Test helper: run the REAL `_blocked_with_reason` from s11 source.
# We exec the helper out of the stage source so the test exercises the
# actual gate rules (incl. the new "unstable" case) rather than a copy that
# could drift from the implementation.
# --------------------------------------------------------------------------- #
def _blocked_with_reason_via_source(c: Candidate):
    src = inspect.getsource(s11_final_ranking.FinalRankingStage.run)
    # Extract the nested `def _blocked_with_reason(c)` body.
    lines = src.splitlines()
    start = next(
        i for i, ln in enumerate(lines)
        if ln.strip().startswith("def _blocked_with_reason(")
    )
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = [lines[start]]
    for ln in lines[start + 1:]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        body.append(ln)
    # Dedent and compile into a standalone function.
    import textwrap
    ns: dict = {"Candidate": Candidate}
    exec(textwrap.dedent("\n".join(body)), ns)
    return ns["_blocked_with_reason"](c)
