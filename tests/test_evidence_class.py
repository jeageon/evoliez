"""P0.5: evidence class + monotone XGBoost + boltz_delta_source persistence.

The plan demands:
- Final report distinguishes Strong / Promising / Uncertain / Reject.
- A proxy-only candidate CANNOT silently become Strong on a top rank.
- Reranker uses xgboost.monotone_constraints when labels exist.
- `boltz_delta_source` reaches the final report.
"""

from __future__ import annotations

import inspect

import pytest

from evoliez.stages import s08_reranker, s11_final_ranking
from evoliez.stages.s11_final_ranking import (
    EVIDENCE_PROMISING,
    EVIDENCE_REJECT,
    EVIDENCE_STRONG,
    EVIDENCE_UNCERTAIN,
    evidence_class,
)
from evoliez.types import Candidate, Mutation


def _make(scores: dict, details: dict) -> Candidate:
    c = Candidate(
        candidate_id="c", mutations=[Mutation("A", 10, "K")],
        generator="test",
    )
    c.scores.update(scores)
    c.details.update(details)
    return c


# --------------------------------------------------------------------------- #
# evidence_class
# --------------------------------------------------------------------------- #
def test_strong_requires_real_boltz_and_clean_md_and_pose():
    c = _make(
        scores={
            "md_status": "ok",
            "md_lite_score": 1.2,
            "pose_validity_status": "valid",
            "pose_validity_valid_methods": 2,
            "pose_validity_total_methods": 2,
            "plif_recovery_min": 0.7,
            "docking_method_disagreement": 0.4,
        },
        details={
            "boltz_delta_source": "real",
            "requires_real_boltz_for_strong": True,
        },
    )
    assert evidence_class(c) == EVIDENCE_STRONG


def test_proxy_only_top_candidate_cannot_be_strong():
    # Same clean signals as the Strong case but boltz_delta_source=proxy +
    # this candidate IS at the top -> must drop to Uncertain (the plan's
    # "no silent strongest class for proxy-derived candidates").
    c = _make(
        scores={
            "md_status": "ok", "md_lite_score": 1.2,
            "pose_validity_status": "valid",
            "plif_recovery_min": 0.7,
        },
        details={
            "boltz_delta_source": "proxy",
            "requires_real_boltz_for_strong": True,
        },
    )
    assert evidence_class(c) == EVIDENCE_UNCERTAIN


def test_proxy_below_top_threshold_can_be_promising():
    # Below the "top" threshold the real-Boltz requirement is relaxed.
    c = _make(
        scores={
            "md_status": "ok", "md_lite_score": 1.2,
            "pose_validity_status": "valid",
            "plif_recovery_min": 0.7,
        },
        details={
            "boltz_delta_source": "proxy",
            "requires_real_boltz_for_strong": False,
        },
    )
    cls = evidence_class(c)
    # Not Strong (proxy still blocks Strong), not Uncertain (no other
    # uncertainty trigger), so Promising is the right default.
    assert cls == EVIDENCE_PROMISING


def test_md_failure_is_reject():
    c = _make(
        scores={"md_status": "failed"},
        details={"boltz_delta_source": "real"},
    )
    assert evidence_class(c) == EVIDENCE_REJECT


def test_invalid_poses_across_all_methods_is_reject():
    c = _make(
        scores={
            "md_status": "ok",
            "pose_validity_valid_methods": 0,
            "pose_validity_total_methods": 3,
        },
        details={"boltz_delta_source": "real"},
    )
    assert evidence_class(c) == EVIDENCE_REJECT


def test_skipped_md_is_uncertain_not_strong():
    c = _make(
        scores={
            "md_status": "skipped_parameterization",
            "pose_validity_status": "valid",
        },
        details={
            "boltz_delta_source": "real",
            "requires_real_boltz_for_strong": True,
        },
    )
    assert evidence_class(c) == EVIDENCE_UNCERTAIN


def test_high_docking_disagreement_is_uncertain():
    c = _make(
        scores={
            "md_status": "ok", "md_lite_score": 1.0,
            "pose_validity_status": "valid",
            "docking_method_disagreement": 2.5,    # > 1.5 threshold
            "plif_recovery_min": 0.7,
        },
        details={
            "boltz_delta_source": "real",
            "requires_real_boltz_for_strong": True,
        },
    )
    assert evidence_class(c) == EVIDENCE_UNCERTAIN


def test_low_plif_recovery_is_uncertain():
    c = _make(
        scores={
            "md_status": "ok", "md_lite_score": 1.0,
            "pose_validity_status": "valid",
            "plif_recovery_min": 0.2,              # < 0.3 threshold
        },
        details={
            "boltz_delta_source": "real",
            "requires_real_boltz_for_strong": True,
        },
    )
    assert evidence_class(c) == EVIDENCE_UNCERTAIN


# --------------------------------------------------------------------------- #
# Source-level: ranker monotone_constraints + s11 wiring
# --------------------------------------------------------------------------- #
def test_xgboost_reranker_uses_monotone_constraints():
    src = inspect.getsource(s08_reranker.RerankerStage._score_xgboost)
    # The directionality table must be present and threaded into XGBoost.
    assert "monotone_constraints" in src
    assert "monotone[" in src or "monotone = {" in src
    # A couple of well-known directions are locked in by the test so a
    # refactor can't silently flip them.
    assert '"interaction_gain":' in src
    assert '"conservation":' in src


def test_s11_emits_evidence_class_and_pool_and_source():
    src = inspect.getsource(s11_final_ranking.FinalRankingStage.run)
    assert 'evidence_class' in src
    assert 'requires_real_boltz_for_strong' in src
    assert 'pool' in src                          # exploit / explore tag
    assert 'boltz_delta_source' in src            # surfaced onto scores
