"""V4-9: round-2 rejects final_score-only candidate input (ROADMAP_V4 invariant #4)."""

import pytest

from evoliez.experimental.acquisition import (
    AcquisitionFeatures,
    Round2Candidate,
    propose_round2_from_rows,
)
from evoliez.experimental.calibration import CalibrationResult


def test_final_score_only_rows_rejected():
    l1 = CalibrationResult(claim_level="L1_screened")
    rows = [
        {"candidate_id": "m1", "mutation": "Q382R", "final_score": 1.4},
        {"candidate_id": "m2", "mutation": "R207K", "final_score": 1.3},
    ]
    cands = [Round2Candidate("Q382R", "exploit", AcquisitionFeatures())]
    with pytest.raises(ValueError):
        propose_round2_from_rows(l1, rows, cands)


def test_rows_with_evidence_are_accepted():
    l1 = CalibrationResult(claim_level="L1_screened")
    rows = [{"candidate_id": "m1", "mutation": "I208T", "lane": "lead_deconvolution"}]
    cands = [Round2Candidate("I208T", "exploit", AcquisitionFeatures(expected_improvement=1.0))]
    proposal = propose_round2_from_rows(l1, rows, cands, size=4)
    assert proposal and proposal[0].mutation == "I208T"
