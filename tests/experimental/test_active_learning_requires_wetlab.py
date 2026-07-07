"""V4-9: round-2 active learning requires wet-lab calibration (ROADMAP_V4 §7.10)."""

import pytest

from evoliez.experimental.acquisition import (
    AcquisitionFeatures,
    Round2Candidate,
    propose_round2_library,
)
from evoliez.experimental.calibration import CalibrationResult


def test_round2_refuses_without_wetlab():
    l0 = CalibrationResult()  # defaults to L0_uncalibrated
    cands = [Round2Candidate("I208T", "exploit", AcquisitionFeatures(expected_improvement=1.0))]
    with pytest.raises(ValueError):
        propose_round2_library(l0, cands)


def test_round2_runs_when_calibrated():
    l1 = CalibrationResult(claim_level="L1_screened")
    cands = [Round2Candidate("I208T", "exploit", AcquisitionFeatures(expected_improvement=1.0))]
    proposal = propose_round2_library(l1, cands, size=4)
    assert [p.mutation for p in proposal] == ["I208T"]
