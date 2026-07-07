"""Round-2 active learning gate.

V4 does not run active learning before wet-lab calibration.
"""

from __future__ import annotations

from .calibration import CalibrationResult


def require_calibrated_for_round2(result: CalibrationResult) -> None:
    if result.claim_level == "L0_uncalibrated":
        raise ValueError("active learning requires wet-lab calibration results")
