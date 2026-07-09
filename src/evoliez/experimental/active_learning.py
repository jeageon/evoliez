"""Round-2 active learning gate.

V4 does not run active learning before wet-lab calibration.
"""

from __future__ import annotations

from .calibration import CALIBRATED_LEVELS, CalibrationResult


def require_calibrated_for_round2(result: CalibrationResult) -> None:
    # Gate on an explicit ALLOWLIST of calibrated levels, not a single denylisted string: a
    # typo / refactor / hand-built result with any other claim_level (''/'L0'/'L1_prelim'/...)
    # previously slipped past the exact-'L0_uncalibrated' check and ran AL on uncalibrated data
    # (Fable review). Anything not known-calibrated fails closed.
    if result.claim_level not in CALIBRATED_LEVELS:
        raise ValueError(
            f"active learning requires wet-lab calibration results; claim_level "
            f"{result.claim_level!r} is not one of the calibrated levels {sorted(CALIBRATED_LEVELS)}")
