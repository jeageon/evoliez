from evoliez.experimental.calibration import calibrate_assay_results
from evoliez.ranking import claim_guard as cg
from evoliez.experimental.calibration import claim_provenance_for_calibration


def test_no_wetlab_keeps_claim_l0():
    result = calibrate_assay_results(None)
    assert result.claim_level == "L0_uncalibrated"
    verdict = cg.evaluate(claim_provenance_for_calibration(result))
    assert verdict.allowed_categories == set()
