from pathlib import Path

from evoliez.experimental.calibration import calibrate_assay_results


def test_calibration_metrics_by_lane(tmp_path: Path):
    csv_path = tmp_path / "assay.csv"
    csv_path.write_text(
        "variant_id,mutation,lane,expression,soluble_fraction,NADP_activity,NAD_activity,assay_conditions,replicate_id\n"
        "v1,WT,baseline,1,1,1.0,1.0,std,r1\n"
        "v2,Q382R,scalar_rank_control,1,1,0.8,0.9,std,r1\n"
        "v3,I208T;R207K;R228P,tier_A_lead,1,1,1.3,1.0,std,r1\n"
        "v4,I208T;R207K;R228P,tier_A_lead,1,1,1.4,1.0,std,r2\n"
    )
    result = calibrate_assay_results(csv_path)
    assert result.metrics["hit_rate_by_lane"]["tier_A_lead"] == 1.0
    assert result.claim_level == "L2_calibrated"
