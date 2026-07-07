from pathlib import Path

from evoliez.experimental.calibration import calibrate_assay_results


def test_low_ml_false_negative_rate(tmp_path: Path):
    csv_path = tmp_path / "assay.csv"
    csv_path.write_text(
        "variant_id,mutation,lane,expression,soluble_fraction,NADP_activity,NAD_activity,assay_conditions,replicate_id\n"
        "v1,WT,baseline,1,1,1.0,1.0,std,r1\n"
        "v2,N260C,low_ml_uncertainty_probe,1,1,1.2,1.0,std,r1\n"
    )
    result = calibrate_assay_results(csv_path)
    assert result.metrics["low_ml_false_negative_rate"] == 1.0
