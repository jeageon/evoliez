from evoliez.reports.v4_geometry_report import render_geometry_report


def test_soft_kernel_no_binary_claim_flip():
    report = render_geometry_report({"c": {"mutation": "A1B", "reaction_geometry_accommodation": 0.0}})
    assert "pass/fail" not in report.lower()
    assert "inactive" not in report.lower()
