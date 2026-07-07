from evoliez.guided.feasibility import render_feasibility_report, run_feasibility_gate


def test_guided_feasibility_report_contains_go_no_go():
    report = render_feasibility_report(run_feasibility_gate())
    assert "Decision: NO_GO" in report
    assert "ReferenceEnsemble v0 remains the default" in report
