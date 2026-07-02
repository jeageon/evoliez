from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0
from evoliez.guided.ensemble_readout import score_manifest_candidates
from evoliez.reports.v4_geometry_report import render_geometry_report


def test_no_nac_pass_fail_claim():
    manifest = load_seed_manifest()
    report = render_geometry_report(score_manifest_candidates(manifest, build_reference_ensemble_v0(manifest)))
    lowered = report.lower()
    assert "nac pass" not in lowered
    assert "nac fail" not in lowered
    assert "inactive" not in lowered
