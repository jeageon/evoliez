from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0


def test_geometry_uncertainty_reported():
    ensemble = build_reference_ensemble_v0(load_seed_manifest())
    assert ensemble.uncertainty.reasons
    assert ensemble.uncertainty.evidence_density == "limited"
