from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0


def test_reference_distributions_nonempty():
    ensemble = build_reference_ensemble_v0(load_seed_manifest())
    assert set(ensemble.distributions) >= {
        "hydride_distance_A",
        "hydride_angle_deg",
        "anchor_contact",
        "strain_penalty",
    }
    assert ensemble.distributions["hydride_distance_A"].mean > 0
