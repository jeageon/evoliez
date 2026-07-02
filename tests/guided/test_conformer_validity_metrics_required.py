from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0


def test_conformer_validity_metrics_required():
    ensemble = build_reference_ensemble_v0(load_seed_manifest())
    for conformer in ensemble.conformers:
        assert conformer.validity.ligand_integrity >= 0.8
        assert conformer.validity.reactive_atom_availability == 1.0
        assert "artifact_hash" in conformer.provenance
