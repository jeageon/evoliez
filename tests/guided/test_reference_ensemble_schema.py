import pytest

from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0
from evoliez.guided.reference_ensemble import ReferenceConformer, ReferenceEnsemble


def test_reference_ensemble_schema():
    ensemble = build_reference_ensemble_v0(load_seed_manifest())
    assert len(ensemble.conformers) == 8
    assert ensemble.claim_level == "L0_uncalibrated"
    assert ensemble.distributions["hydride_distance_A"].n == 8


def test_ensemble_requires_8_or_reason():
    conformer = build_reference_ensemble_v0(load_seed_manifest()).conformers[0]
    with pytest.raises(ValueError):
        ReferenceEnsemble(
            ensemble_id="bad",
            seed_manifest_hash="x",
            conformers=[ReferenceConformer(**conformer.model_dump())],
        )
