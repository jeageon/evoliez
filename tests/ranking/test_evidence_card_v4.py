import pytest

from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0
from evoliez.guided.ensemble_readout import score_candidate_against_ensemble
from evoliez.ranking.evidence_card_v4 import EvidenceAxisV4, build_evidence_card_v4


def test_evidence_card_schema_v4():
    manifest = load_seed_manifest()
    seed = manifest.find_by_mutation("I208T;R207K;R228P")
    ensemble = build_reference_ensemble_v0(manifest)
    score = score_candidate_against_ensemble(seed.candidate_id, seed.mutation, ensemble)
    card = build_evidence_card_v4(seed, score)
    assert card.reaction_geometry_accommodation.evidence
    assert card.experimental_calibration.evidence == ["no_wetlab_calibration"]
    assert card.pose_uncertainty.reason.endswith("inactivity evidence")


def test_no_bare_boolean_evidence_axes():
    with pytest.raises(ValueError):
        EvidenceAxisV4(score=True)


def test_high_score_low_confidence_representable():
    axis = EvidenceAxisV4(score=0.9, confidence="low", evidence=["synthetic"], provenance={})
    assert axis.score == 0.9
    assert axis.confidence == "low"
