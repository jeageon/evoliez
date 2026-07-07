import pytest

from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0
from evoliez.guided.ensemble_readout import score_candidate_against_ensemble
from evoliez.ranking.candidate_accommodation import (
    CandidateAccommodationScore,
    compute_candidate_accommodation,
)
from evoliez.ranking.evidence_card_v4 import build_evidence_card_v4


def test_candidate_accommodation_separate_from_final_score():
    manifest = load_seed_manifest()
    seed = manifest.find_by_mutation("I208T;R207K;R228P")
    ensemble = build_reference_ensemble_v0(manifest)
    readout = score_candidate_against_ensemble(seed.candidate_id, seed.mutation, ensemble)
    card = build_evidence_card_v4(seed, readout)
    score = compute_candidate_accommodation(card)
    assert score.uses_final_score is False
    assert "reaction_geometry_accommodation" in score.components
    with pytest.raises(ValueError):
        CandidateAccommodationScore(
            candidate_id="x",
            mutation="A1B",
            score=1.0,
            uses_final_score=True,
        )
