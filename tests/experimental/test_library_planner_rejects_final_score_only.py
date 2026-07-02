import pytest

from evoliez.experimental.library_plan import reject_final_score_only


def test_library_planner_rejects_final_score_only():
    with pytest.raises(ValueError):
        reject_final_score_only([
            {"candidate_id": "mut_00081", "mutation": "Q382R", "final_score": 1.4},
            {"candidate_id": "mut_00013", "mutation": "R207K", "final_score": 1.3},
        ])
    reject_final_score_only([
        {"candidate_id": "mut_00479", "mutation": "I208T;R207K;R228P", "lane": "tier_A_lead"},
    ])
