from evoliez.experimental.seed_manifest import load_seed_manifest


def test_q382r_not_catalytic_lead():
    q382r = load_seed_manifest().find_by_mutation("Q382R")
    assert q382r.role == "scalar_rank_control"
    assert q382r.role != "tier_A_catalytic_hypothesis"
