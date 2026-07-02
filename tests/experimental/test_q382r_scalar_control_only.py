from evoliez.experimental.library_plan import build_first_round_library
from evoliez.experimental.seed_manifest import load_seed_manifest


def test_q382r_scalar_control_only():
    rows = build_first_round_library(load_seed_manifest(), size=20)
    q382r = [r for r in rows if r.mutation == "Q382R"]
    assert len(q382r) == 1
    assert q382r[0].lane == "scalar_rank_control"
    assert "tier_A_lead" not in q382r[0].all_lanes
