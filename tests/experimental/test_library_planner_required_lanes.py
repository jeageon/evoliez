from evoliez.experimental.library_plan import build_first_round_library
from evoliez.experimental.seed_manifest import load_seed_manifest


def test_library_planner_required_lanes():
    manifest = load_seed_manifest()
    rows = build_first_round_library(manifest, size=32)
    lanes = {r.lane for r in rows} | {lane for r in rows for lane in r.all_lanes}
    for lane in manifest.selection_policy.required_lanes:
        assert lane in lanes
    assert len(rows) == 32
