from evoliez.experimental.library_plan import build_first_round_library
from evoliez.experimental.seed_manifest import load_seed_manifest


def test_mut00479_deconvolution_included():
    rows = build_first_round_library(load_seed_manifest(), size=20)
    muts = {r.mutation for r in rows}
    assert "I208T;R207K;R228P" in muts
    for mutation in ["I208T", "R207K", "R228P", "I208T;R207K", "I208T;R228P", "R207K;R228P"]:
        assert mutation in muts
