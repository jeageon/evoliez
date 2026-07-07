from evoliez.experimental.seed_manifest import load_seed_manifest
from evoliez.guided.ensemble_builder import build_reference_ensemble_v0
from evoliez.guided.ensemble_readout import score_candidate_against_ensemble


def test_mut00479_separates_from_controls_v0():
    manifest = load_seed_manifest()
    ensemble = build_reference_ensemble_v0(manifest)
    lead = score_candidate_against_ensemble("mut_00479", "I208T;R207K;R228P", ensemble)
    q382r = score_candidate_against_ensemble("mut_00081", "Q382R", ensemble)
    s340g = score_candidate_against_ensemble("mut_00224", "S340G", ensemble)
    g291a = score_candidate_against_ensemble("no_gain_g291a", "G291A", ensemble)
    assert lead["reaction_geometry_accommodation"] > q382r["reaction_geometry_accommodation"]
    assert lead["reaction_geometry_accommodation"] > s340g["reaction_geometry_accommodation"]
    assert lead["reaction_geometry_accommodation"] > g291a["reaction_geometry_accommodation"]
    assert q382r["structural_viability"] > lead["structural_viability"]
