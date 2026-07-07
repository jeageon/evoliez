from evoliez.experimental.seed_manifest import load_seed_manifest


def test_seed_manifest_expected_roles():
    manifest = load_seed_manifest()
    by_id = {c.candidate_id: c for c in manifest.candidates}
    assert by_id["mut_00479"].mutation == "I208T;R207K;R228P"
    assert by_id["mut_00479"].role == "tier_A_catalytic_hypothesis"
    assert manifest.find_by_mutation("Q382R").role == "scalar_rank_control"


def test_v2_positive_candidates_are_probes_not_confirmed_leads():
    manifest = load_seed_manifest()
    probes = [c for c in manifest.candidates if c.role == "v2_positive_probe"]
    assert probes
    assert all(c.role != "confirmed_lead" for c in probes)
