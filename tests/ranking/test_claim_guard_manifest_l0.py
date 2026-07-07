from evoliez.experimental.seed_manifest import load_seed_manifest, manifest_claim_provenance
from evoliez.mechanism import vocab
from evoliez.ranking import claim_guard as cg


def test_claim_guard_manifest_l0_without_wetlab():
    manifest = load_seed_manifest()
    verdict = cg.evaluate(manifest_claim_provenance(manifest))
    assert verdict.claim_ceiling == vocab.UNCALIBRATED
    assert verdict.allowed_categories == set()
    assert "uncalibrated" in verdict.flags


def test_manifest_report_overclaim_fails_without_wetlab():
    manifest = load_seed_manifest()
    prov = manifest_claim_provenance(manifest)
    cg.assert_report_clean("Prioritized for experimental testing.", prov)
    try:
        cg.assert_report_clean("Activity improved and kcat improved.", prov)
    except AssertionError:
        pass
    else:
        raise AssertionError("over-claim did not fail")
