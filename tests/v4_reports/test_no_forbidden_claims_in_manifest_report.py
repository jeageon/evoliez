from evoliez.experimental.library_plan import build_first_round_library, render_library_markdown
from evoliez.experimental.seed_manifest import load_seed_manifest, manifest_claim_provenance
from evoliez.ranking import claim_guard as cg


def test_no_forbidden_claims_in_manifest_report():
    manifest = load_seed_manifest()
    report = render_library_markdown(build_first_round_library(manifest, size=20))
    assert cg.lint_report(report, manifest_claim_provenance(manifest)) == []
