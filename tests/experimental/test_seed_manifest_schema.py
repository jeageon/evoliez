from pathlib import Path

import json

from evoliez.experimental.seed_manifest import DEFAULT_SEED_MANIFEST, compute_manifest_hash, load_seed_manifest


def test_seed_manifest_schema_file_exists():
    schema = Path("src/evoliez/experimental/schemas/v4_seed_manifest.schema.json")
    data = json.loads(schema.read_text())
    assert data["properties"]["schema_version"]["const"] == 4.0


def test_seed_manifest_loads_and_hashes():
    manifest = load_seed_manifest()
    assert manifest.schema_version == 4.0
    assert manifest.claim_level_default == "L0_uncalibrated"
    digest = compute_manifest_hash(DEFAULT_SEED_MANIFEST)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_every_candidate_has_role_source_claim_and_uncertainty():
    manifest = load_seed_manifest()
    for candidate in manifest.candidates:
        assert candidate.role
        assert candidate.evidence_source
        assert candidate.claim_level == "L0_uncalibrated"
        assert candidate.uncertainty_flags
