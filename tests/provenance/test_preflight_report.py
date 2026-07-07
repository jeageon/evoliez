"""V4-1: preflight report lists non-valid artifacts with reasons (ROADMAP_V4 §7.2)."""

from pathlib import Path

from evoliez.provenance.artifacts import ArtifactProvenance
from evoliez.provenance.preflight import INVALID, classify_artifact
from evoliez.provenance.preflight_report import (
    render_preflight_report,
    run_reference_preflight,
)

ROOT = Path(__file__).resolve().parents[2]


def test_report_lists_nonvalid_artifacts_with_reasons():
    summary = run_reference_preflight(ROOT)
    assert summary.results, "preflight must classify at least one reference artifact"
    nonvalid = [(s, r) for s, r in summary.results if r.status != "valid"]
    assert nonvalid, "unverified local state should surface non-valid artifacts"
    for _, result in nonvalid:
        assert result.reasons, "every non-valid artifact must carry a reason"


def test_report_text_has_statuses_and_gate():
    summary = run_reference_preflight(ROOT)
    text = render_preflight_report(summary)
    assert "skipped_not_validated" in text
    assert "Next-phase gate" in text
    # Without a structure-verified atom map, V4-2 cannot be cleared for real geometry.
    assert summary.ready_for_v4_2 is False


def test_ca_only_structure_is_hard_invalid():
    # the safety lock the report depends on: a CA-only trace never passes
    ca_only = ArtifactProvenance(
        structure_path="x.pdb", source_stage="s04", backend="boltz",
        atom_count=100, is_full_atom=False,
    )
    assert classify_artifact(ca_only).status == INVALID
