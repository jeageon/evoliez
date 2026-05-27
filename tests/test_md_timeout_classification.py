"""F (post-expert-audit) — timeout status classification.

Before F, `openmm_subprocess.py` reported subprocess timeout as
`status="failed"`. That collapsed two distinct failure modes — a hang
in sqm vs a crash in create_system — into one bucket, and pinned
`md_timeout_rate` at 0 even when MD was timing out.

This file pins:
  - s10_md buckets timeout separately (`n_timeout`).
  - The honesty log line surfaces the count.
  - persist_meta records `n_md_timeout`.
  - bench_summary's `_md_validation_counts` already has a 'timeout'
    branch; this test makes that explicit.
"""

from __future__ import annotations

import inspect

from evoliez.adapters.openmm_engine import MDResult
from evoliez.ml.bench_summary import _md_validation_counts
from evoliez.stages import s10_md


def test_s10_md_run_buckets_timeout_separately():
    """Source-level guard: a future refactor must not collapse
    n_timeout back into n_failed."""
    src = inspect.getsource(s10_md.MDStage.run)
    assert "n_timeout" in src, (
        "s10 must initialise a separate n_timeout counter (F)"
    )
    # Both classifications present in the dispatch chain.
    assert 'n_timeout += 1' in src
    assert 'n_failed += 1' in src
    # persist_meta records the new bucket.
    assert 'persist_meta("n_md_timeout"' in src


def test_s10_md_honesty_line_shows_timeout_count():
    """The greppable real-execution honesty log must surface timeout
    separately so a server operator scanning logs sees 'timeout=N'
    instead of mass 'failed'."""
    src = inspect.getsource(s10_md.MDStage.run)
    # Look for the formatted log string.
    assert "timeout=%d" in src, (
        "s10 honesty log line should include timeout=N alongside "
        "skipped=N, failed=N"
    )


def test_md_did_run_treats_timeout_as_did_not_run():
    """Companion to E: md_did_run must be False for status=timeout,
    not True. Otherwise a timed-out MD would be classified as 'ran'
    in the report which is the exact dishonesty F is trying to fix."""
    src = inspect.getsource(s10_md.MDStage.run)
    # The md_did_run definition must include the timeout branch.
    assert 'status == "timeout"' in src or 'result.status == "timeout"' in src


def test_bench_summary_recognises_timeout_status():
    """`_md_validation_counts` must put status='timeout' in its 'timeout'
    bucket (not 'failed', not 'missing'). The harness uses this to
    distinguish hung-MD runs from crashed-MD runs."""
    accepted_rows = [
        {"candidate_id": "c1"},
        {"candidate_id": "c2"},
        {"candidate_id": "c3"},
    ]
    statuses = {
        "c1": "ok",
        "c2": "timeout",
        "c3": "failed",
    }
    counts = _md_validation_counts(accepted_rows, statuses, top_k=3)
    assert counts["ok"] == 1
    assert counts["timeout"] == 1
    assert counts["failed"] == 1
    assert counts["total"] == 3
    # `not validated` total should include the timeout (it didn't run).
    not_validated = (
        counts["failed"] + counts["timeout"]
        + counts["skipped"] + counts["missing"]
    )
    assert not_validated == 2


def test_mdresult_default_status_is_failed_not_timeout():
    """Sanity check that MDResult's default `status="failed"` argument
    on the synthetic helper still works — we only changed the timeout
    branch, not the crash/marshal-error paths."""
    from evoliez.adapters.openmm_subprocess import _failed_result
    from evoliez.config import MDConfig

    cfg = MDConfig()
    crash = _failed_result("cX", cfg, "subprocess crashed (rc=7)")
    assert crash.status == "failed"

    timeout = _failed_result(
        "cY", cfg, "subprocess timeout after 60s", status="timeout"
    )
    assert timeout.status == "timeout"


def test_timeout_status_persists_through_analysis_json():
    """Integration: a timeout MDResult round-tripped through
    `md.analysis.to_json` must land status='timeout' on disk so
    scan_md_statuses can read it back as such."""
    from evoliez.md.analysis import MDMetrics, to_json

    metrics = MDMetrics()
    metrics.passed = False
    metrics.failure_reasons = ["subprocess timeout"]
    result = MDResult(
        candidate_id="c_to",
        status="timeout",
        protocol_level=1,
        solvent_mode="implicit",
        simulation_time_ns=0.0,
        integration_failed=True,
        failure_reason="subprocess timeout after 1800s",
    )
    out = to_json(metrics, result)
    assert out["status"] == "timeout"
    assert out["md_did_run"] is False
    assert "timeout" in out["failure_reason"].lower()
