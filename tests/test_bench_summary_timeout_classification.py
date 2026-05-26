"""Wave 5-C: bench-summary must correctly classify ``failed_timeout``
(Wave 5-B sibling vocabulary) as BOTH a failure AND a timeout, with the
denominator excluding skipped candidates.

Bug it fixes (TEM-1 cheap run):
    MD failure rate | 1.000     ← correct: all 12 MD failed
    MD timeout rate | 0.000     ← WRONG: all 12 actually timed out

The old computation counted exact-match on ``"timeout"`` only, which
silently dropped every ``failed_timeout`` row → operators couldn't see
that the failure mode was specifically a timeout. These tests pin
down the new contract so a future refactor that renames the literal
without updating the rate computation breaks LOUDLY.
"""

from __future__ import annotations

import csv
import inspect
from pathlib import Path
from typing import Dict, List

import pytest

from evoliez.ml import bench_summary as bs
from evoliez.ml.bench_summary import (
    DEFAULT_THRESHOLDS,
    compute_summary,
    pass_fail,
)


# ---------------------------------------------------------------------------
# CSV fixture helper
# ---------------------------------------------------------------------------


_COLS = [
    "rank", "candidate_id", "mutations", "evidence_class",
    "final_score", "is_blocked", "block_reason",
    "pose_validity_status", "md_did_run", "md_status",
]


def _write_csv(path: Path, rows: List[Dict[str, str]]) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COLS})
    return path


def _row(
    rank: int,
    mut: str,
    *,
    md_did_run: str = "1",
    md_status: str = "ok",
    is_blocked: str = "0",
) -> Dict[str, str]:
    return {
        "rank": str(rank),
        "candidate_id": f"c{rank}",
        "mutations": mut,
        "evidence_class": "Promising",
        "final_score": "1.0",
        "is_blocked": is_blocked,
        "block_reason": "",
        "pose_validity_status": "valid",
        "md_did_run": md_did_run,
        "md_status": md_status,
    }


def _bench(path: Path) -> Path:
    # A tiny benchmark so compute_summary doesn't bail early. The recall
    # numbers aren't what we're testing here.
    path.write_text(
        "mutation,label,activity,source\n"
        "A1K,beneficial,1.0,fixture\n"
    )
    return path


# ---------------------------------------------------------------------------
# Required behaviour contract
# ---------------------------------------------------------------------------


def test_timeout_rate_counts_failed_timeout(tmp_path: Path):
    """failed_timeout rows must count toward md_timeout_rate AND
    md_failure_rate (a timeout IS a failure). Skipped rows are
    excluded from the denominator."""
    rows = [
        _row(1, "A1K", md_did_run="1", md_status="ok"),
        _row(2, "A2K", md_did_run="1", md_status="failed_timeout"),
        _row(3, "A3K", md_did_run="1", md_status="failed_timeout"),
        _row(4, "A4K", md_did_run="1", md_status="failed"),
        _row(5, "A5K", md_did_run="0", md_status="skipped_no_params"),
    ]
    cand = _write_csv(tmp_path / "fc.csv", rows)
    bench = _bench(tmp_path / "b.csv")

    s = compute_summary(cand, bench)
    # Denominator = 4 (the skip is excluded).
    # Timeouts = 2 / 4 = 0.5.
    assert s["md_timeout_rate"] == pytest.approx(0.5, abs=1e-4)
    # Failures = 3 / 4 = 0.75 (timeouts count as failures too).
    assert s["md_failure_rate"] == pytest.approx(0.75, abs=1e-4)


def test_timeout_rate_zero_when_all_ok(tmp_path: Path):
    """All-ok run → both rates are 0.0 (and the denominator is non-zero
    so we know it's a real 0, not the empty-denominator fallback)."""
    rows = [
        _row(1, "A1K", md_did_run="1", md_status="ok"),
        _row(2, "A2K", md_did_run="1", md_status="ok"),
        _row(3, "A3K", md_did_run="1", md_status="ok"),
    ]
    cand = _write_csv(tmp_path / "fc.csv", rows)
    bench = _bench(tmp_path / "b.csv")

    s = compute_summary(cand, bench)
    assert s["md_timeout_rate"] == 0.0
    assert s["md_failure_rate"] == 0.0


def test_skipped_candidates_excluded_from_denominator(tmp_path: Path):
    """A run where the pipeline wisely skipped every catalytic mutant
    (e.g. skipped_no_params) must NOT be penalized with a 100 % failure
    rate. Skips aren't in the denominator at all."""
    rows = [
        _row(1, "A1K", md_did_run="0", md_status="skipped_no_params"),
        _row(2, "A2K", md_did_run="0", md_status="skipped_catalytic"),
        _row(3, "A3K", md_did_run="0", md_status="skipped"),
        _row(4, "A4K", md_did_run="1", md_status="ok"),
    ]
    cand = _write_csv(tmp_path / "fc.csv", rows)
    bench = _bench(tmp_path / "b.csv")

    s = compute_summary(cand, bench)
    # Denominator = 1 (only the ok row ran). 0 failures / 0 timeouts.
    assert s["md_failure_rate"] == 0.0
    assert s["md_timeout_rate"] == 0.0


def test_all_timeout_run_matches_tem1_bug_repro(tmp_path: Path):
    """Reproduce the exact TEM-1 cheap-run symptom: 12 candidates, all
    timed out → both rates must read 1.000."""
    rows = [
        _row(i + 1, f"A{i+1}K", md_did_run="1", md_status="failed_timeout")
        for i in range(12)
    ]
    cand = _write_csv(tmp_path / "fc.csv", rows)
    bench = _bench(tmp_path / "b.csv")

    s = compute_summary(cand, bench)
    assert s["md_failure_rate"] == 1.0
    assert s["md_timeout_rate"] == 1.0

    # And the pass/fail check should flag BOTH gates (failure AND
    # timeout), not just the failure one.
    pf = pass_fail(s)
    assert pf["passed"] is False
    assert any("timeout" in f for f in pf["failures"]), pf["failures"]
    assert any("failure" in f for f in pf["failures"]), pf["failures"]


def test_legacy_bare_timeout_status_still_counted(tmp_path: Path):
    """Backwards compat: pre-Wave-5-B code emitted bare ``timeout``.
    That literal must still feed the timeout rate so old artefacts on
    disk re-summarize correctly."""
    rows = [
        _row(1, "A1K", md_did_run="1", md_status="ok"),
        _row(2, "A2K", md_did_run="1", md_status="timeout"),
    ]
    cand = _write_csv(tmp_path / "fc.csv", rows)
    bench = _bench(tmp_path / "b.csv")

    s = compute_summary(cand, bench)
    assert s["md_timeout_rate"] == 0.5
    # And legacy bare timeout still counts as a failure.
    assert s["md_failure_rate"] == 0.5


def test_failed_only_status_does_not_inflate_timeout_rate(tmp_path: Path):
    """A plain ``failed`` (not a timeout) must NOT count toward
    md_timeout_rate — that's the asymmetry the bug fix is built on."""
    rows = [
        _row(1, "A1K", md_did_run="1", md_status="failed"),
        _row(2, "A2K", md_did_run="1", md_status="failed"),
        _row(3, "A3K", md_did_run="1", md_status="ok"),
    ]
    cand = _write_csv(tmp_path / "fc.csv", rows)
    bench = _bench(tmp_path / "b.csv")

    s = compute_summary(cand, bench)
    assert s["md_failure_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["md_timeout_rate"] == 0.0


def test_md_did_run_flag_alone_admits_to_denominator(tmp_path: Path):
    """If md_did_run=1 but md_status is empty (interrupted write?), the
    row is still in the denominator — we just can't classify it, so it
    counts toward neither failure nor timeout."""
    rows = [
        _row(1, "A1K", md_did_run="1", md_status=""),
        _row(2, "A2K", md_did_run="1", md_status="failed_timeout"),
    ]
    cand = _write_csv(tmp_path / "fc.csv", rows)
    bench = _bench(tmp_path / "b.csv")

    s = compute_summary(cand, bench)
    # Denominator = 2. 1 timeout / 1 failure.
    assert s["md_timeout_rate"] == 0.5
    assert s["md_failure_rate"] == 0.5


def test_empty_md_status_and_no_did_run_flag_is_not_in_denominator(
    tmp_path: Path,
):
    """Rows with no MD signal at all (pre-MD run, or MD stage skipped
    entirely) are NOT in the denominator — otherwise pre-MD runs would
    report (0/N) = 0 with N inflated by all-not-run rows, which is
    technically the same number but hides the difference between
    'didn't run' and 'ran with 0 failures'."""
    rows = [
        _row(1, "A1K", md_did_run="", md_status=""),
        _row(2, "A2K", md_did_run="", md_status=""),
    ]
    cand = _write_csv(tmp_path / "fc.csv", rows)
    bench = _bench(tmp_path / "b.csv")

    s = compute_summary(cand, bench)
    # Empty-denominator fallback: both rates are 0.0.
    assert s["md_failure_rate"] == 0.0
    assert s["md_timeout_rate"] == 0.0


# ---------------------------------------------------------------------------
# Threshold integration
# ---------------------------------------------------------------------------


def test_default_thresholds_have_timeout_gate():
    """The default-thresholds dict MUST carry max_md_timeout_rate so
    the pass/fail check can fire on a timeout-dominated run. Without
    this key the gate is silently 0/inf."""
    assert "max_md_timeout_rate" in DEFAULT_THRESHOLDS
    assert 0.0 < DEFAULT_THRESHOLDS["max_md_timeout_rate"] <= 1.0


def test_pass_fail_fires_timeout_gate_independently():
    """A run with high timeout rate but LOW failure rate (somehow) must
    still fail the timeout gate. Synthetic, but it pins down the two
    gates as independent."""
    s = {
        "ok": True,
        "ks": [1, 5, 10, 30],
        "beneficial_recall_at_k": {1: 0.5, 5: 0.7, 10: 0.8, 30: 0.95},
        "deleterious_bottom_quintile_rate": 0.9,
        "deleterious_mean_percentile": 10.0,
        "beneficial_mean_percentile": 90.0,
        "valid_top_candidate_rate": 1.0,
        "reject_top_leakage_count": 0,
        "md_failure_rate": 0.10,          # under the 0.30 gate
        "md_timeout_rate": 0.50,          # over the 0.15 gate
        "benchmark_in_pool_rate": 1.0,
        "n_benchmark_in_pool": 1,
        "n_benchmark_active": 1,
    }
    pf = pass_fail(s)
    assert pf["passed"] is False
    assert any("timeout" in f for f in pf["failures"]), pf["failures"]
    # And the failure gate is NOT in the message list — proof of
    # independence.
    assert not any(
        f.startswith("MD failure rate") for f in pf["failures"]
    ), pf["failures"]


# ---------------------------------------------------------------------------
# Source guard
# ---------------------------------------------------------------------------


def test_source_guard_failed_timeout_literal_present():
    """Source guard: the rate-computation module MUST reference the
    literal ``failed_timeout`` somewhere. If a future refactor renames
    the Wave 5-B vocabulary without updating this module, this test
    breaks LOUDLY rather than silently regressing the TEM-1 bug."""
    src = inspect.getsource(bs)
    assert "failed_timeout" in src, (
        "bench_summary.py must reference the 'failed_timeout' literal "
        "so timeout-class MD failures are surfaced as timeouts (Wave 5-C)."
    )


def test_source_guard_md_rates_helper_exists():
    """The internal helper that computes both rates atomically must
    exist — splitting it back into two single-target calls (the old
    shape) is what produced the TEM-1 bug, because the timeout
    computation looked for exact-match ``"timeout"`` and missed
    ``"failed_timeout"`` rows."""
    assert hasattr(bs, "_md_rates"), (
        "bench_summary must expose a _md_rates(rows, ext) -> (fail, "
        "timeout) helper so failure and timeout rates share a single "
        "classification pass."
    )
