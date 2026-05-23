"""Tests for the post-hoc bench summary harness (evoliez.ml.bench_summary).

The harness is the multi-enzyme validation plan's central tool: it
must (a) compute every metric in the expert audit's list, (b) detect a
P0a regression (Reject/invalid leaking into the accepted top), (c)
gracefully degrade when artefacts are missing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

import pytest

from evoliez.ml.bench_summary import (
    DEFAULT_THRESHOLDS,
    compute_summary,
    load_benchmark_rows,
    load_final_candidates,
    pass_fail,
    render_card_markdown,
    render_multi_card_markdown,
    scan_md_statuses,
)


def _write_candidates(
    path: Path,
    accepted: List[Tuple[str, str, float]],  # (mutation, evidence, score)
    blocked: List[Tuple[str, str, float, str]],  # +pose_validity / md_status
    *,
    pose_validities: List[str] = None,
    md_statuses: List[str] = None,
) -> Path:
    cols = [
        "rank", "candidate_id", "mutations", "evidence_class",
        "final_score", "is_blocked", "block_reason",
        "pose_validity_status", "md_status",
    ]
    rank = 1
    lines = [",".join(cols)]
    pv = pose_validities or ["valid"] * len(accepted)
    ms = md_statuses or ["ok"] * len(accepted)
    for i, (mut, ev, score) in enumerate(accepted):
        lines.append(
            f"{rank},cand_{rank:03d},{mut},{ev},{score:.3f},0,,"
            f"{pv[i]},{ms[i]}"
        )
        rank += 1
    for (mut, ev, score, reason) in blocked:
        lines.append(
            f"{rank},cand_{rank:03d},{mut},{ev},{score:.3f},1,{reason},"
            f"invalid,failed"
        )
        rank += 1
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_benchmark(path: Path, rows: List[Tuple[str, str, float]]) -> Path:
    lines = ["mutation,label,activity,source"]
    for mut, lab, act in rows:
        lines.append(f"{mut},{lab},{act},test-fixture")
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_load_final_candidates_normalises_columns(tmp_path: Path):
    csv_path = _write_candidates(
        tmp_path / "fc.csv",
        accepted=[("A1K", "Strong", 1.0), ("B2L", "Promising", 0.9)],
        blocked=[("C3M", "Reject", 0.1, "evidence_reject")],
    )
    rows = load_final_candidates(csv_path)
    assert len(rows) == 3
    assert rows[0]["mutation"] == "A1K"
    assert rows[0]["is_blocked"] is False
    assert rows[2]["is_blocked"] is True
    assert rows[0]["rank"] == 1
    assert rows[2]["rank"] == 3


def test_load_benchmark_tolerates_extra_columns(tmp_path: Path):
    p = tmp_path / "b.csv"
    p.write_text(
        "mutation,label,activity,source,note\n"
        "D222S,beneficial,1.8,Tishkov,canonical\n"
        ",beneficial,1.0,blank,\n"   # empty mutation should be skipped
        "R285E,deleterious,0.05,configs,catalytic\n"
    )
    rows = load_benchmark_rows(p)
    assert [r["mutation"] for r in rows] == ["D222S", "R285E"]


def test_scan_md_statuses_reads_analysis_json(tmp_path: Path):
    md = tmp_path / "md"
    md.mkdir()
    (md / "cand_001").mkdir()
    (md / "cand_001" / "analysis.json").write_text(
        json.dumps({"status": "ok"})
    )
    (md / "cand_002").mkdir()
    (md / "cand_002" / "analysis.json").write_text(
        json.dumps({"md_status": "FAILED"})
    )
    (md / "cand_003").mkdir()  # no analysis.json → unknown
    statuses = scan_md_statuses(md)
    assert statuses["cand_001"] == "ok"
    assert statuses["cand_002"] == "failed"
    assert statuses["cand_003"] == "unknown"


# ---------------------------------------------------------------------------
# compute_summary — happy path
# ---------------------------------------------------------------------------


def test_compute_summary_basic_metrics(tmp_path: Path):
    """All beneficial in top-3, all deleterious in blocked bin → high
    recall, full bottom-quintile, 0 leakage."""
    accepted = [
        ("D222S", "Strong",     1.5),
        ("D222N", "Strong",     1.3),
        ("Y223H", "Promising",  1.1),
        ("X1Y",   "Promising",  0.9),
        ("X2Y",   "Uncertain",  0.7),
        ("T221N", "Uncertain",  0.5),  # benchmark neutral row
    ]
    blocked = [
        ("R285E", "Reject", 0.1, "evidence_reject"),
        ("H333A", "Reject", 0.1, "evidence_reject"),
        ("N146A", "Reject", 0.1, "evidence_reject"),
    ]
    cand = _write_candidates(tmp_path / "fc.csv", accepted, blocked)
    bench = _write_benchmark(
        tmp_path / "b.csv",
        [("D222S", "beneficial", 1.8),
         ("D222N", "beneficial", 1.4),
         ("Y223H", "beneficial", 1.2),
         ("T221N", "neutral",    1.0),
         ("R285E", "deleterious",0.05),
         ("H333A", "deleterious",0.05),
         ("N146A", "deleterious",0.15)],
    )
    s = compute_summary(cand, bench)
    assert s["ok"] is True
    assert s["n_candidates_total"] == 9
    assert s["n_candidates_accepted"] == 6
    assert s["n_candidates_blocked"] == 3
    assert s["n_benchmark"] == 7
    assert s["n_beneficial"] == 3
    assert s["n_deleterious"] == 3
    # All three beneficial within accepted ranks 1-3 → recall@5 = 1.0
    assert s["beneficial_recall_at_k"][5] == 1.0
    # All three deleterious are in the blocked bin at ranks 7, 8, 9.
    # Bottom-20 % of 9 candidates = ranks 8, 9 (threshold = n*(1-0.2) = 7.2).
    # Rank-7 row sits in the next quintile up → 2/3 = 0.667.
    assert s["deleterious_bottom_quintile_rate"] == pytest.approx(0.6667, abs=1e-3)
    # Mean percentile for the three (ranks 7/8/9 in 9-row list).
    # percentile(r) = 100*(1 - (r-1)/n) → r=7:33.3, r=8:22.2, r=9:11.1
    # → mean ≈ 22.2 (worse half of the list).
    assert s["deleterious_mean_percentile"] == pytest.approx(22.2, abs=0.5)
    # P0a contract: 0 leakage.
    assert s["reject_top_leakage_count"] == 0
    # All accepted rows are pose_validity=valid by default.
    assert s["valid_top_candidate_rate"] == 1.0
    # MD rates count across the FULL ranking (accepted + blocked).
    # The fixture writes blocked rows with md_status=failed so the
    # 3-of-9 blocked rows contribute 0.333 to the failure rate.
    # Accepted rows are md_status=ok by default → 0 failures there.
    assert s["md_failure_rate"] == pytest.approx(3 / 9, abs=1e-3)
    assert s["md_timeout_rate"] == 0.0


def test_compute_summary_detects_p0a_regression(tmp_path: Path):
    """If a Reject row sneaks into accepted, reject_top_leakage_count > 0."""
    accepted = [
        ("D222S", "Reject",    1.5),   # P0a violation simulation
        ("D222N", "Strong",    1.3),
    ]
    cand = _write_candidates(tmp_path / "fc.csv", accepted, blocked=[])
    bench = _write_benchmark(
        tmp_path / "b.csv",
        [("D222S", "beneficial", 1.8), ("D222N", "beneficial", 1.4),
         ("R285E", "deleterious", 0.05), ("R285A", "deleterious", 0.05),
         ("H333A", "deleterious", 0.05)],
    )
    s = compute_summary(cand, bench)
    assert s["reject_top_leakage_count"] == 1
    pf = pass_fail(s)
    assert pf["passed"] is False
    assert any("P0a regression" in msg for msg in pf["failures"])


def test_compute_summary_handles_missing_md_column(tmp_path: Path):
    """Pre-P0a CSV (no md_status column) → MD rates = 0, doesn't crash."""
    cand = tmp_path / "fc.csv"
    cand.write_text(
        "rank,candidate_id,mutations,evidence_class,final_score\n"
        "1,c1,A1K,Strong,1.0\n"
        "2,c2,B2L,Promising,0.9\n"
    )
    bench = _write_benchmark(
        tmp_path / "b.csv",
        [("A1K", "beneficial", 1.8),
         ("X9Y", "deleterious", 0.1),
         ("X8Y", "deleterious", 0.1),
         ("X7Y", "deleterious", 0.1)],
    )
    s = compute_summary(cand, bench)
    assert s["ok"] is True
    assert s["md_failure_rate"] == 0.0
    assert s["md_timeout_rate"] == 0.0
    # Without is_blocked column, all rows are accepted (legacy path).
    assert s["n_candidates_blocked"] == 0


def test_compute_summary_missing_final_candidates_returns_not_ok(tmp_path: Path):
    bench = _write_benchmark(tmp_path / "b.csv", [("A1K", "beneficial", 1.0)])
    s = compute_summary(tmp_path / "nope.csv", bench)
    assert s["ok"] is False
    assert "not found" in s["reason"]


# ---------------------------------------------------------------------------
# pass_fail thresholds
# ---------------------------------------------------------------------------


def test_pass_fail_uses_defaults_when_no_thresholds_passed():
    """Synthetic summary at the defaults - should pass."""
    s = {
        "ok": True,
        "ks": [1, 5, 10, 30],
        "beneficial_recall_at_k": {1: 0.5, 5: 0.7, 10: 0.8, 30: 0.95},
        "deleterious_bottom_quintile_rate": 0.9,
        "deleterious_mean_percentile": 10.0,
        "beneficial_mean_percentile": 85.0,
        "valid_top_candidate_rate": 0.95,
        "reject_top_leakage_count": 0,
        "md_failure_rate": 0.05,
        "md_timeout_rate": 0.02,
        "n_candidates_total": 100,
        "n_candidates_accepted": 80,
        "n_candidates_blocked": 20,
        "n_benchmark": 15,
        "n_beneficial": 7,
        "n_deleterious": 6,
        "per_mutation": [],
        "inputs": {},
    }
    pf = pass_fail(s)
    assert pf["passed"] is True
    assert pf["failures"] == []


def test_pass_fail_fails_on_recall_below_threshold():
    s = {
        "ok": True, "ks": [10],
        "beneficial_recall_at_k": {10: 0.05},
        "deleterious_bottom_quintile_rate": 1.0,
        "deleterious_mean_percentile": 5.0,
        "beneficial_mean_percentile": 90.0,
        "valid_top_candidate_rate": 1.0,
        "reject_top_leakage_count": 0,
        "md_failure_rate": 0.0,
        "md_timeout_rate": 0.0,
        "n_candidates_total": 50, "n_candidates_accepted": 40,
        "n_candidates_blocked": 10, "n_benchmark": 10, "n_beneficial": 5,
        "n_deleterious": 5, "per_mutation": [], "inputs": {},
    }
    pf = pass_fail(s)
    assert pf["passed"] is False
    assert any("recall@10" in f for f in pf["failures"])


def test_pass_fail_threshold_override():
    s = {
        "ok": True, "ks": [10],
        "beneficial_recall_at_k": {10: 0.20},
        "deleterious_bottom_quintile_rate": 1.0,
        "deleterious_mean_percentile": 5.0,
        "beneficial_mean_percentile": 90.0,
        "valid_top_candidate_rate": 1.0,
        "reject_top_leakage_count": 0,
        "md_failure_rate": 0.0, "md_timeout_rate": 0.0,
        "n_candidates_total": 50, "n_candidates_accepted": 40,
        "n_candidates_blocked": 10, "n_benchmark": 10, "n_beneficial": 5,
        "n_deleterious": 5, "per_mutation": [], "inputs": {},
    }
    # Default recall@10 threshold = 0.30 → fails. Override BOTH the @10
    # and @30 thresholds (the harness reuses r10 as r30 when no recall@30
    # bucket is reported, so the default @30 = 0.60 would still trip).
    assert pass_fail(s)["passed"] is False
    pf = pass_fail(s, thresholds={
        "min_recall_at_10": 0.15, "min_recall_at_30": 0.15,
    })
    assert pf["passed"] is True, pf["failures"]


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_card_markdown_contains_metric_rows(tmp_path: Path):
    cand = _write_candidates(
        tmp_path / "fc.csv",
        accepted=[("D222S", "Strong", 1.5), ("D222N", "Strong", 1.3)],
        blocked=[("R285E", "Reject", 0.1, "evidence_reject")],
    )
    bench = _write_benchmark(
        tmp_path / "b.csv",
        [("D222S", "beneficial", 1.8), ("D222N", "beneficial", 1.4),
         ("R285E", "deleterious", 0.05),
         ("R285A", "deleterious", 0.05),
         ("H333A", "deleterious", 0.05)],
    )
    s = compute_summary(cand, bench)
    pf = pass_fail(s)
    md = render_card_markdown("PseFDH", s, pf)
    # Sanity: contains key labels and the per-mutation chase table.
    for needle in ["PseFDH", "Recall@", "Reject/invalid",
                   "MD failure rate", "Per-mutation chase",
                   "D222S", "R285E"]:
        assert needle in md, f"missing {needle!r} in markdown"


def test_render_multi_card_markdown_overview(tmp_path: Path):
    cand = _write_candidates(
        tmp_path / "fc.csv",
        accepted=[("A1K", "Strong", 1.0), ("B2L", "Promising", 0.9)],
        blocked=[("C3M", "Reject", 0.1, "evidence_reject")],
    )
    bench = _write_benchmark(
        tmp_path / "b.csv",
        [("A1K", "beneficial", 1.5),
         ("X9Y", "deleterious", 0.1),
         ("X8Y", "deleterious", 0.1),
         ("X7Y", "deleterious", 0.1)],
    )
    s = compute_summary(cand, bench)
    pf = pass_fail(s)
    md = render_multi_card_markdown([
        ("enzyme_A", s, pf),
        ("enzyme_B", {"ok": False, "reason": "no candidates"}, {"passed": False, "failures": [], "thresholds": {}}),
    ])
    assert "Multi-enzyme benchmark summary" in md
    assert "enzyme_A" in md and "enzyme_B" in md
    assert "Cross-card overview" in md
    # The not-OK card should render with dashes and a fail badge.
    failed_line = [
        ln for ln in md.splitlines()
        if ln.startswith("| enzyme_B |")
    ]
    assert failed_line and "–" in failed_line[0]
