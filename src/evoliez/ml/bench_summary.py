"""Post-hoc multi-enzyme benchmark summary (the expert's metric set).

The existing :func:`evoliez.ml.benchmark.run_benchmark` requires a live
``Pipeline`` run because it consumes ``Candidate`` objects. For the
**multi-enzyme validation campaign** we need a metric writer that:

  * operates on the persisted artefacts (``final_candidates.csv``
    + ``benchmark.csv`` + per-candidate ``md/<cand>/analysis.json``),
  * never re-runs the pipeline (so we can re-score a 4-hour production
    run in <1 s),
  * scales across N cards without re-implementing the metric per card,
  * emits a single markdown the user can paste into the report.

The metric set is **the one the expert validation plan asks for**:

  - ``beneficial_recall_at_k`` — recall@K of known beneficial mutations
    across K ∈ {1, 5, 10, 30}.
  - ``deleterious_bottom_quintile_rate`` — fraction of known deleterious
    mutations landing in the bottom 20 % of the ranked list.
  - ``deleterious_mean_percentile`` — mean rank-percentile of known
    deleterious mutations (100 = best, 0 = worst; closer to 0 means
    we are correctly pushing them down).
  - ``valid_top_candidate_rate`` — fraction of the top-K accepted
    candidates whose ``pose_validity_status`` is ``valid``.
  - ``reject_top_leakage_count`` — number of rows in the top-K
    *accepted* list with ``evidence_class == Reject`` /
    ``pose_validity_status == invalid`` / ``md_status == failed``. The
    P0a ranking gate guarantees this is ``0``; the harness asserts it
    so a regression is caught immediately.
  - ``md_failure_rate`` / ``md_timeout_rate`` — fraction of rows whose
    ``md_status`` is ``failed`` / ``timeout`` respectively.
  - ``per_mutation`` — a small per-row table the markdown writer
    embeds so the reviewer can chase any single benchmark mutation
    back to its rank / evidence / percentile in one place.

The module deliberately does NOT depend on ``evoliez.ml.benchmark`` or
``evoliez.ml.candidate`` so it can be imported in tests / scripts that
don't carry the heavy stack.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

# Default K-set the expert plan asks for. K values above the candidate
# count are clamped automatically.
DEFAULT_KS: Tuple[int, ...] = (1, 5, 10, 30)


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _coerce_bool(v: Any) -> bool:
    """`is_blocked` column round-trips through CSV as '0'/'1'/'True'/'False'."""
    s = str(v).strip().lower()
    return s in ("1", "true", "yes")


def load_final_candidates(csv_path: Path) -> List[Dict[str, Any]]:
    """Load ``final_candidates.csv`` rows. Returns the list in CSV order
    (which is rank-order: P0a puts accepted first then blocked).

    Each row is a dict with the raw column values plus normalised
    ``rank`` (int), ``is_blocked`` (bool), ``mutation`` (str alias for
    ``mutations``).
    """
    out: List[Dict[str, Any]] = []
    if not csv_path.exists():
        return out
    with csv_path.open("r", newline="") as fh:
        for raw in csv.DictReader(fh):
            row = dict(raw)
            row["rank"] = _coerce_int(raw.get("rank"), default=0)
            row["is_blocked"] = (
                _coerce_bool(raw.get("is_blocked", "0"))
                if "is_blocked" in raw else False
            )
            row["mutation"] = (raw.get("mutations") or raw.get("mutation") or "").strip()
            out.append(row)
    return out


def load_benchmark_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """Load ``benchmark.csv``. Tolerates the standard
    ``mutation,label,activity[,source]`` schema."""
    out: List[Dict[str, Any]] = []
    if not csv_path.exists():
        return out
    with csv_path.open("r", newline="") as fh:
        for raw in csv.DictReader(fh):
            mut = (raw.get("mutation") or raw.get("mutations") or "").strip()
            lab = (raw.get("label") or "").strip().lower()
            if not mut or not lab:
                continue
            try:
                act = float(raw.get("activity") or 0.0)
            except (TypeError, ValueError):
                act = 0.0
            out.append({
                "mutation": mut,
                "label": lab,
                "activity": act,
                "source": (raw.get("source") or "").strip(),
            })
    return out


# ---------------------------------------------------------------------------
# MD status scanning
# ---------------------------------------------------------------------------


def scan_md_statuses(
    md_root: Optional[Path],
) -> Dict[str, str]:
    """Walk ``md_root`` for per-candidate ``analysis.json`` files and
    return ``{candidate_id: md_status}``.

    ``md_status`` follows the s10_md vocabulary:
    ``ok / failed / timeout / skipped / unknown``. ``md_root`` is the
    directory holding one sub-directory per candidate (e.g.
    ``run/md/`` with children ``cand_001/``, ``cand_002/``, ...).
    Missing dirs / files are silently ignored - the summary reports
    the rate, not the per-candidate detail.
    """
    statuses: Dict[str, str] = {}
    if md_root is None or not md_root.exists():
        return statuses
    for child in sorted(md_root.iterdir()):
        if not child.is_dir():
            continue
        analysis = child / "analysis.json"
        if not analysis.exists():
            statuses[child.name] = "unknown"
            continue
        try:
            doc = json.loads(analysis.read_text())
        except (OSError, ValueError):
            statuses[child.name] = "unknown"
            continue
        st = (doc.get("status")
              or doc.get("md_status")
              or doc.get("result")
              or "unknown")
        statuses[child.name] = str(st).lower()
    return statuses


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _partition_accepted_blocked(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """P0a guarantees accepted-first ordering; this just splits on the
    column. Falls back to all-accepted when the column is absent (legacy
    pre-P0a runs)."""
    has_gate = any("is_blocked" in r for r in rows)
    if not has_gate:
        return list(rows), []
    accepted = [r for r in rows if not r["is_blocked"]]
    blocked = [r for r in rows if r["is_blocked"]]
    return accepted, blocked


def _percentile_of_rank(rank: int, n: int) -> float:
    """Convert a 1-based rank into a 0-100 percentile (100 = best)."""
    if rank <= 0 or n <= 0:
        return 0.0
    return round(100.0 * (1.0 - (rank - 1) / max(1, n)), 1)


def _recall_at_k(
    beneficial_muts: Sequence[str],
    accepted_in_order: Sequence[Dict[str, Any]],
    ks: Sequence[int],
) -> Dict[int, float]:
    """recall@K = (#beneficial with accepted-rank <= K) / total beneficial.

    Beneficial mutations that are absent from the accepted list (i.e.
    blocked OR not produced) count as not-recovered, NEVER as recovered.
    """
    if not beneficial_muts:
        return {int(k): 0.0 for k in ks}
    by_mut = {r["mutation"]: i for i, r in enumerate(accepted_in_order)}
    out: Dict[int, float] = {}
    for k in ks:
        hits = sum(
            1 for m in beneficial_muts
            if m in by_mut and by_mut[m] < int(k)
        )
        out[int(k)] = round(hits / len(beneficial_muts), 4)
    return out


def _bottom_quintile_rate(
    deleterious_muts: Sequence[str],
    ranked_rows: Sequence[Dict[str, Any]],
    quintile: float = 0.20,
) -> Optional[float]:
    """Fraction of deleterious mutations that land in the bottom
    ``quintile`` of the full ranking (default = bottom 20 %).

    Mutations missing from the ranking get a rank of N+1 (i.e. land in
    the bottom by definition).

    Returns ``None`` (not 0.0) when there are no deleterious rows to
    evaluate. The previous 0.0 return tripped the pass/fail check on
    enzymes whose entire deleterious set was catalytic-protected (e.g.
    XR: all 6 deleterious rows at Y52/K81/H114, all excluded), making
    them FAIL on a metric that literally couldn't be measured.
    """
    if not deleterious_muts:
        return None
    n = len(ranked_rows)
    by_mut = {r["mutation"]: r["rank"] for r in ranked_rows}
    threshold = n * (1.0 - quintile)
    hits = sum(
        1 for m in deleterious_muts
        if by_mut.get(m, n + 1) >= threshold
    )
    return round(hits / len(deleterious_muts), 4)


def _mean_percentile(
    muts: Sequence[str],
    ranked_rows: Sequence[Dict[str, Any]],
) -> Optional[float]:
    """Mean rank percentile (100 = best, 0 = worst). Returns ``None``
    when there are no mutations to evaluate so the markdown reads
    "N/A" instead of a misleading 0.0."""
    if not muts:
        return None
    n = len(ranked_rows)
    by_mut = {r["mutation"]: r["rank"] for r in ranked_rows}
    pcts = [
        _percentile_of_rank(by_mut.get(m, n + 1), n)
        for m in muts
    ]
    return round(sum(pcts) / len(pcts), 1)


def _is_reject_or_invalid(row: Dict[str, Any]) -> bool:
    ev = (row.get("evidence_class") or "").strip().lower()
    pose = (row.get("pose_validity_status") or "").strip().lower()
    md = (row.get("md_status") or "").strip().lower()
    return ev == "reject" or pose == "invalid" or md == "failed"


def _valid_top_rate(
    accepted_top: Sequence[Dict[str, Any]],
) -> float:
    if not accepted_top:
        return 0.0
    n_valid = sum(
        1 for r in accepted_top
        if (r.get("pose_validity_status") or "").strip().lower() == "valid"
    )
    return round(n_valid / len(accepted_top), 4)


def _md_rate(rows: Sequence[Dict[str, Any]], target: str,
             external_statuses: Optional[Dict[str, str]]) -> float:
    """Either the CSV ``md_status`` column OR an externally-scanned
    ``analysis.json`` dict drives this metric - whichever has signal."""
    statuses: List[str] = []
    for r in rows:
        cid = (r.get("candidate_id") or "").strip()
        st = (r.get("md_status") or "").strip().lower()
        if not st and external_statuses and cid in external_statuses:
            st = external_statuses[cid]
        if st:
            statuses.append(st)
    if not statuses:
        return 0.0
    return round(sum(1 for s in statuses if s == target) / len(statuses), 4)


def _md_validation_counts(
    accepted: Sequence[Dict[str, Any]],
    external_statuses: Optional[Dict[str, str]],
    top_k: int,
) -> Dict[str, int]:
    """Per-status counts for the ACCEPTED top-K (the rows that get the
    Strong-evidence boost from real MD). This is the honest accounting
    the expert audit asks for: a candidate that the pipeline put in
    the accepted top-K but whose MD was skipped/failed/timeout/missing
    is NOT validated.

    Returns ``{ok, failed, timeout, skipped, missing, total}`` counts.
    `skipped` covers any md_status starting with "skipped" (including
    ``skipped_no_full_atom_structure``, ``skipped_parameterization``,
    ``skipped_preflight`` for the upcoming early-preflight hook).
    `missing` is when md_status is empty/unknown and the candidate is
    in the top-K (i.e. expected to have run MD but didn't).
    """
    bucket = {"ok": 0, "failed": 0, "timeout": 0, "skipped": 0,
              "missing": 0, "total": 0}
    top_accepted = list(accepted[:top_k])
    bucket["total"] = len(top_accepted)
    for r in top_accepted:
        cid = (r.get("candidate_id") or "").strip()
        st = (r.get("md_status") or "").strip().lower()
        if not st and external_statuses and cid in external_statuses:
            st = external_statuses[cid]
        if not st or st == "unknown":
            bucket["missing"] += 1
        elif st == "ok":
            bucket["ok"] += 1
        elif st == "failed":
            bucket["failed"] += 1
        elif st == "timeout":
            bucket["timeout"] += 1
        elif st.startswith("skipped"):
            bucket["skipped"] += 1
        else:
            # Any unrecognised status (e.g. ``unstable``) counts as a
            # MD that didn't fully validate but didn't crash either.
            # Conservative: treat as "missing" so the operator sees it.
            bucket["missing"] += 1
    return bucket


_MUT_RE = __import__("re").compile(r"^([A-Z])(\d+)([A-Z])$")


def _mut_position(mutation: str) -> Optional[int]:
    """`'D222S'` -> 222, multi-residue or malformed -> None."""
    m = _MUT_RE.match((mutation or "").strip())
    return int(m.group(2)) if m else None


def compute_summary(
    final_candidates_csv: Path,
    benchmark_csv: Path,
    *,
    md_root: Optional[Path] = None,
    ks: Sequence[int] = DEFAULT_KS,
    protected_positions: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Compute the expert-recommended metric set for one enzyme card.

    Returns a dict with stable keys so the markdown writer and tests
    can rely on the layout. Missing inputs degrade gracefully: an
    absent ``benchmark.csv`` zeros the recall/percentile metrics, an
    absent ``final_candidates.csv`` returns a dict with everything zero
    and ``ok=False``.
    """
    cand_rows = load_final_candidates(Path(final_candidates_csv))
    bench_rows = load_benchmark_rows(Path(benchmark_csv))
    md_statuses = scan_md_statuses(md_root) if md_root else {}

    if not cand_rows:
        return {
            "ok": False,
            "reason": f"final_candidates.csv not found / empty: {final_candidates_csv}",
            "ks": [int(k) for k in ks],
        }

    accepted, blocked = _partition_accepted_blocked(cand_rows)
    n_total = len(cand_rows)
    n_accepted = len(accepted)
    n_blocked = len(blocked)

    # Benchmark mutations at catalytic / fixed positions are EXPECTED to be
    # blocked by the pipeline (s07 never proposes a mutation at a residue in
    # `fixed_positions`). Counting them as "not recovered" is dishonest -
    # the pipeline did the right thing. Split them out and exclude from the
    # recall / percentile denominators; the per-mutation chase still lists
    # them with a clear "expected-blocked (catalytic-protected)" reason.
    protected = set(int(p) for p in (protected_positions or []) if p)
    def _is_protected(b):
        p = _mut_position(b.get("mutation", ""))
        return p is not None and p in protected
    bench_active = [b for b in bench_rows if not _is_protected(b)]
    bench_protected = [b for b in bench_rows if _is_protected(b)]
    beneficial = [b["mutation"] for b in bench_active if b["label"] == "beneficial"]
    deleterious = [b["mutation"] for b in bench_active
                   if b["label"] in ("deleterious", "inactive")]

    # Clamp K-set to the accepted-list size so recall@30 on an 8-row
    # accepted set doesn't report a meaningless number.
    eff_ks = [int(k) for k in ks if int(k) <= max(1, n_accepted)] or [max(1, n_accepted)]

    recall_at_k = _recall_at_k(beneficial, accepted, eff_ks)
    del_bottom = _bottom_quintile_rate(deleterious, cand_rows)
    del_mean_pct = _mean_percentile(deleterious, cand_rows)
    ben_mean_pct = _mean_percentile(beneficial, cand_rows)

    top_k_for_validity = max(1, min(10, n_accepted))
    valid_top = _valid_top_rate(accepted[:top_k_for_validity])

    # Reject leakage: must be 0 by P0a contract. Counts the number of
    # *accepted* rows whose evidence/pose/md flags say they should have
    # been blocked. Anything > 0 is a P0a regression.
    reject_leakage = sum(
        1 for r in accepted if _is_reject_or_invalid(r)
    )

    md_fail_rate = _md_rate(cand_rows, "failed", md_statuses)
    md_timeout_rate = _md_rate(cand_rows, "timeout", md_statuses)

    # Honest MD validation accounting on the ACCEPTED top-K. Catches
    # the XR fake-PASS case: 12/12 candidates' MD got preflight-
    # skipped, but md_failure_rate=0 (skip != fail) → harness reports
    # PASS. The expert audit calls this out as the most dangerous
    # blind spot. md_not_validated_rate = (failed + timeout + skipped
    # + missing) / total over the top-K → forces those skips to
    # surface as a benchmark FAIL.
    # 12 matches the cheap-run top_for_md default; clamps to accepted
    # size so smaller runs don't divide by an inflated denominator.
    md_top_k = min(12, n_accepted) if n_accepted else 0
    md_counts = _md_validation_counts(accepted, md_statuses, md_top_k)
    md_not_validated = (
        md_counts["failed"] + md_counts["timeout"]
        + md_counts["skipped"] + md_counts["missing"]
    )
    md_not_validated_rate = (
        round(md_not_validated / md_counts["total"], 4)
        if md_counts["total"] > 0 else 0.0
    )

    # Per-row table for the markdown writer. Keeps only benchmark rows
    # so the file size stays small.
    by_mut_full = {r["mutation"]: r for r in cand_rows}
    per_mutation: List[Dict[str, Any]] = []
    for b in bench_rows:
        row = by_mut_full.get(b["mutation"])
        prot = _is_protected(b)
        if row is None:
            per_mutation.append({
                "mutation": b["mutation"],
                "label": b["label"],
                "rank": None,
                "percentile": 0.0,
                "is_blocked": False,
                "evidence_class": "catalytic-protected (excluded)" if prot else "",
                "in_pipeline": False,
                "protected": prot,
            })
        else:
            per_mutation.append({
                "mutation": b["mutation"],
                "label": b["label"],
                "rank": int(row["rank"]),
                "percentile": _percentile_of_rank(int(row["rank"]), n_total),
                "is_blocked": bool(row["is_blocked"]),
                "evidence_class": row.get("evidence_class", "").strip(),
                "in_pipeline": True,
                "protected": prot,
            })

    # Cheap-run profile metric: fraction of benchmark beneficial +
    # deleterious-active rows that landed ANYWHERE in the final ranking
    # (not necessarily top-K). Validates that the generator + reranker
    # floors are surfacing literature mutations into the pool, even when
    # the cheap-scale signal isn't enough to push them into top-K.
    bench_active_muts = (
        [b["mutation"] for b in bench_active
         if b["label"] in ("beneficial", "deleterious", "inactive")]
    )
    n_in_pool = sum(1 for r in per_mutation
                    if r.get("in_pipeline") and r["mutation"] in bench_active_muts)
    pool_rate = round(n_in_pool / max(1, len(bench_active_muts)), 4)

    return {
        "ok": True,
        "ks": eff_ks,
        "beneficial_recall_at_k": recall_at_k,
        "deleterious_bottom_quintile_rate": del_bottom,
        "deleterious_mean_percentile": del_mean_pct,
        "beneficial_mean_percentile": ben_mean_pct,
        "valid_top_candidate_rate": valid_top,
        "reject_top_leakage_count": int(reject_leakage),
        "md_failure_rate": md_fail_rate,
        "md_timeout_rate": md_timeout_rate,
        # Honest MD validation gate (covers the XR fake-PASS case).
        "md_not_validated_rate": md_not_validated_rate,
        "md_validation_counts": md_counts,
        "benchmark_in_pool_rate": pool_rate,
        "n_candidates_total": n_total,
        "n_candidates_accepted": n_accepted,
        "n_candidates_blocked": n_blocked,
        "n_benchmark": len(bench_rows),
        "n_benchmark_active": len(bench_active),
        "n_benchmark_protected": len(bench_protected),
        "n_beneficial": len(beneficial),
        "n_deleterious": len(deleterious),
        "n_benchmark_in_pool": int(n_in_pool),
        "protected_positions": sorted(protected),
        "per_mutation": per_mutation,
        "inputs": {
            "final_candidates_csv": str(final_candidates_csv),
            "benchmark_csv": str(benchmark_csv),
            "md_root": str(md_root) if md_root else None,
        },
    }


# ---------------------------------------------------------------------------
# Pass/fail gating
# ---------------------------------------------------------------------------


# Default thresholds for FULL-PRODUCTION runs from the expert plan's
# "최소한 이 정도는 봐야 합니다" list. These are starting points; tune
# per enzyme as real data arrives.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_recall_at_10": 0.30,
    "min_recall_at_30": 0.60,
    "min_deleterious_bottom_quintile": 0.50,
    "min_valid_top_rate": 0.70,
    "max_reject_leakage": 0,
    "max_md_failure_rate": 0.30,
    "max_md_timeout_rate": 0.15,
    # Honest MD validation gate. Catches the XR-style fake PASS where
    # 12/12 candidates' MD got preflight-skipped → md_failure_rate
    # stays at 0 but no real MD evidence backs the recommendations.
    # Counts failed + timeout + skipped + missing analysis.json on
    # the accepted top-K.
    "max_md_not_validated_rate": 0.30,
    "min_benchmark_in_pool_rate": 0.30,   # ≥30% of benchmark rows visible
}

# CHEAP-RUN profile: the cheap config (8 Boltz samples / 0.5 ns MD /
# top_for_md=12 / mock homologs) is meant to validate PIPELINE
# CORRECTNESS, not discovery power. At cheap scale:
#   * top_for_md=12 vs ~17 known_binding_site positions: mathematically
#     impossible to guarantee even 1 slot per BS position. The Tishkov
#     D222 family CAN'T reliably claim top-10 ranks regardless of how
#     well the pipeline scores them, because chemistry_rules picks at
#     BS positions (G336/S148/S335) consume the slots.
#   * 8 Boltz diffusion samples vs 30 in prod: pose variance is ~4 Å,
#     so per-mutant Boltz delta features are noisy.
#   * 0.5 ns MD vs 2 ns: too short for the cofactor-switch ligand
#     to drift definitively in WT-poses vs mutant poses.
# The cheap-run still meaningfully checks the recurring expert-plan
# failure modes (Reject leakage, MD failure/timeout, blocking gate,
# subprocess isolation, benchmark-in-pool). It does NOT meaningfully
# check recall@K. Use this profile for cheap-run validation.
CHEAP_RUN_THRESHOLDS: Dict[str, float] = {
    **DEFAULT_THRESHOLDS,
    # Recall@K relaxed to 0 — cheap-run isn't expected to achieve it.
    # The check is still computed and surfaced in the markdown for
    # operator visibility, just doesn't gate pass/fail.
    "min_recall_at_10": 0.0,
    "min_recall_at_30": 0.0,
    # Replace the recall gate with a pool-coverage gate: at least N% of
    # benchmark beneficial+deleterious-active mutations must end up
    # somewhere in the final ranking (not necessarily top-K). This
    # validates the generator + reranker floors actually surface
    # literature mutations into the candidate pool.
    "min_benchmark_in_pool_rate": 0.30,
    # Deleterious bottom-quintile still strict — pipeline must keep
    # known-bad mutations DOWN.
    "min_deleterious_bottom_quintile": 0.50,
    # Honest MD gate stays strict in cheap-run too. XR's fake PASS
    # (12/12 preflight-skipped) only passed because md_failure_rate
    # was 0; this metric counts skips, so the same scenario now FAILS.
    "max_md_not_validated_rate": 0.30,
}

PROFILES: Dict[str, Dict[str, float]] = {
    "prod":  DEFAULT_THRESHOLDS,
    "cheap": CHEAP_RUN_THRESHOLDS,
}


def pass_fail(
    summary: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Apply pass/fail thresholds. Returns ``{passed: bool,
    failures: [str], thresholds: {...}}``."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    failures: List[str] = []
    if not summary.get("ok"):
        failures.append(f"summary not OK: {summary.get('reason')}")
        return {"passed": False, "failures": failures, "thresholds": th}

    recall = summary["beneficial_recall_at_k"]
    r10 = recall.get(10, recall.get(max(recall.keys())))
    r30 = recall.get(30, r10)
    if r10 is not None and r10 < th["min_recall_at_10"]:
        failures.append(
            f"recall@10 = {r10:.3f} < {th['min_recall_at_10']}"
        )
    if r30 is not None and r30 < th["min_recall_at_30"]:
        failures.append(
            f"recall@30 = {r30:.3f} < {th['min_recall_at_30']}"
        )
    # `deleterious_bottom_quintile_rate` is None when there are no
    # active deleterious rows to evaluate (XR case: every deleterious
    # is at a catalytic-protected position). Skip the threshold check
    # in that case rather than failing on a metric that can't be
    # measured — the operator sees "N/A" in the markdown instead.
    del_bq = summary["deleterious_bottom_quintile_rate"]
    if del_bq is not None and del_bq < th["min_deleterious_bottom_quintile"]:
        failures.append(
            f"deleterious bottom-quintile rate = "
            f"{del_bq:.3f} "
            f"< {th['min_deleterious_bottom_quintile']}"
        )
    if summary["valid_top_candidate_rate"] < th["min_valid_top_rate"]:
        failures.append(
            f"valid top-candidate rate = "
            f"{summary['valid_top_candidate_rate']:.3f} "
            f"< {th['min_valid_top_rate']}"
        )
    if summary["reject_top_leakage_count"] > th["max_reject_leakage"]:
        failures.append(
            f"Reject/invalid leakage into accepted top = "
            f"{summary['reject_top_leakage_count']} "
            f"> {th['max_reject_leakage']} (P0a regression!)"
        )
    if summary["md_failure_rate"] > th["max_md_failure_rate"]:
        failures.append(
            f"MD failure rate = {summary['md_failure_rate']:.3f} "
            f"> {th['max_md_failure_rate']}"
        )
    if summary["md_timeout_rate"] > th["max_md_timeout_rate"]:
        failures.append(
            f"MD timeout rate = {summary['md_timeout_rate']:.3f} "
            f"> {th['max_md_timeout_rate']}"
        )
    # Honest MD validation gate. md_failure_rate alone is a lying
    # metric when the pipeline preflight-skips MD (XR fake-PASS case);
    # the not_validated rate counts skipped + missing analysis.json
    # against the top-K so a "MD never ran" run FAILS.
    not_val = summary.get("md_not_validated_rate", 0.0)
    max_not_val = th.get("max_md_not_validated_rate")
    if max_not_val is not None and not_val > max_not_val:
        counts = summary.get("md_validation_counts", {})
        failures.append(
            f"MD not-validated rate = {not_val:.3f} > {max_not_val} "
            f"(top-{counts.get('total', '?')}: "
            f"ok={counts.get('ok', 0)} "
            f"failed={counts.get('failed', 0)} "
            f"timeout={counts.get('timeout', 0)} "
            f"skipped={counts.get('skipped', 0)} "
            f"missing={counts.get('missing', 0)})"
        )
    # Cheap-run-friendly: pool-coverage gate. Only checked when the
    # threshold is set; defaults to 0 in DEFAULT_THRESHOLDS so prod
    # runs don't get this on top of their stricter recall@K checks
    # (recall@K already implies in-pool).
    pool_floor = th.get("min_benchmark_in_pool_rate", 0)
    pool_rate = summary.get("benchmark_in_pool_rate")
    if pool_floor > 0 and pool_rate is not None and pool_rate < pool_floor:
        failures.append(
            f"benchmark_in_pool_rate = {pool_rate:.3f} < {pool_floor} "
            f"({summary.get('n_benchmark_in_pool', '?')}/"
            f"{summary.get('n_benchmark_active', '?')} active rows in CSV)"
        )
    return {"passed": not failures, "failures": failures, "thresholds": th}


# ---------------------------------------------------------------------------
# Markdown renderers
# ---------------------------------------------------------------------------


_PASS_BADGE = "✅ PASS"
_FAIL_BADGE = "❌ FAIL"


def render_card_markdown(name: str, summary: Dict[str, Any],
                         pf: Dict[str, Any]) -> str:
    """Single-card report (one enzyme)."""
    if not summary.get("ok"):
        return (
            f"## {name} — {_FAIL_BADGE}\n\n"
            f"`{summary.get('reason', 'unknown error')}`\n"
        )
    badge = _PASS_BADGE if pf["passed"] else _FAIL_BADGE
    lines: List[str] = [
        f"## {name} — {badge}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Candidates (total / accepted / blocked) | "
        f"{summary['n_candidates_total']} / "
        f"{summary['n_candidates_accepted']} / "
        f"{summary['n_candidates_blocked']} |",
        f"| Benchmark rows (beneficial / deleterious) | "
        f"{summary['n_benchmark']} "
        f"({summary['n_beneficial']} / {summary['n_deleterious']}) |",
    ]
    n_prot = int(summary.get("n_benchmark_protected", 0))
    if n_prot:
        prot_pos = summary.get("protected_positions", [])
        lines.append(
            f"| Catalytic-protected benchmark rows (excluded from "
            f"recall denominator) | {n_prot} at positions "
            f"{','.join(str(p) for p in prot_pos)} |"
        )
    for k, v in sorted(summary["beneficial_recall_at_k"].items()):
        lines.append(f"| Recall@{k} | {v:.3f} |")
    # Helpers to render None as "N/A" (e.g. when all deleterious rows
    # are catalytic-protected and the metric has no denominator).
    def _f3(x): return "N/A" if x is None else f"{x:.3f}"
    def _f1(x): return "N/A" if x is None else f"{x:.1f}"
    lines.extend([
        f"| Deleterious bottom-quintile rate | "
        f"{_f3(summary['deleterious_bottom_quintile_rate'])} |",
        f"| Deleterious mean percentile (lower = better) | "
        f"{_f1(summary['deleterious_mean_percentile'])} |",
        f"| Beneficial mean percentile (higher = better) | "
        f"{_f1(summary['beneficial_mean_percentile'])} |",
        f"| Valid top-K candidate rate | "
        f"{summary['valid_top_candidate_rate']:.3f} |",
        f"| Reject/invalid top-leakage (must be 0) | "
        f"{summary['reject_top_leakage_count']} |",
        f"| MD failure rate | {summary['md_failure_rate']:.3f} |",
        f"| MD timeout rate | {summary['md_timeout_rate']:.3f} |",
    ])
    # MD validation breakdown — surface the honest counts on the
    # accepted top-K so the operator can tell ok / skipped / missing
    # at a glance, not just a rolled-up rate.
    nv = summary.get("md_validation_counts") or {}
    if nv:
        lines.append(
            f"| **MD not-validated rate** (accepted top-{nv.get('total', '?')}) | "
            f"**{summary.get('md_not_validated_rate', 0):.3f}**  "
            f"ok={nv.get('ok', 0)} "
            f"failed={nv.get('failed', 0)} "
            f"timeout={nv.get('timeout', 0)} "
            f"skipped={nv.get('skipped', 0)} "
            f"missing={nv.get('missing', 0)} |"
        )
    lines.extend([
        f"| Benchmark-in-pool rate | "
        f"{summary.get('benchmark_in_pool_rate', 0):.3f} "
        f"({summary.get('n_benchmark_in_pool', 0)}/"
        f"{summary.get('n_benchmark_active', 0)}) |",
    ])
    if pf["failures"]:
        lines.extend([
            "",
            "**Failed checks**",
            "",
        ])
        for f in pf["failures"]:
            lines.append(f"  - {f}")

    # Per-mutation chase table
    lines.extend(["", "### Per-mutation chase", "",
                  "| Mutation | Label | Rank | Percentile | Blocked? | Evidence |",
                  "|---|---|---|---|---|---|"])
    for r in summary["per_mutation"]:
        rank = r["rank"] if r["in_pipeline"] else "—"
        if r.get("protected"):
            blocked = "catalytic-protected"
        else:
            blocked = "yes" if r["is_blocked"] else ""
        lines.append(
            f"| {r['mutation']} | {r['label']} | {rank} | "
            f"{r['percentile']:.1f} | {blocked} | {r['evidence_class']} |"
        )
    return "\n".join(lines) + "\n"


def render_multi_card_markdown(
    cards: Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> str:
    """Aggregate markdown across multiple enzyme cards. ``cards`` is an
    iterable of ``(name, summary, pass_fail)`` tuples."""
    cards_list = list(cards)
    lines: List[str] = [
        "# Multi-enzyme benchmark summary",
        "",
        "Auto-generated by `evoliez.ml.bench_summary` — one row per "
        "enzyme card. The expert validation plan's pass-fail criteria "
        "are the default thresholds; tune per enzyme as real data "
        "arrives.",
        "",
        "## Cross-card overview",
        "",
        "| Enzyme | Status | Recall@10 | Recall@30 | Del. bottom-Q | "
        "Valid top | Reject leak | MD fail | MD timeout | "
        "**MD not-validated** | Bench-in-pool |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s, pf in cards_list:
        if not s.get("ok"):
            lines.append(
                f"| {name} | {_FAIL_BADGE} | – | – | – | – | – | – | – | – | – |"
            )
            continue
        recall = s["beneficial_recall_at_k"]
        r10 = recall.get(10, recall.get(max(recall.keys())))
        r30 = recall.get(30, r10)
        del_bq = s["deleterious_bottom_quintile_rate"]
        del_bq_s = "N/A" if del_bq is None else f"{del_bq:.3f}"
        lines.append(
            f"| {name} | {_PASS_BADGE if pf['passed'] else _FAIL_BADGE} "
            f"| {r10:.3f} | {r30:.3f} "
            f"| {del_bq_s} "
            f"| {s['valid_top_candidate_rate']:.3f} "
            f"| {s['reject_top_leakage_count']} "
            f"| {s['md_failure_rate']:.3f} "
            f"| {s['md_timeout_rate']:.3f} "
            f"| **{s.get('md_not_validated_rate', 0):.3f}** "
            f"| {s.get('benchmark_in_pool_rate', 0):.3f} |"
        )
    lines.append("")
    # Then expand each card.
    for name, s, pf in cards_list:
        lines.append("---")
        lines.append("")
        lines.append(render_card_markdown(name, s, pf))
    return "\n".join(lines) + "\n"
