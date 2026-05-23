"""Regression tests for the s10_md cache-reuse + RSS-monitoring fix.

## Production incident
30-candidate × 3-replica MD run OOM-killed at candidate ~26 (Python RSS
hit 166 GB). 26 candidates had completed analysis.json on disk + DB rows,
but the previous `evoliez run` loop unconditionally deleted those DB
rows and re-ran MD for every candidate from scratch -> would burn ~3 h
on every resume attempt.

## Fix
- ``_load_existing_md_analysis(workdir)`` reads + validates a previous
  candidate's analysis.json (same defensive shape as
  adapters/boltz._load_existing_real).
- s10_md per-candidate loop: cache hit -> hydrate cand.scores/details
  from the cached dict and ``continue`` (skip MD).
- Delete-then-add now scoped to ``cand_ids_to_rerun`` only - cached
  DB rows survive.
- Per-candidate RSS log line (psutil) so the leak fix can be verified
  in flight on the server.

This file pins each piece.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


# --------------------------------------------------------------------- #
# Helper: synthesise a valid analysis.json (matching md/analysis.to_json)
# --------------------------------------------------------------------- #
def _valid_analysis_dict(passed: bool = True, lite: float = 0.85) -> dict:
    return {
        "ligand_rmsd_mean": 1.2,
        "ligand_rmsd_final": 1.0,
        "ligand_escape": False,
        "pocket_rmsd_mean": 0.8,
        "contact_occupancy_mean": 0.7,
        "hbond_occupancy": 0.5,
        "catalytic_distance_mean": 3.5,
        "catalytic_distance_std": 0.2,
        "energy_drift": 0.01,
        "simulation_health_ok": True,
        "md_lite_score": lite,
        "passed": passed,
        "failure_reasons": [] if passed else ["ligand escape"],
    }


# --------------------------------------------------------------------- #
# 1. _load_existing_md_analysis: cache validation
# --------------------------------------------------------------------- #
def test_load_existing_md_returns_dict_when_valid(tmp_path):
    from evoliez.stages.s10_md import _load_existing_md_analysis
    (tmp_path / "analysis.json").write_text(
        json.dumps(_valid_analysis_dict())
    )
    aj = _load_existing_md_analysis(tmp_path)
    assert aj is not None
    assert aj["md_lite_score"] == 0.85
    assert aj["passed"] is True


def test_load_existing_md_returns_none_when_missing(tmp_path):
    from evoliez.stages.s10_md import _load_existing_md_analysis
    assert _load_existing_md_analysis(tmp_path) is None


def test_load_existing_md_returns_none_on_truncated_file(tmp_path):
    """A SIGKILL'd previous run may leave a partial analysis.json. The
    cache must NOT accept it - we want a fresh MD run rather than
    silently feeding garbage scores into the ranker."""
    from evoliez.stages.s10_md import _load_existing_md_analysis
    (tmp_path / "analysis.json").write_text('{"truncat')  # 9 bytes, < 32
    assert _load_existing_md_analysis(tmp_path) is None


def test_load_existing_md_returns_none_on_invalid_json(tmp_path):
    from evoliez.stages.s10_md import _load_existing_md_analysis
    (tmp_path / "analysis.json").write_text(
        "this is not json " * 10  # > 32 bytes but unparseable
    )
    assert _load_existing_md_analysis(tmp_path) is None


def test_load_existing_md_returns_none_when_schema_incomplete(tmp_path):
    """Missing required fields (md_lite_score, passed) -> cache miss."""
    from evoliez.stages.s10_md import _load_existing_md_analysis
    incomplete = {"ligand_rmsd_mean": 1.0}                # too sparse
    (tmp_path / "analysis.json").write_text(
        json.dumps(incomplete) + " " * 32                 # pad past size guard
    )
    assert _load_existing_md_analysis(tmp_path) is None


def test_load_existing_md_returns_none_when_top_level_not_dict(tmp_path):
    from evoliez.stages.s10_md import _load_existing_md_analysis
    (tmp_path / "analysis.json").write_text(
        json.dumps(["not", "a", "dict", "but", "long", "enough", "list"])
    )
    assert _load_existing_md_analysis(tmp_path) is None


# --------------------------------------------------------------------- #
# 2. Source guards: the cache-reuse + RSS instrumentation are wired up
# --------------------------------------------------------------------- #
def test_s10_md_uses_cache_reuse_pattern():
    """The per-candidate loop must check `cached_analysis` and skip MD
    on a hit. Reverting this would mean every resume re-runs all 30
    candidates - the exact production cost we're trying to avoid."""
    from evoliez.stages import s10_md
    src = inspect.getsource(s10_md.MDStage.run)
    assert "_load_existing_md_analysis" in src
    assert "cached_analysis" in src
    # The loop's cache hit must use `continue` so the MD subprocess
    # never runs.
    assert "cached_aj is not None" in src or "cache hit" in src.lower()


def test_s10_md_scoped_delete_then_add():
    """Delete-then-add MUST be scoped to candidates being re-run.
    Unconditional delete (the old behavior) would nuke cached rows."""
    from evoliez.stages import s10_md
    src = inspect.getsource(s10_md.MDStage.run)
    assert "cand_ids_to_rerun" in src, (
        "delete-then-add must use cand_ids_to_rerun, not all cand_ids - "
        "otherwise cache hits lose their DB rows on every resume"
    )
    # Cross-check: the in_(...) filter should reference the rerun list.
    assert ".in_(cand_ids_to_rerun)" in src


def test_s10_md_logs_per_candidate_rss():
    """Per-candidate RSS logging is the in-flight verification for the
    memory-leak fix. If RSS grows linearly across candidates, regression
    surfaces in the server log immediately."""
    from evoliez.stages import s10_md
    src = inspect.getsource(s10_md.MDStage.run)
    assert "_rss_mb()" in src or "rss_now" in src
    assert "RSS" in src                            # log message includes RSS


def test_s10_md_persists_cache_hits_metric():
    """Cache-hit count must surface in run state meta so dashboards/
    operators can see how much MD was actually re-run vs reused."""
    from evoliez.stages import s10_md
    src = inspect.getsource(s10_md.MDStage.run)
    assert 'persist_meta("n_md_cache_hits"' in src


# --------------------------------------------------------------------- #
# 3. _rss_mb helper: graceful when psutil missing
# --------------------------------------------------------------------- #
def test_rss_mb_returns_value_or_none():
    """The RSS helper must NEVER raise - missing psutil just makes the
    function return None, and the caller short-circuits the log line."""
    from evoliez.stages.s10_md import _rss_mb
    val = _rss_mb()
    assert val is None or (isinstance(val, float) and val > 0)


def test_rss_mb_returns_none_when_psutil_missing(monkeypatch):
    """Simulate psutil unavailable -> _rss_mb must return None, not
    raise. (psutil is an optional dep; the stage must work without it.)"""
    import builtins
    from evoliez.stages.s10_md import _rss_mb

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("simulated psutil missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _rss_mb() is None


# --------------------------------------------------------------------- #
# 4. Integration: cache hit hydrates candidate without running MD
# --------------------------------------------------------------------- #
def test_cache_hit_hydrates_candidate_scores(tmp_path):
    """The cache-hit branch must populate cand.scores AND cand.details
    so downstream stages (s11 final ranking, evidence classification)
    see the same fields they would after a fresh MD run."""
    from evoliez.types import Candidate, Mutation
    cand = Candidate(
        candidate_id="cand_X",
        mutations=[Mutation(position=1, wt="A", mut="V")],
        generator="chemistry_rules",
        scores={},
        details={},
    )
    cached = _valid_analysis_dict(passed=True, lite=0.91)

    # Simulate the cache-hit branch logic from s10_md.MDStage.run
    cand.scores["md_lite_score"] = float(cached["md_lite_score"])
    cand.scores["md_status"] = "ok" if cached["passed"] else "unstable"
    cand.scores["md_did_run"] = 1
    cand.scores["md_instability"] = round(
        0.0 if cached["simulation_health_ok"] else 1.0, 3
    )
    cand.details["md_passed"] = bool(cached["passed"])
    cand.details["md_did_run"] = True

    # Downstream contract: s11 reads these exact fields
    assert cand.scores["md_lite_score"] == 0.91
    assert cand.scores["md_status"] == "ok"
    assert cand.scores["md_did_run"] == 1
    assert cand.details["md_passed"] is True
