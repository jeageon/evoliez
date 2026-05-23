"""P0b regression: md/analysis.to_json must persist per-frame RMSD
time-series when an MDResult is provided.

## Production observation
Server's `md/<cand>/analysis.json` carried only summary stats
(`ligand_rmsd_mean`, `ligand_rmsd_final`, ...). The HTML report's
`08_md_rmsd_timeseries` plot then fell back to summary bars because
no time-series arrays were available. Downstream trajectory analysis
(e.g. ligand-escape detection per frame) was also impossible.

## Fix
`to_json(metrics, result=None)`:
  - `result=None` (legacy): summary-only payload, unchanged.
  - `result` provided: also writes `ligand_rmsd_series`,
    `pocket_rmsd_series`, `key_distances`, replica spreads,
    `time_ps` axis derived from `simulation_time_ns` / n_frames.

This file pins both halves.
"""

from __future__ import annotations

import inspect
from typing import List

import pytest

from evoliez.adapters.openmm_engine import MDResult
from evoliez.md.analysis import MDMetrics, to_json


def _mk_metrics() -> MDMetrics:
    m = MDMetrics()
    m.ligand_rmsd_mean = 1.2
    m.ligand_rmsd_final = 2.1
    m.pocket_rmsd_mean = 0.8
    m.md_lite_score = 0.5
    m.passed = True
    m.simulation_health_ok = True
    return m


def _mk_result(n_frames: int = 50,
               sim_time_ns: float = 1.0,
               replicas: int = 1) -> MDResult:
    """Build an MDResult with realistic series shapes."""
    lig_series: List[float] = [round(0.5 + i * 0.02, 3) for i in range(n_frames)]
    pkt_series: List[float] = [round(0.3 + i * 0.01, 3) for i in range(n_frames)]
    result = MDResult(
        candidate_id="cand_X",
        status="ok",
        protocol_level=1,
        solvent_mode="implicit",
        simulation_time_ns=sim_time_ns,
        ligand_rmsd_series=lig_series,
        pocket_rmsd_series=pkt_series,
        key_distances={"D222_NADP_O": [3.0 + i * 0.01 for i in range(n_frames)]},
        replicas_run=replicas,
    )
    if replicas > 1:
        result.ligand_rmsd_replicas = [list(lig_series) for _ in range(replicas)]
        result.pocket_rmsd_replicas = [list(pkt_series) for _ in range(replicas)]
    return result


# --------------------------------------------------------------------- #
# 1. Backward compat: to_json(metrics) without result = legacy payload
# --------------------------------------------------------------------- #
def test_to_json_without_result_is_summary_only():
    """Callers that don't pass `result` get the EXACT pre-P0b payload.
    Critical for any older code that imported to_json(metrics)."""
    metrics = _mk_metrics()
    out = to_json(metrics)
    # Required summary keys present.
    for k in ("ligand_rmsd_mean", "ligand_rmsd_final", "pocket_rmsd_mean",
              "md_lite_score", "passed", "failure_reasons"):
        assert k in out
    # NO time-series keys when result is absent.
    for series_key in ("ligand_rmsd_series", "pocket_rmsd_series",
                       "key_distances", "time_ps", "n_frames"):
        assert series_key not in out


# --------------------------------------------------------------------- #
# 2. With result: per-frame series + derived time axis
# --------------------------------------------------------------------- #
def test_to_json_with_result_persists_series():
    metrics = _mk_metrics()
    result = _mk_result(n_frames=50, sim_time_ns=2.0)
    out = to_json(metrics, result)
    assert out["ligand_rmsd_series"] == result.ligand_rmsd_series
    assert out["pocket_rmsd_series"] == result.pocket_rmsd_series
    assert isinstance(out["key_distances"], dict)
    assert "D222_NADP_O" in out["key_distances"]
    assert len(out["key_distances"]["D222_NADP_O"]) == 50


def test_to_json_derives_time_axis_from_n_frames():
    """`time_ps[i] = i * (sim_time_ns * 1000 / n_frames)`. With 2 ns +
    50 frames, dt = 40 ps. Last frame's time = 49 * 40 = 1960 ps."""
    metrics = _mk_metrics()
    result = _mk_result(n_frames=50, sim_time_ns=2.0)
    out = to_json(metrics, result)
    assert out["n_frames"] == 50
    assert out["dt_ps"] == 40.0
    assert len(out["time_ps"]) == 50
    assert out["time_ps"][0] == 0.0
    assert out["time_ps"][-1] == 1960.0


def test_to_json_includes_replicas_when_present():
    metrics = _mk_metrics()
    result = _mk_result(n_frames=30, sim_time_ns=1.0, replicas=3)
    out = to_json(metrics, result)
    assert "ligand_rmsd_replicas" in out
    assert "pocket_rmsd_replicas" in out
    assert len(out["ligand_rmsd_replicas"]) == 3
    assert len(out["ligand_rmsd_replicas"][0]) == 30


def test_to_json_omits_replica_keys_for_single_replica():
    """No noise in single-replica analysis.json - replica keys appear
    only when there's actually a replica spread to show."""
    metrics = _mk_metrics()
    result = _mk_result(n_frames=20, replicas=1)
    out = to_json(metrics, result)
    assert "ligand_rmsd_replicas" not in out
    assert "pocket_rmsd_replicas" not in out


# --------------------------------------------------------------------- #
# 3. JSON-serialisable: round-trip through json.dumps
# --------------------------------------------------------------------- #
def test_to_json_output_is_json_serialisable():
    import json
    metrics = _mk_metrics()
    result = _mk_result(n_frames=10, replicas=2)
    out = to_json(metrics, result)
    s = json.dumps(out)  # must not raise
    back = json.loads(s)
    assert back["ligand_rmsd_series"] == out["ligand_rmsd_series"]


# --------------------------------------------------------------------- #
# 4. Source guard: s10_md calls to_json with the result
# --------------------------------------------------------------------- #
def test_s10_md_passes_result_to_to_json():
    """s10_md.MDStage.run must call `to_json(metrics, result)`, not
    just `to_json(metrics)`. Otherwise the analysis.json on disk is
    summary-only and the figure layer can't render time series."""
    from evoliez.stages import s10_md
    src = inspect.getsource(s10_md.MDStage.run)
    assert "to_json(metrics, result)" in src, (
        "s10_md must invoke to_json(metrics, result) so analysis.json "
        "carries per-frame RMSD series; otherwise the production output "
        "is summary-only (the P0b production gap)"
    )


# --------------------------------------------------------------------- #
# 5. Empty / mock-backend round-trip
# --------------------------------------------------------------------- #
def test_to_json_handles_empty_series_gracefully():
    """Mock-backend MDResult may have empty series lists. to_json must
    still produce a valid dict with empty lists (not crash)."""
    metrics = _mk_metrics()
    result = MDResult(
        candidate_id="cand_mock",
        status="ok",
        protocol_level=0,
        solvent_mode="implicit",
        simulation_time_ns=0.0,
        ligand_rmsd_series=[],
        pocket_rmsd_series=[],
        key_distances={},
    )
    out = to_json(metrics, result)
    assert out["ligand_rmsd_series"] == []
    assert out["pocket_rmsd_series"] == []
    assert out["key_distances"] == {}
    # No time axis when there are no frames (avoid div-by-zero).
    assert "time_ps" not in out or out.get("time_ps") == []
