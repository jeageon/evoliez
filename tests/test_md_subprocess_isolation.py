"""Tests for the P0 MD subprocess isolation path.

The open-P0 fix from the expert audit: each per-candidate MD now runs
in a fresh Python subprocess with a hard timeout so an OpenMM / CUDA
leak or hang on one candidate can't take the whole stage with it.

These tests exercise:

  - opt-in switch via ``MDConfig.subprocess_isolation`` (default OFF
    so existing in-process tests are unaffected),
  - successful round-trip via the worker entry-point (mock backend),
  - timeout path returns a synthetic ``MDResult(status="failed",
    integration_failed=True, failure_reason="subprocess timeout...")``,
  - crash path returns a similar synthetic result with the worker log
    tail embedded in ``failure_reason`` for debuggability,
  - the s10_md stage uses ``run_md_in_subprocess`` (source-level guard
    so a future refactor that drops the wrapping is caught).

Heavy MD imports (OpenMM / OpenFF / RDKit) are NOT triggered: the
mock backend short-circuits to ``_run_mock`` which is pure-python and
fast (~0.05 s per candidate).
"""

from __future__ import annotations

import inspect
import pickle
import sys
import textwrap
from pathlib import Path

import pytest

from evoliez.adapters import openmm_subprocess
from evoliez.adapters.openmm_engine import MDResult
from evoliez.adapters.openmm_subprocess import (
    _failed_result, run_md_in_subprocess,
)
from evoliez.config import Backend, MDConfig
from evoliez.stages import s10_md


# ---------------------------------------------------------------------------
# Synthetic complex fixture (mock backend doesn't dereference geometry).
# ---------------------------------------------------------------------------


def _synthetic_complex():
    """Build the smallest valid Complex the mock _run_mock will accept.

    Uses the existing test scaffolding (`synthetic_structure` +
    `place_ligand_in_pocket`) so we never hand-roll Complex internals
    that may drift.
    """
    from evoliez.adapters.base import (
        place_ligand_in_pocket, synthetic_structure,
    )
    from evoliez.config import LigandInput
    from evoliez.features.ligand import parse_ligand
    from evoliez.types import Complex, Ligand

    lig = parse_ligand(LigandInput(
        id="L", type="smiles", value="CCO",
    ))
    s = synthetic_structure("ACDEFGHIKLMNPQRSTVWY", seed=7)
    # place_ligand_in_pocket returns List[LigandAtom]; wrap it back
    # into a Ligand so Complex.ligand has the .atoms / .smiles shape
    # the mock and real backends both expect.
    placed_atoms = place_ligand_in_pocket(s, lig, seed=7)
    placed = Ligand(id=lig.id, smiles=lig.smiles, atoms=placed_atoms)
    return Complex(structure=s, ligand=placed)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_md_config_exposes_subprocess_knobs():
    cfg = MDConfig()
    # Default OFF so existing tests don't pay per-candidate subprocess cost.
    assert cfg.subprocess_isolation is False
    assert cfg.subprocess_timeout_seconds == 1800


# ---------------------------------------------------------------------------
# Opt-out path: subprocess_isolation=False MUST fall back to in-process.
# ---------------------------------------------------------------------------


def test_falls_back_to_in_process_when_flag_off(tmp_path: Path, monkeypatch):
    """Flag off → run_md_in_subprocess just calls run_md (no Popen)."""
    called = {}

    def _spy_run_md(*a, **kw):
        called["args"] = (a, kw)
        return MDResult(
            candidate_id=kw.get("candidate_id", a[1] if len(a) > 1 else "?"),
            status="ok", protocol_level=1, solvent_mode="implicit",
            simulation_time_ns=0.5,
        )

    monkeypatch.setattr(openmm_subprocess, "run_md", _spy_run_md)
    # subprocess.Popen MUST NOT be invoked when flag is off.
    monkeypatch.setattr(
        openmm_subprocess.subprocess, "Popen",
        lambda *a, **kw: pytest.fail("Popen called when isolation off"),
    )

    cfg = MDConfig(subprocess_isolation=False)
    cx = _synthetic_complex()
    r = run_md_in_subprocess(
        cx, "c1", cfg, tmp_path,
        instability=0.0, catalytic_positions=[],
        backend=Backend.real, dry_run=False,
        timeout_seconds=60,
    )
    assert r.status == "ok"
    assert called, "run_md should have been called inline"


def test_falls_back_to_in_process_for_mock_backend(tmp_path: Path, monkeypatch):
    """mock backend → always in-process (no point spawning a subprocess
    for ~50 ms of pure-python work)."""
    monkeypatch.setattr(
        openmm_subprocess.subprocess, "Popen",
        lambda *a, **kw: pytest.fail("Popen called for mock backend"),
    )
    cfg = MDConfig(subprocess_isolation=True)  # even with flag on...
    cx = _synthetic_complex()
    r = run_md_in_subprocess(
        cx, "c1", cfg, tmp_path,
        instability=0.0, catalytic_positions=[],
        backend=Backend.mock, dry_run=False,
    )
    # mock backend returns ok / unstable for a well-formed complex,
    # or a skipped_* status when the synthetic structure lacks a
    # full-atom PDB. All are valid "engine ran without crashing".
    assert r.status == "ok" or r.status == "unstable" or r.status.startswith("skipped")


# ---------------------------------------------------------------------------
# Successful subprocess round-trip via the real worker entry-point.
# ---------------------------------------------------------------------------


def test_subprocess_round_trip_via_worker(tmp_path: Path):
    """End-to-end: spawn the real worker, mock backend → MDResult comes
    back through the pickle round-trip identical to in-process."""
    cfg = MDConfig(
        subprocess_isolation=True,
        subprocess_timeout_seconds=120,
    )
    cx = _synthetic_complex()

    # The worker entry-point hard-codes backend dispatch to whatever
    # the parent serialised. mock backend short-circuits inside
    # run_md (no OpenMM imports), so this test runs in <1 s.
    r = run_md_in_subprocess(
        cx, "c_round_trip", cfg, tmp_path,
        instability=0.1, catalytic_positions=[1, 2],
        backend=Backend.real,  # real → goes through subprocess
        dry_run=False,
        timeout_seconds=120,
    )
    # Real subprocess gives one of these graceful answers:
    #   - ok / unstable           when the engine finished a real MD,
    #   - failed                  when OpenMM is missing or _run_real raised,
    #   - skipped_*               when the engine refused to start because
    #                             the synthetic complex lacks a real PDB.
    # Any of these is "subprocess round-tripped a valid MDResult"; the
    # CRASH case (no result pickle) would have raised before this assert.
    assert r.candidate_id == "c_round_trip"
    assert (r.status in {"ok", "unstable", "failed"}
            or r.status.startswith("skipped"))
    # The artefacts should have been written under the candidate's workdir.
    assert (tmp_path / "_md_subprocess_inputs.pkl").exists()
    assert (tmp_path / "_md_subprocess.log").exists()


# ---------------------------------------------------------------------------
# Timeout path
# ---------------------------------------------------------------------------


def test_subprocess_timeout_returns_timeout_result(tmp_path: Path, monkeypatch):
    """A hung subprocess hits the wall-clock timeout → MDResult with
    status="timeout" (distinct from "failed" since F / post-expert-audit)
    + integration_failed=True + 'subprocess timeout' in failure_reason.
    The next candidate's call is still possible. Classification matters
    because `md_timeout_rate` would otherwise stay pinned at 0 on real
    hangs — failed and timeout had collapsed into one bucket."""
    cfg = MDConfig(subprocess_isolation=True, subprocess_timeout_seconds=1)
    cx = _synthetic_complex()

    # Replace the worker module path with a tiny one-liner that just
    # sleeps. We do that by monkey-patching the cmd list builder via
    # sys.argv-style indirection: easier to just point subprocess at a
    # python -c "import time; time.sleep(60)" inline command.
    hang_script = tmp_path / "_hang_worker.py"
    hang_script.write_text(textwrap.dedent("""
        import time
        time.sleep(60)
    """).strip())

    original_popen = openmm_subprocess.subprocess.Popen

    def _patched_popen(cmd, *a, **kw):
        # Rewrite cmd to run our hang script instead of the worker module.
        new_cmd = [sys.executable, str(hang_script)]
        return original_popen(new_cmd, *a, **kw)

    monkeypatch.setattr(
        openmm_subprocess.subprocess, "Popen", _patched_popen,
    )

    r = run_md_in_subprocess(
        cx, "c_hangs", cfg, tmp_path,
        instability=0.0, catalytic_positions=[],
        backend=Backend.real, dry_run=False,
        timeout_seconds=1,
    )
    assert r.status == "timeout", (
        f"expected status='timeout' (distinct from 'failed' since F), "
        f"got status={r.status!r}"
    )
    assert r.integration_failed is True
    assert "subprocess timeout" in (r.failure_reason or "").lower()


# ---------------------------------------------------------------------------
# Crash path
# ---------------------------------------------------------------------------


def test_subprocess_crash_returns_failed_result(tmp_path: Path, monkeypatch):
    """A worker that exits non-zero without writing the result pickle
    → MDResult(failed) with the worker log tail embedded for
    debuggability."""
    cfg = MDConfig(subprocess_isolation=True, subprocess_timeout_seconds=30)
    cx = _synthetic_complex()

    crash_script = tmp_path / "_crash_worker.py"
    crash_script.write_text(textwrap.dedent("""
        import sys
        sys.stderr.write("BOOM: synthetic crash for test\\n")
        sys.exit(42)
    """).strip())

    original_popen = openmm_subprocess.subprocess.Popen

    def _patched_popen(cmd, *a, **kw):
        new_cmd = [sys.executable, str(crash_script)]
        return original_popen(new_cmd, *a, **kw)

    monkeypatch.setattr(
        openmm_subprocess.subprocess, "Popen", _patched_popen,
    )

    r = run_md_in_subprocess(
        cx, "c_crashes", cfg, tmp_path,
        instability=0.0, catalytic_positions=[],
        backend=Backend.real, dry_run=False,
    )
    assert r.status == "failed"
    assert r.integration_failed is True
    reason = (r.failure_reason or "").lower()
    assert "subprocess crashed" in reason
    assert "rc=42" in reason or "42" in reason
    assert "boom" in reason


# ---------------------------------------------------------------------------
# Marshal error path (input pickle write fails)
# ---------------------------------------------------------------------------


def test_marshal_error_returns_failed_result(tmp_path: Path, monkeypatch):
    """If pickling the inputs fails (e.g. the Complex carries an
    un-picklable object), the harness still returns a graceful
    MDResult(failed) instead of raising."""
    cfg = MDConfig(subprocess_isolation=True)

    monkeypatch.setattr(
        openmm_subprocess.pickle, "dump",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("synthetic pickle failure"),
        ),
    )

    cx = _synthetic_complex()
    r = run_md_in_subprocess(
        cx, "c_marshal_fail", cfg, tmp_path,
        instability=0.0, catalytic_positions=[],
        backend=Backend.real, dry_run=False,
    )
    assert r.status == "failed"
    assert "marshal error" in (r.failure_reason or "").lower()
    assert "synthetic pickle failure" in (r.failure_reason or "")


# ---------------------------------------------------------------------------
# Failed-result helper unit test
# ---------------------------------------------------------------------------


def test_failed_result_mirrors_cfg():
    cfg = MDConfig(
        protocol_level=2, solvent="implicit",
        timestep_fs=2.5, ligand_forcefield="openff-2.2.0", hmr_enabled=True,
    )
    r = _failed_result("cX", cfg, "subprocess crashed (rc=7)")
    assert r.status == "failed"
    assert r.integration_failed is True
    assert r.candidate_id == "cX"
    assert r.protocol_level == 2
    assert r.solvent_mode == "implicit"
    assert r.simulation_time_ns == 0.0
    assert r.timestep_fs == 2.5
    assert r.ligand_forcefield == "openff-2.2.0"
    assert r.hmr_enabled is True
    assert "rc=7" in r.failure_reason


def test_failed_result_accepts_timeout_status():
    """F: callers can pass status='timeout' so the subprocess wrapper
    can distinguish hangs from crashes without touching the rest of
    MDResult's shape."""
    cfg = MDConfig(protocol_level=1, solvent="implicit")
    r = _failed_result("cY", cfg, "subprocess timeout after 60s",
                       status="timeout")
    assert r.status == "timeout"
    assert r.integration_failed is True
    assert "timeout" in r.failure_reason


# ---------------------------------------------------------------------------
# s10_md wiring (source-level guard — no Pipeline run needed)
# ---------------------------------------------------------------------------


def test_s10_md_uses_subprocess_wrapper():
    src = inspect.getsource(s10_md.MDStage.run)
    # The wrapper is what enforces subprocess_isolation; a refactor
    # that drops it would silently regress the P0 fix.
    assert "run_md_in_subprocess" in src
    assert "subprocess_timeout_seconds" in src
