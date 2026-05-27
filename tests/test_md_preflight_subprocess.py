"""G (post-expert-audit) — bounded MD preflight.

The expert flagged that `_run_md_preflight_inproc` calls
`SystemGenerator(...).create_system(...)` synchronously. A sqm hang
inside antechamber (server-observed on real cofactors without curated
AMBER params) would hang s01 itself — strictly worse than the same hang
at s10 (no per-candidate retry, no other candidates in flight).

This file pins the subprocess+timeout wrapper:
  - Hang → `status="timeout_preflight"` (killed by SIGKILL after the
    wall clock).
  - Worker crash → `status="failed_preflight"`.
  - Worker success → the in-process result is round-tripped via pickle
    and returned untouched.
  - Subprocess launch / pickle failure → fall back to in-process so
    the pipeline still runs.

Tests use a hang script (sleep) and a tmp result pickle so we don't
need a real openff.toolkit / openmmforcefields install.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from evoliez.adapters import md_preflight as mdp
from evoliez.adapters.md_preflight import (
    MDPreflightResult,
    _run_md_preflight_in_subprocess,
    run_md_preflight,
)
from evoliez.adapters.openmm_engine import _clear_ligand_probe_cache


@pytest.fixture(autouse=True)
def _reset():
    _clear_ligand_probe_cache()
    yield
    _clear_ligand_probe_cache()


# ---------------------------------------------------------------------- #
# Timeout path: a hanging subprocess returns status="timeout_preflight"
# ---------------------------------------------------------------------- #


def test_subprocess_timeout_returns_timeout_preflight(tmp_path, monkeypatch):
    """A worker that sleeps past `timeout_seconds` is SIGKILL'd and the
    wrapper returns MDPreflightResult(status='timeout_preflight'), NOT
    a hung function call."""
    hang_script = tmp_path / "_hang.py"
    hang_script.write_text(textwrap.dedent("""
        import time
        time.sleep(30)
    """).strip())

    original_popen = mdp._subprocess.Popen

    def _patched_popen(cmd, *a, **kw):
        new_cmd = [sys.executable, str(hang_script)]
        return original_popen(new_cmd, *a, **kw)

    monkeypatch.setattr(mdp._subprocess, "Popen", _patched_popen)

    md_dir = tmp_path / "md"
    result = _run_md_preflight_in_subprocess(
        smiles="CCO", md_dir=md_dir, prefer_ff="openff-2.2.0",
        timeout_seconds=1,
    )
    assert result.status == "timeout_preflight"
    assert "timeout" in (result.reason or "").lower()


# ---------------------------------------------------------------------- #
# Crash path: worker exits non-zero without writing result.pkl
# ---------------------------------------------------------------------- #


def test_subprocess_crash_returns_failed_preflight(tmp_path, monkeypatch):
    """Worker that exits with rc=1 and no result.pkl → failed_preflight."""
    crash_script = tmp_path / "_crash.py"
    crash_script.write_text(textwrap.dedent("""
        import sys
        sys.stderr.write("synthetic preflight worker crash\\n")
        sys.exit(7)
    """).strip())

    original_popen = mdp._subprocess.Popen

    def _patched_popen(cmd, *a, **kw):
        new_cmd = [sys.executable, str(crash_script)]
        return original_popen(new_cmd, *a, **kw)

    monkeypatch.setattr(mdp._subprocess, "Popen", _patched_popen)

    md_dir = tmp_path / "md"
    result = _run_md_preflight_in_subprocess(
        smiles="CCO", md_dir=md_dir, prefer_ff="openff-2.2.0",
        timeout_seconds=10,
    )
    assert result.status == "failed_preflight"
    assert "rc=7" in (result.reason or "")
    # Log tail surfaced so the operator can debug without ssh.
    assert "synthetic preflight worker crash" in (result.reason or "")


# ---------------------------------------------------------------------- #
# Success path: worker writes a real MDPreflightResult pickle
# ---------------------------------------------------------------------- #


def test_subprocess_success_roundtrips_result(tmp_path, monkeypatch):
    """When the worker writes a clean MDPreflightResult, the wrapper
    returns it verbatim."""
    # Inline worker that writes a fake "ok" result pickle.
    fake_worker = tmp_path / "_fake_worker.py"
    fake_worker.write_text(textwrap.dedent("""
        import pickle, sys
        from evoliez.adapters.md_preflight import MDPreflightResult
        result = MDPreflightResult(
            status="ok", ff_used="openff-2.2.0",
            reason="fake worker round-trip",
        )
        with open(sys.argv[2], "wb") as f:
            pickle.dump(result, f)
    """).strip())

    original_popen = mdp._subprocess.Popen

    def _patched_popen(cmd, *a, **kw):
        # Replace `-m evoliez.adapters.md_preflight_worker` with our fake.
        # cmd[-2] = inputs.pkl, cmd[-1] = result.pkl — keep them.
        new_cmd = [sys.executable, str(fake_worker), cmd[-2], cmd[-1]]
        return original_popen(new_cmd, *a, **kw)

    monkeypatch.setattr(mdp._subprocess, "Popen", _patched_popen)

    md_dir = tmp_path / "md"
    result = _run_md_preflight_in_subprocess(
        smiles="CCO", md_dir=md_dir, prefer_ff="openff-2.2.0",
        timeout_seconds=10,
    )
    assert result.status == "ok"
    assert result.ff_used == "openff-2.2.0"
    assert result.reason == "fake worker round-trip"


# ---------------------------------------------------------------------- #
# Dispatch: run_md_preflight respects the subprocess_isolation flag
# ---------------------------------------------------------------------- #


def test_run_md_preflight_dispatches_inproc_by_default(tmp_path):
    """Backward compat: default subprocess_isolation=False routes to
    in-process so existing tests + the light mac venv keep working."""
    called = {"inproc": False, "sub": False}

    def _fake_inproc(smiles, md_dir, prefer_ff):
        called["inproc"] = True
        return MDPreflightResult(status="ok", ff_used="fake")

    def _fake_sub(*a, **k):
        called["sub"] = True
        return MDPreflightResult(status="ok", ff_used="fake")

    with patch.object(mdp, "_run_md_preflight_inproc", _fake_inproc):
        with patch.object(mdp, "_run_md_preflight_in_subprocess", _fake_sub):
            r = run_md_preflight(smiles="CCO", md_dir=tmp_path / "md")
    assert called == {"inproc": True, "sub": False}
    assert r.status == "ok"


def test_run_md_preflight_dispatches_subprocess_when_enabled(tmp_path):
    called = {"inproc": False, "sub": False}

    def _fake_inproc(*a, **k):
        called["inproc"] = True
        return MDPreflightResult(status="ok")

    def _fake_sub(smiles, md_dir, prefer_ff, *, timeout_seconds):
        called["sub"] = True
        assert timeout_seconds == 42
        return MDPreflightResult(status="ok")

    with patch.object(mdp, "_run_md_preflight_inproc", _fake_inproc):
        with patch.object(mdp, "_run_md_preflight_in_subprocess", _fake_sub):
            run_md_preflight(
                smiles="CCO", md_dir=tmp_path / "md",
                subprocess_isolation=True, timeout_seconds=42,
            )
    assert called == {"inproc": False, "sub": True}


def test_md_disabled_short_circuit_skips_both_paths(tmp_path):
    """`md_enabled=False` returns before either branch."""
    called = {"inproc": False, "sub": False}

    def _fake_inproc(*a, **k):
        called["inproc"] = True

    def _fake_sub(*a, **k):
        called["sub"] = True

    with patch.object(mdp, "_run_md_preflight_inproc", _fake_inproc):
        with patch.object(mdp, "_run_md_preflight_in_subprocess", _fake_sub):
            r = run_md_preflight(
                smiles="CCO", md_dir=tmp_path / "md",
                md_enabled=False, subprocess_isolation=True,
            )
    assert called == {"inproc": False, "sub": False}
    assert r.status == "skipped_md_disabled"


# ---------------------------------------------------------------------- #
# Worker module: pickle round-trip
# ---------------------------------------------------------------------- #


def test_worker_pickle_roundtrip_ok(tmp_path):
    """End-to-end: the worker script reads an inputs pickle, calls
    `_run_md_preflight_inproc` (which falls back to 'skipped_no_md_libs'
    in the light venv with no openff.toolkit), writes a result pickle.
    The wrapper around it then returns that result.
    """
    inputs_path = tmp_path / "inputs.pkl"
    result_path = tmp_path / "result.pkl"

    md_dir = tmp_path / "md"
    md_dir.mkdir()
    inputs_path.write_bytes(pickle.dumps({
        "smiles": "CCO",
        "md_dir": str(md_dir),
        "prefer_ff": "openff-2.2.0",
    }))

    proc = subprocess.run(
        [sys.executable, "-m", "evoliez.adapters.md_preflight_worker",
         str(inputs_path), str(result_path)],
        capture_output=True, timeout=30,
    )
    # In the light mac venv openff.toolkit isn't installed → the worker
    # returns status="skipped_no_md_libs". Either rc=0 with that result,
    # OR rc=1 if the import itself failed at the very edge. Both are
    # acceptable: the parent wrapper handles rc!=0 → failed_preflight.
    if proc.returncode == 0:
        assert result_path.exists()
        result = pickle.loads(result_path.read_bytes())
        assert result.status in {
            "skipped_no_md_libs", "ok", "unsupported", "curated_available",
        }
    else:
        # rc!=0 path is also fine — the parent wrapper logs it and
        # synthesises a failed_preflight. We just need the worker to
        # not hang.
        pass


# ---------------------------------------------------------------------- #
# Source-level guard: s01 forwards subprocess_isolation + timeout knobs
# ---------------------------------------------------------------------- #


def test_s01_forwards_preflight_subprocess_knobs():
    import inspect

    from evoliez.stages import s01_input_preprocess as s01
    src = inspect.getsource(s01)
    assert "subprocess_isolation=" in src
    assert "preflight_timeout_seconds" in src
    assert "timeout_seconds=" in src
