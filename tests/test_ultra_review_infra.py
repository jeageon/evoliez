"""Ultra-review P0/P1 infra fixes (no real GPU tools required).

Covers three central-infra hardening changes:

  * ``subprocess_utils.run`` now launches in its own session and on
    timeout reaps the *whole* process group (so leaked GPU workers from
    DiffDock / LigandMPNN don't survive a kill) - and still raises
    ``TimeoutExpired`` exactly like ``subprocess.run(timeout=...)`` did,
    so existing catchers keep working.
  * ``diagnostics`` actually executes a cheap ``--version``/``--help`` for
    load-bearing tools, so a present-but-broken binary WARNs instead of
    falsely reporting ``[OK]``.
  * ``db.store`` opens SQLite in WAL mode with a busy_timeout so two
    stages (or a concurrent figures run) don't hit "database is locked".
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from evoliez import diagnostics
from evoliez.utils import subprocess_utils
from evoliez.utils.subprocess_utils import RunResult, run


# ---------------------------------------------------------------------------
# subprocess_utils.run - timeout reaps the process group, raises as before
# ---------------------------------------------------------------------------


def test_run_quick_command_still_works():
    """The Popen path must not regress an ordinary fast command: output is
    captured, returncode is 0, return shape is unchanged."""
    res = run([sys.executable, "-c", "print('hello'); import sys; sys.stderr.write('werr')"])
    assert isinstance(res, RunResult)
    assert res.returncode == 0
    assert "hello" in res.stdout
    assert "werr" in res.stderr
    assert res.dry_run is False


def test_run_dry_run_short_circuits(monkeypatch):
    """dry_run must never spawn a process (semantics unchanged)."""
    monkeypatch.setattr(
        subprocess_utils.subprocess, "Popen",
        lambda *a, **k: pytest.fail("Popen called on dry_run"),
    )
    res = run([sys.executable, "-c", "print(1)"], dry_run=True)
    assert res.dry_run is True
    assert res.returncode == 0


def test_run_nonzero_raises_when_check(monkeypatch):
    """check=True still raises RuntimeError carrying stderr (unchanged)."""
    with pytest.raises(RuntimeError) as ei:
        run([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    assert "boom" in str(ei.value)


def test_run_nonzero_no_check_returns_result():
    """check=False returns the RunResult with the nonzero code (unchanged)."""
    res = run([sys.executable, "-c", "import sys; sys.exit(7)"], check=False)
    assert res.returncode == 7


def test_run_timeout_raises_and_does_not_hang():
    """A command that exceeds the timeout must raise TimeoutExpired promptly
    (well within timeout + the SIGTERM grace), not block forever."""
    timeout = 0.5
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=timeout)
    elapsed = time.monotonic() - t0
    # timeout + (term grace) + generous CI slack. Must NOT approach 30s.
    assert elapsed < timeout + subprocess_utils._TERM_GRACE_S + 5.0


def test_run_timeout_kills_whole_process_group():
    """The parent forks a child that outlives a naive parent-only kill.
    After our timeout reaps the *group*, the grandchild must be gone too -
    this is the VRAM-leak fix for DiffDock/LigandMPNN forked workers.

    The child writes its own PID to a file, then both parent and child
    sleep. We confirm the grandchild PID is dead after run() times out.
    """
    import os
    import tempfile

    pidfile = tempfile.NamedTemporaryFile(
        prefix="evz_grandchild_", suffix=".pid", delete=False,
    )
    pidfile.close()

    # Parent python forks a child that records its PID and sleeps long.
    # start_new_session=True (inside run) puts parent+child in one group.
    script = (
        "import os, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        f"    open({pidfile.name!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "else:\n"
        "    time.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run([sys.executable, "-c", script], timeout=1.5)

    # Give the kernel a beat to deliver SIGKILL to the group.
    grandchild_pid = None
    for _ in range(50):
        txt = open(pidfile.name).read().strip()
        if txt:
            grandchild_pid = int(txt)
            break
        time.sleep(0.1)
    assert grandchild_pid is not None, "child never recorded its PID"

    # Poll until the grandchild is reaped (it was SIGKILLed via the group).
    alive = True
    for _ in range(50):
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            alive = False
            break
        except PermissionError:  # pragma: no cover - exists but not ours
            alive = True
            break
        time.sleep(0.1)
    assert not alive, (
        f"grandchild {grandchild_pid} survived the timeout kill - "
        "process-group reaping regressed (would leak GPU VRAM)"
    )
    try:
        os.unlink(pidfile.name)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# diagnostics - present-but-broken tool WARNs (not OK)
# ---------------------------------------------------------------------------


def test_doctor_tool_version_nonzero_warns(monkeypatch):
    """A load-bearing tool that is on PATH but whose --version exits non-zero
    must report WARN, not the old false [OK]."""
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: f"/fake/bin/{name}")

    def _fake_probe(path, args):
        return False, "present but '--version' exited 1"

    monkeypatch.setattr(diagnostics, "_probe_tool", _fake_probe)
    rep = diagnostics.collect()
    # boltz is in the probe set -> should be WARN now.
    boltz = [c for c in rep.checks if c.name == "tool:boltz"]
    assert boltz and boltz[0].status == diagnostics.WARN
    assert "exited 1" in boltz[0].detail


def test_doctor_tool_healthy_probe_is_ok(monkeypatch):
    """Happy path: present and probe succeeds -> OK (no false WARN)."""
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: f"/fake/bin/{name}")
    monkeypatch.setattr(diagnostics, "_probe_tool", lambda p, a: (True, ""))
    rep = diagnostics.collect()
    vina = [c for c in rep.checks if c.name == "tool:vina"]
    assert vina and vina[0].status == diagnostics.OK


def test_doctor_tool_missing_stays_missing(monkeypatch):
    """Not on PATH -> MISSING (probe never called)."""
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        diagnostics, "_probe_tool",
        lambda p, a: pytest.fail("probe should not run for a missing tool"),
    )
    rep = diagnostics.collect()
    gnina = [c for c in rep.checks if c.name == "tool:gnina"]
    assert gnina and gnina[0].status == diagnostics.MISSING


def test_doctor_which_only_tool_not_probed(monkeypatch):
    """foldx / iupred2a.py have no cheap version flag: present => OK via
    which alone, probe must not be invoked for them."""
    monkeypatch.setattr(diagnostics.shutil, "which",
                        lambda name: f"/fake/bin/{name}")
    probed = []
    monkeypatch.setattr(
        diagnostics, "_probe_tool",
        lambda p, a: (probed.append(p), (True, ""))[1],
    )
    rep = diagnostics.collect()
    foldx = [c for c in rep.checks if c.name == "tool:foldx"]
    assert foldx and foldx[0].status == diagnostics.OK
    assert not any("foldx" in p for p in probed)
    assert not any("iupred2a" in p for p in probed)


def test_probe_tool_timeout_returns_unhealthy_not_raise(monkeypatch):
    """A hung --version must downgrade to unhealthy (WARN), never crash
    doctor with a raised TimeoutExpired."""
    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd=a[0] if a else "x", timeout=3)

    monkeypatch.setattr(diagnostics.subprocess, "run", _raise_timeout)
    healthy, why = diagnostics._probe_tool("/fake/bin/boltz", ["--help"])
    assert healthy is False
    assert "timed out" in why


def test_probe_tool_oserror_returns_unhealthy(monkeypatch):
    """A binary that can't even exec (ENOEXEC / missing .so) -> unhealthy,
    no exception escapes."""
    def _raise_oserror(*a, **k):
        raise OSError("Exec format error")

    monkeypatch.setattr(diagnostics.subprocess, "run", _raise_oserror)
    healthy, why = diagnostics._probe_tool("/fake/bin/vina", ["--version"])
    assert healthy is False
    assert "failed to execute" in why


def test_doctor_full_run_never_raises_and_statuses_valid():
    """Real environment smoke test: collect() runs (probing whatever tools
    happen to be installed locally) and every status is a known constant."""
    rep = diagnostics.collect()
    valid = {diagnostics.OK, diagnostics.WARN,
             diagnostics.MISSING, diagnostics.BLOCK}
    for c in rep.checks:
        assert c.status in valid
    assert any(c.name.startswith("tool:") for c in rep.checks)


# ---------------------------------------------------------------------------
# db.store - WAL + busy_timeout applied on connect
# ---------------------------------------------------------------------------


def test_store_enables_wal_and_busy_timeout(tmp_path):
    """Opening a Store must put the SQLite file in WAL mode with a non-zero
    busy_timeout, so concurrent stage writes wait instead of erroring."""
    from sqlalchemy import text

    try:
        from evoliez.db.store import Store
        store = Store(tmp_path / "evoliez.sqlite")
    except Exception as exc:  # pragma: no cover - tmp fs can't host sqlite
        pytest.skip(f"cannot create sqlite in tmp: {exc}")

    with store.session() as s:
        mode = s.execute(text("PRAGMA journal_mode")).scalar()
        busy = s.execute(text("PRAGMA busy_timeout")).scalar()
    assert str(mode).lower() == "wal", f"expected WAL, got {mode!r}"
    assert int(busy) >= 30000, f"busy_timeout not applied, got {busy!r}"


def test_store_connect_event_registered(tmp_path):
    """The connect-time PRAGMA hook must be wired on the engine (guards a
    refactor that drops the event listener)."""
    from sqlalchemy import event

    try:
        from evoliez.db.store import Store, _enable_sqlite_concurrency
        store = Store(tmp_path / "evoliez.sqlite")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"cannot create sqlite in tmp: {exc}")

    assert event.contains(store.engine, "connect", _enable_sqlite_concurrency)
