"""Per-candidate MD subprocess isolation (open-P0 from the expert audit).

The in-process OpenMM cleanup (commits 3d840bf / 99488a2 / 8bbf44d)
gets RSS drift down to 0 in the local 30-iter stress test, but
production hit a 166 GB RSS leak that took the multi-hour run with it.
The expert validation plan calls for **subprocess isolation per
candidate** as the strong containment:

  - one fresh Python interpreter per candidate,
  - OS-level cleanup of OpenMM / CUDA contexts when the process exits,
  - hard wall-clock timeout per candidate (default 30 min) so a hang
    or runaway minimisation can't stall the entire stage,
  - graceful degradation: on timeout / crash, return a synthetic
    ``MDResult(status="failed", integration_failed=True,
    failure_reason="subprocess <X>")`` and continue with the next
    candidate.

Design choice — pickle for marshalling: the worker IS the same
codebase and the same install. We just want process isolation, not
serialisation portability. Pickle keeps the worker contract trivial
("call run_md with the unpickled inputs"); switching to PDB/SDF on
disk would mean duplicating the Complex round-tripping logic.

The legacy in-process path is preserved (``run_md_in_subprocess``
falls back to ``run_md`` when ``subprocess_isolation`` is off in
config), so existing tests that exercise the mock backend in-process
don't change behaviour.
"""

from __future__ import annotations

import logging
import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from evoliez.adapters.openmm_engine import MDResult, run_md
from evoliez.config import Backend, MDConfig
from evoliez.types import Complex

log = logging.getLogger("evoliez.openmm.subprocess")

# Pickle filenames inside the candidate's workdir. Kept under "_md_"
# prefix so an `ls` of the workdir shows them next to each other but
# they don't collide with the user-visible artefacts (analysis.json,
# trajectory.dcd, ...).
_INPUTS_PKL = "_md_subprocess_inputs.pkl"
_RESULT_PKL = "_md_subprocess_result.pkl"
_WORKER_LOG = "_md_subprocess.log"


def _failed_result(
    candidate_id: str, cfg: MDConfig, reason: str,
) -> MDResult:
    """Synthetic MDResult for the timeout / crash paths. Matches what
    ``run_md`` itself returns on a per-candidate exception, so the
    downstream analyser doesn't need a separate code path."""
    return MDResult(
        candidate_id=candidate_id,
        status="failed",
        protocol_level=cfg.protocol_level,
        solvent_mode=cfg.solvent,
        simulation_time_ns=0.0,
        integration_failed=True,
        failure_reason=reason,
        timestep_fs=float(cfg.timestep_fs),
        ligand_forcefield=cfg.ligand_forcefield,
        hmr_enabled=bool(getattr(cfg, "hmr_enabled", False)),
    )


def run_md_in_subprocess(
    cx: Complex,
    candidate_id: str,
    cfg: MDConfig,
    workdir: Path,
    *,
    instability: float,
    catalytic_positions: Sequence[int],
    backend: Backend,
    dry_run: bool = False,
    timeout_seconds: int = 1800,
) -> MDResult:
    """Run MD for one candidate in an isolated subprocess.

    Returns the same :class:`MDResult` an in-process ``run_md`` would
    return, plus three new failure modes captured as
    ``status="failed"``:

      * ``subprocess timeout``     - wall clock exceeded ``timeout_seconds``
      * ``subprocess crashed``     - non-zero exit / no result pickle
      * ``subprocess marshal error`` - pickling input/output failed

    Falls back to the in-process ``run_md`` when ``subprocess_isolation``
    is disabled in config, or when the backend isn't ``real`` (the
    mock backend's per-candidate cost isn't worth a fresh process).
    """
    use_subprocess = bool(
        getattr(cfg, "subprocess_isolation", False)
    ) and backend is Backend.real

    if not use_subprocess:
        return run_md(
            cx, candidate_id, cfg, workdir,
            instability=instability,
            catalytic_positions=catalytic_positions,
            backend=backend, dry_run=dry_run,
        )

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    inputs_path = workdir / _INPUTS_PKL
    result_path = workdir / _RESULT_PKL
    log_path = workdir / _WORKER_LOG

    # Clean up any stale pickles from a prior aborted run.
    for p in (inputs_path, result_path):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    try:
        with inputs_path.open("wb") as fh:
            pickle.dump(
                {
                    "complex": cx,
                    "candidate_id": candidate_id,
                    "cfg": cfg,
                    "workdir": str(workdir),
                    "instability": float(instability),
                    "catalytic_positions": list(catalytic_positions),
                    "backend": backend.value if hasattr(backend, "value") else str(backend),
                    "dry_run": bool(dry_run),
                },
                fh,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("MD subprocess: failed to pickle inputs for %s: %s",
                    candidate_id, exc)
        return _failed_result(
            candidate_id, cfg, f"subprocess marshal error: {exc}",
        )

    cmd = [
        sys.executable, "-u",
        "-m", "evoliez.adapters.openmm_subprocess_worker",
        str(inputs_path), str(result_path),
    ]
    log.info(
        "MD subprocess: launching for %s (timeout=%ds, workdir=%s)",
        candidate_id, timeout_seconds, workdir,
    )
    start = time.monotonic()
    proc = None
    try:
        # start_new_session=True so a SIGKILL hits the whole process
        # group (OpenMM + any tleap/sqm subprocesses it spawned).
        with log_path.open("wb") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=logf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            log.warning(
                "MD subprocess: %s exceeded timeout=%ds; killing process group",
                candidate_id, timeout_seconds,
            )
            _kill_process_group(proc)
            return _failed_result(
                candidate_id, cfg,
                f"subprocess timeout after {timeout_seconds}s",
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("MD subprocess: launch failed for %s: %s",
                    candidate_id, exc)
        if proc is not None:
            _kill_process_group(proc)
        return _failed_result(
            candidate_id, cfg, f"subprocess launch failed: {exc}",
        )

    elapsed = time.monotonic() - start
    rc = proc.returncode if proc is not None else -1
    if rc != 0 or not result_path.exists():
        # Surface the tail of the worker log so the failure_reason
        # actually tells the operator what went wrong (no need to ssh
        # in and tail -f the per-candidate log to debug a crash).
        tail = _read_log_tail(log_path)
        log.warning(
            "MD subprocess: %s crashed (rc=%d, %.1fs); tail=%s",
            candidate_id, rc, elapsed, tail[:200],
        )
        return _failed_result(
            candidate_id, cfg,
            f"subprocess crashed (rc={rc}) after {elapsed:.1f}s: {tail[-400:]}",
        )

    try:
        with result_path.open("rb") as fh:
            result: MDResult = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        log.warning("MD subprocess: failed to unpickle result for %s: %s",
                    candidate_id, exc)
        return _failed_result(
            candidate_id, cfg, f"subprocess marshal error: {exc}",
        )

    log.info(
        "MD subprocess: %s finished status=%s in %.1fs (rc=%d)",
        candidate_id, result.status, elapsed, rc,
    )
    return result


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the entire process group started by
    ``start_new_session=True``."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    # Give SIGTERM 5 s to clean up before forcing.
    for _ in range(50):
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _read_log_tail(log_path: Path, max_bytes: int = 4096) -> str:
    try:
        size = log_path.stat().st_size
        with log_path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
