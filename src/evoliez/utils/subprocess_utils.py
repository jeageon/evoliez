"""Safe subprocess wrapper used by every ``real`` adapter.

Centralises logging, timeouts, dry-run, and ``executable not found`` handling so
adapters never call ``subprocess`` directly.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from evoliez.logging_utils import get_logger

log = get_logger("evoliez.exec")

# Grace period (seconds) between SIGTERM and SIGKILL when reaping a timed-out
# process group. Short, because the immediate child is already presumed hung;
# this only gives well-behaved children a chance to flush before the hammer.
_TERM_GRACE_S = 5.0


class ToolNotFoundError(RuntimeError):
    """Raised when a ``real`` backend tool is not on PATH."""


@dataclass
class RunResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    dry_run: bool = False


def which(executable: str) -> Optional[str]:
    return shutil.which(executable)


def require(executable: str) -> str:
    path = which(executable)
    if path is None:
        if os.environ.get("EVOLIEZ_DRY_RUN") == "1":
            log.warning("[dry-run] '%s' not on PATH (would be required)", executable)
            return executable
        raise ToolNotFoundError(
            f"'{executable}' not found on PATH. Install it on the server "
            f"(see docs/DATA_AND_WEIGHTS.md) or run this stage with backend=mock."
        )
    return path


def run(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    dry_run: bool = False,
    check: bool = True,
) -> RunResult:
    cmd = [str(c) for c in cmd]
    log.info("exec: %s", " ".join(cmd))
    if dry_run:
        return RunResult(cmd=cmd, returncode=0, stdout="", stderr="", dry_run=True)

    # Launch in its own session/process group so a timeout can reap the whole
    # tree, not just the immediate child. GPU tools (DiffDock / LigandMPNN /
    # gnina) fork CUDA workers that would otherwise leak VRAM on the shared
    # box when only the parent is killed by subprocess.run(timeout=...).
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill the entire process group, then drain whatever output was
        # buffered so the TimeoutExpired we re-raise carries it (matching
        # subprocess.run(timeout=...) semantics for downstream catchers).
        log.warning(
            "exec: timeout after %ss, killing process group: %s",
            timeout, " ".join(cmd),
        )
        _kill_process_group(proc)
        stdout, stderr = _drain(proc)
        raise subprocess.TimeoutExpired(
            cmd, timeout, output=stdout, stderr=stderr,
        )
    except BaseException:
        # Any other interruption (e.g. KeyboardInterrupt) must not leave an
        # orphaned process group holding a GPU.
        _kill_process_group(proc)
        _drain(proc)
        raise

    returncode = proc.returncode
    if check and returncode != 0:
        raise RuntimeError(
            f"command failed ({returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{(stderr or '')[-4000:]}"
        )
    return RunResult(
        cmd=cmd,
        returncode=returncode,
        stdout=stdout or "",
        stderr=stderr or "",
    )


def _kill_process_group(proc: "subprocess.Popen") -> None:
    """SIGTERM then (after a grace period) SIGKILL the whole process group.

    Started via ``start_new_session=True`` so ``os.getpgid(proc.pid)`` is the
    group leader; killing the group takes down forked GPU workers too.
    No-op if the process already exited or we lack permission.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=_TERM_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=_TERM_GRACE_S)
    except subprocess.TimeoutExpired:  # pragma: no cover - kernel didn't reap
        pass


def _drain(proc: "subprocess.Popen") -> tuple[str, str]:
    """Best-effort read of any buffered output after the process is dead.

    The pipes are still open; ``communicate`` with a tiny timeout collects
    what's there without blocking forever on a wedged FD.
    """
    try:
        out, err = proc.communicate(timeout=_TERM_GRACE_S)
        return out or "", err or ""
    except Exception:  # pragma: no cover - defensive; pipes may be gone
        return "", ""
