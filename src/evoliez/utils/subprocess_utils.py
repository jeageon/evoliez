"""Safe subprocess wrapper used by every ``real`` adapter.

Centralises logging, timeouts, dry-run, and ``executable not found`` handling so
adapters never call ``subprocess`` directly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from evoliez.logging_utils import get_logger

log = get_logger("evoliez.exec")

# Bind every child's lifetime to the parent (evoliez): when the parent dies -- killed
# (pkill), OOM, or crashed -- the kernel SIGKILLs the child too, so a stopped/crashed run
# never leaves an ORPHANED boltz/gnina/diffdock subprocess hung on a GPU. (A real bug: an
# s08b `boltz predict` child whose parent was killed reparented to init and held a GPU for
# 23 h.) Linux-only via prctl(PR_SET_PDEATHSIG). libc is loaded ONCE here, and the
# post-fork hook does nothing but a single allocation-free syscall (no imports / no Python
# locks), so it is safe even if the parent has threads at fork time.
try:
    import ctypes as _ctypes
    import signal as _signal

    _LIBC = _ctypes.CDLL("libc.so.6", use_errno=True)
    _SIGKILL = int(_signal.SIGKILL)
except Exception:                                    # non-Linux / no libc -> no-op
    _LIBC = None


def _set_pdeathsig() -> None:                        # runs in the child, between fork+exec
    if _LIBC is not None:
        _LIBC.prctl(1, _SIGKILL)                     # PR_SET_PDEATHSIG = 1


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


def tool_env(binexe) -> Optional[dict]:
    """Env for running a binary that lives in a *different* conda env than the
    one the pipeline runs in: prepend that env's ``lib`` to LD_LIBRARY_PATH so
    its shared libs (libopenblas, …) load. Returns None for a bare name (on
    PATH, same env) or when no sibling ``lib`` exists, so callers can pass the
    result straight to ``run(env=...)`` (None -> inherit the current env)."""
    p = Path(str(binexe))
    if p.parent in (Path("."), Path("")):     # bare name -> on PATH, same env
        return None
    lib = p.parent.parent / "lib"
    if not lib.is_dir():
        return None
    env = dict(os.environ)
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{lib}:{prev}" if prev else str(lib)
    return env


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
    # Honour the pipeline-wide EVOLIEZ_DRY_RUN env var as well as the explicit
    # kwarg: the orchestrator signals dry-run via the env var, so an adapter
    # that forgets to thread dry_run= must still NOT execute a real command.
    dry = dry_run or os.environ.get("EVOLIEZ_DRY_RUN") == "1"
    log.info("%sexec: %s", "[dry-run] " if dry else "", " ".join(cmd))
    if dry:
        return RunResult(cmd=cmd, returncode=0, stdout="", stderr="", dry_run=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        # die with the parent (Linux) so a killed/crashed run leaves no GPU-holding orphan
        preexec_fn=_set_pdeathsig if _LIBC is not None else None,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    return RunResult(
        cmd=cmd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
