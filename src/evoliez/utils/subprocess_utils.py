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
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        capture_output=True,
        text=True,
        timeout=timeout,
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
