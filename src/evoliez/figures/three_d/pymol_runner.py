"""Headless-PyMOL subprocess wrapper.

We intentionally call PyMOL via ``pymol -cq /dev/stdin`` rather than
``import pymol`` to keep the (sometimes GPL) PyMOL bindings out of the
EvoLiEZ Python process.  Both helpers below are safe to call when PyMOL is
not installed: ``pymol_available`` returns ``False`` and
``run_pymol_script`` returns ``(False, "pymol not on PATH")``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Tuple

_LOGGER = logging.getLogger(__name__)


def _which_pymol() -> str:
    """Return the resolved PyMOL CLI path, or ``""`` if not found."""
    # Most installs (conda, brew, schrodinger) put a ``pymol`` shim on PATH.
    found = shutil.which("pymol")
    if found:
        return found
    # macOS Schrodinger PyMOL ships under /Applications/PyMOL.app but its
    # CLI is also exposed as ``pymolWizard`` or ``pymolcli`` on some
    # distributions; we accept either as a best-effort fallback.
    for alt in ("pymolcli", "PyMOL"):
        found = shutil.which(alt)
        if found:
            return found
    return ""


def pymol_available() -> bool:
    """True iff a PyMOL CLI binary is on PATH."""
    return bool(_which_pymol())


def run_pymol_script(script: str, *, timeout: float = 120) -> Tuple[bool, str]:
    """Run ``script`` through ``pymol -cq``.

    Returns ``(ok, output)`` where ``output`` is stdout on success or a
    short error message on failure.  Never raises - any subprocess /
    timeout error is caught and surfaced via the ``ok=False`` branch.
    """
    binary = _which_pymol()
    if not binary:
        return (False, "pymol not on PATH")

    cmd = [binary, "-cq", "/dev/stdin"]
    try:
        proc = subprocess.run(
            cmd,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (False, f"pymol timed out after {timeout}s")
    except (OSError, ValueError) as exc:
        _LOGGER.warning("pymol subprocess failed to launch: %s", exc)
        return (False, f"pymol launch failed: {exc}")

    if proc.returncode != 0:
        # PyMOL prints fatal errors to stderr but also writes recoverable
        # warnings there, so we surface stderr only on non-zero exit.
        err = (proc.stderr or proc.stdout or "").strip()
        return (False, err or f"pymol exited with code {proc.returncode}")

    return (True, proc.stdout or "")
