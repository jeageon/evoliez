"""Subprocess entry-point for isolated per-candidate MD runs.

Invoked by :mod:`evoliez.adapters.openmm_subprocess` as

    python -u -m evoliez.adapters.openmm_subprocess_worker <inputs.pkl> <result.pkl>

Reads the pickled inputs, calls the existing in-process ``run_md`` with
them, pickles the :class:`MDResult` back to disk. Exit code 0 on
successful pickle write, non-zero on any exception. The parent process
(``run_md_in_subprocess``) treats a missing result pickle or a non-zero
exit as a subprocess crash and synthesises a graceful
``MDResult(status="failed", ...)``.

Kept deliberately small: every line is a potential crash site, and we
want the parent to see the crash, not us to catch it and mask it.
"""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path


def _coerce_backend(value):
    """The parent serialises ``Backend`` as a string (its .value)."""
    from evoliez.config import Backend
    if isinstance(value, Backend):
        return value
    return Backend(str(value))


def main(inputs_path: str, result_path: str) -> int:
    from evoliez.adapters.openmm_engine import run_md

    with Path(inputs_path).open("rb") as fh:
        payload = pickle.load(fh)

    result = run_md(
        payload["complex"],
        payload["candidate_id"],
        payload["cfg"],
        Path(payload["workdir"]),
        instability=float(payload["instability"]),
        catalytic_positions=list(payload["catalytic_positions"]),
        backend=_coerce_backend(payload["backend"]),
        dry_run=bool(payload.get("dry_run", False)),
    )

    with Path(result_path).open("wb") as fh:
        pickle.dump(result, fh)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write(
            "usage: python -m evoliez.adapters.openmm_subprocess_worker "
            "<inputs.pkl> <result.pkl>\n"
        )
        sys.exit(2)
    try:
        sys.exit(main(sys.argv[1], sys.argv[2]))
    except SystemExit:
        raise
    except BaseException:
        # Print to stderr (caught into the parent's worker log) AND
        # exit non-zero so the parent treats this as a crash, not a
        # silent success-with-empty-result.
        traceback.print_exc()
        sys.exit(1)
