"""Subprocess worker for G — bounded MD preflight.

Mirrors `openmm_subprocess_worker` but for the lightweight ligand-only
parameterisation probe. Takes a pickled ``{"smiles", "md_dir",
"prefer_ff"}`` dict, runs `_run_md_preflight_inproc`, writes the
``MDPreflightResult`` pickle to the second argv.

Why a separate worker module? `md_preflight.py` imports `openff.toolkit`
inline (so the light mac venv doesn't crash on module import). The
worker runs in the production server env where those imports succeed,
and a hang in `SystemGenerator.create_system` here is bounded by the
parent process's wall clock — not by hope.

Invocation:

    python -m evoliez.adapters.md_preflight_worker <inputs.pkl> <result.pkl>

Exit codes:

  - 0: result.pkl written successfully (even when the probe returns
       status="unsupported"; that's a normal outcome, not a worker
       failure).
  - 1: result.pkl could NOT be written (worker itself crashed). The
       parent reads rc != 0 and synthesises a failed_preflight result.
"""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(
            "usage: md_preflight_worker <inputs.pkl> <result.pkl>",
            file=sys.stderr,
        )
        return 1

    inputs_path = Path(argv[1])
    result_path = Path(argv[2])

    try:
        with inputs_path.open("rb") as fh:
            inputs = pickle.load(fh)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1

    smiles = inputs.get("smiles", "")
    md_dir = Path(inputs.get("md_dir", "."))
    prefer_ff = inputs.get("prefer_ff")

    try:
        # Defer the heavy import until inside the worker — keeps the
        # parent's `import evoliez.adapters.md_preflight` cheap and
        # tolerant of light mac envs without openmmforcefields.
        from evoliez.adapters.md_preflight import _run_md_preflight_inproc

        result = _run_md_preflight_inproc(smiles, md_dir, prefer_ff)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1

    try:
        with result_path.open("wb") as fh:
            pickle.dump(result, fh)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
