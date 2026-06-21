#!/usr/bin/env python
"""Purge-safe --resume pre-flight: compare the CURRENT run_fingerprint to a run's
stored ``_state.json`` fingerprint WITHOUT calling ``setup()``/``_load_state`` (so
it can NEVER trigger ``_purge_stale_outputs`` — the rmtree that wipes complexes/
docking/interaction_graphs/ml_datasets on a fingerprint change). Run this before
any ``evoliez run --resume`` to confirm the resume will reuse, not purge.

Exit 0 = fingerprint MATCH (resume is safe).
Exit 1 = MISMATCH (a bare --resume WOULD purge — investigate / patch _state.json first).
Exit 2 = error (no run / unreadable state).

Usage: python scripts/check_run_fingerprint.py <config.yaml> [run_dir]
       (run_dir defaults to the config's project.output_dir)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv) -> int:
    if not argv:
        print("usage: check_run_fingerprint.py <config.yaml> [run_dir]")
        return 2
    from evoliez.config import load_config
    from evoliez.context import RunContext

    cfg = load_config(argv[0])
    ctx = RunContext(cfg)          # __init__ ONLY — no setup(), no _load_state, no purge
    cur = ctx.run_fingerprint()
    run_dir = Path(argv[1]) if len(argv) > 1 else ctx.root
    state_path = run_dir / "_state.json"
    if not state_path.exists():
        print(f"no _state.json at {state_path} — this would be a FRESH run (nothing to purge).")
        return 0
    try:
        st = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR reading {state_path}: {exc}")
        return 2
    stored = st.get("fingerprint", {})
    ok = cur == stored
    print("FINGERPRINT:", "MATCH -- resume is SAFE (no purge)"
          if ok else "MISMATCH -- a bare --resume WOULD PURGE this run")
    for k in cur:
        same = cur.get(k) == stored.get(k)
        tag = "OK " if same else "CHG"
        print(f"  [{tag}] {k}: cur={cur[k]!r} stored={stored.get(k)!r}")
    extra = [k for k in stored if k not in cur]
    for k in extra:
        print(f"  [GONE] {k}: stored={stored[k]!r} (not in current)")
    print("completed_stages:", st.get("completed_stages", []))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
