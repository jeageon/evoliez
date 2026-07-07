#!/usr/bin/env python
"""Purge-safe --resume pre-flight: compare the CURRENT run_fingerprint to a run's
stored ``_state.json`` fingerprint WITHOUT calling ``setup()``/``_load_state`` (so it
can NEVER trigger ``_purge_stale_outputs`` — the rmtree that wipes complexes/docking/
interaction_graphs/ml_datasets on a fingerprint change). Run before any
``evoliez run --resume`` to confirm the resume reuses, not purges.

Exit 0 = MATCH (resume safe).  Exit 1 = MISMATCH (a bare --resume WOULD purge).
Exit 2 = error.

--patch : rewrite the stored fingerprint to the current one, but ONLY when the sole
changed component is ``config_sha1`` (a deliberate config edit that does not touch
input/version/backend/gnn and therefore must NOT purge the s01..s08b artifacts).
Refuses to patch if anything else changed (that signals a real invalidation).

Usage: check_run_fingerprint.py <config.yaml> [run_dir] [--patch]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# components safe to reconcile via --patch (they do not invalidate upstream artifacts)
_PATCHABLE = {"config_sha1"}


def main(argv) -> int:
    patch = "--patch" in argv
    argv = [a for a in argv if a != "--patch"]
    if not argv:
        print("usage: check_run_fingerprint.py <config.yaml> [run_dir] [--patch]")
        return 2
    from evoliez.config import load_config
    from evoliez.context import RunContext

    cfg = load_config(argv[0])
    ctx = RunContext(cfg)          # __init__ ONLY — no setup(), no _load_state, no purge
    cur = ctx.run_fingerprint()
    run_dir = Path(argv[1]) if len(argv) > 1 else ctx.root
    state_path = run_dir / "_state.json"
    if not state_path.exists():
        print(f"no _state.json at {state_path} — FRESH run (nothing to purge).")
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
    changed = []
    for k in cur:
        same = cur.get(k) == stored.get(k)
        if not same:
            changed.append(k)
        print(f"  [{'OK ' if same else 'CHG'}] {k}: cur={cur[k]!r} stored={stored.get(k)!r}")
    print("completed_stages:", st.get("completed_stages", []))

    if patch:
        if ok:
            print(">> already MATCH; nothing to patch.")
            return 0
        if set(changed) <= _PATCHABLE:
            st["fingerprint"] = cur
            state_path.write_text(json.dumps(st, indent=2))
            print(f">> PATCHED _state.json fingerprint (changed: {changed}); "
                  f"--resume will reuse, not purge.")
            return 0
        print(f">> REFUSED to patch: changed components {changed} exceed the safe "
              f"set {sorted(_PATCHABLE)} (real invalidation — do NOT bypass).")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
