#!/usr/bin/env python3
"""Preview how ``interaction_model.representative_homologs: auto`` resolves on a
run's REAL homolog set — read-only, no Boltz, no GPU.

Reads the persisted homolog rows (subfamily ``cluster_id`` + identity) from the
run's SQLite DB, reconstructs the subfamily-cluster size distribution, and shows
the cumulative-coverage curve plus what ``auto`` would pick under the config's
``representative_coverage`` / ``representative_min`` / ``representative_max`` — so
you can tune the bounds against the real MSA before spending GPU time.

    python scripts/preview_representatives.py CONFIG.yaml
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter
from pathlib import Path

from evoliez.config import load_config
from evoliez.stages.s06b_interaction_model import _resolve_n_reps


def cluster_sizes(db: Path) -> list:
    """Per-subfamily member counts (descending) from the homolog rows.

    Plain connect (not ``mode=ro``): the run DB is WAL-mode, which a pure
    read-only URI open cannot attach to (needs shared memory). The caller
    guards ``db.exists()`` so this never creates a stray file, and we only
    SELECT — the DB is left unchanged."""
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(
            "SELECT cluster_id FROM sequence WHERE source='homolog'"
        ).fetchall()
    finally:
        con.close()
    cids = [r[0] if r[0] is not None else -1 for r in rows]
    return sorted(Counter(cids).values(), reverse=True)


def _clusters_for_coverage(sizes: list, cov: float, n_total: int) -> int:
    cum = k = 0
    for s in sizes:
        cum += s
        k += 1
        if cum >= cov * n_total:
            break
    return k


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cfg = load_config(argv[1])
    db = Path(cfg.project.output_dir) / "evoliez.sqlite"
    if not db.exists():
        print(f"no DB at {db}\nrun at least s02_homolog for this config first")
        return 1
    sizes = cluster_sizes(db)
    if not sizes:
        print(f"no homolog rows (source='homolog') in {db}")
        return 1

    n_total, n_clusters = sum(sizes), len(sizes)
    im = cfg.interaction_model
    print(f"homologs:            {n_total}")
    print(f"subfamily clusters:  {n_clusters}")
    print(f"largest cluster sizes: {sizes[:10]}")
    print()
    print("coverage -> # largest clusters needed (the auto count BEFORE clamp):")
    for cov in (0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99):
        k = _clusters_for_coverage(sizes, cov, n_total)
        print(f"  {cov:>4.0%} -> {k:>5d}")
    print()

    # What THIS config's auto settings resolve to (force auto regardless of the
    # current representative_homologs value so the preview is always meaningful).
    auto_cfg = im.model_copy(update={"representative_homologs": "auto"})
    n, note = _resolve_n_reps(sizes, n_total, auto_cfg)
    poses = n * im.poses_per_homolog
    print(f"config: coverage={im.representative_coverage:.0%}  "
          f"min={im.representative_min}  max={im.representative_max}  "
          f"poses_per_homolog={im.poses_per_homolog}")
    print(f"  -> {note}")
    print(f"  -> builds {n} representatives x {im.poses_per_homolog} "
          f"= {poses} Boltz poses for the s06b ML dataset")
    if isinstance(im.representative_homologs, int):
        cur = im.representative_homologs
        print(f"\n(current config uses a FIXED representative_homologs={cur}"
              f" -> {cur * im.poses_per_homolog} poses; set it to \"auto\" to use"
              " the above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
