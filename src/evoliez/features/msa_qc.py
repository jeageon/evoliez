"""MSA quality control (ROADMAP_V2 Phase G). Raw depth is NOT diversity — a 10k-sequence MSA
can be one redundant subfamily, or padded with synthetic/remote-derived rows. Surface:

- ``neff_clusters`` — effective independent sequence count = number of subfamily clusters
  (each sequence weighted 1/cluster_size, so a cluster of any size contributes ~1).
- ``largest_subfamily_frac`` / ``subfamily_balance`` — is the alignment dominated by one
  subfamily (redundant) or well spread?
- ``sources`` / ``real_fraction`` — how much of the depth is REAL homology vs synthetic/mock.

So a shallow / redundant / synthetic-padded MSA is visible in the report instead of hidden
behind a big raw depth. Pure + unit-testable.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence


def msa_qc(depth: int, cluster_ids: Optional[Sequence] = None,
           source_breakdown: Optional[Dict[str, int]] = None) -> dict:
    qc: Dict[str, object] = {"depth": int(depth)}

    if cluster_ids:
        sizes = Counter(cluster_ids)
        n_clusters = len(sizes)
        total = sum(sizes.values()) or 1
        largest = max(sizes.values()) if sizes else 0
        qc["neff_clusters"] = n_clusters            # effective independent count
        qc["n_subfamilies"] = n_clusters
        qc["largest_subfamily_frac"] = round(largest / total, 3)
        qc["subfamily_balance"] = round(n_clusters / total, 3)  # 1=all-singletons; low=redundant

    if source_breakdown:
        sb = dict(source_breakdown)
        qc["sources"] = sb
        tot = sum(sb.values()) or 1
        real = sum(v for k, v in sb.items()
                   if not any(t in str(k).lower()
                              for t in ("mock", "synthetic", "placeholder")))
        qc["real_fraction"] = round(real / tot, 3)

    return qc
