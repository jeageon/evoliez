"""Subfamily-aware evolutionary prior (user §7).

A single global MSA buries specificity signal. We split the family into
subfamilies (homolog clusters) and, per target position, compute:

  - specificity_divergence : how differently subfamilies fix this position
    (high = specificity-determining residue, a cofactor/substrate-switch
     engineering candidate)

Features only - never labels.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

from evoliez.features.evolutionary import PositionFeature, target_column_map


def annotate_subfamilies(
    msa: Sequence[Tuple[str, str]],
    feats: List[PositionFeature],
    cluster_of: Dict[str, int],
) -> None:
    """Mutate ``feats`` in place. ``cluster_of`` maps MSA row id -> subfamily
    id (target row uses the most common / its own cluster)."""
    if not msa or len(msa) < 3:
        return
    ids = [cid for cid, _ in msa]
    seqs = [s for _, s in msa]
    colmap = target_column_map(msa)

    # group row indices by subfamily
    groups: Dict[int, List[int]] = {}
    for i, rid in enumerate(ids):
        groups.setdefault(cluster_of.get(rid, -1), []).append(i)
    by_pos = {f.alignment_position: f for f in feats}
    for col, f in by_pos.items():
        if f.target_position is None:
            continue
        # specificity divergence = #distinct subfamily-majority residues
        majs = []
        for sf, rows in groups.items():
            col_aa = [seqs[i][col] for i in rows
                      if col < len(seqs[i]) and seqs[i][col] != "-"]
            if col_aa:
                majs.append(Counter(col_aa).most_common(1)[0][0])
        if majs:
            f.specificity_divergence = round(
                len(set(majs)) / len(majs), 4
            )
    _ = colmap  # alignment->target map kept for callers/debugging
