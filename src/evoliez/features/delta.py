"""WT - mutant delta features (user spec section 6).

Relative change vs the wild type is often more informative than absolute
Boltz values. All deltas are FEATURES, never labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

from evoliez.features.boltz_features import pocket_plddt
from evoliez.features.geometry import (
    catalytic_distances,
    residue_ligand_contacts,
)
from evoliez.types import Complex

_METRICS = [
    "confidence_score", "ptm", "iptm", "ligand_iptm", "complex_plddt",
    "complex_iplddt", "complex_pde", "complex_ipde",
    "affinity_pred_value", "affinity_probability_binary",
]


@dataclass(frozen=True)
class WTDeltaCache:
    """WT-side terms used by `boltz_delta_features`. Identical for every
    mutant in a run since the WT complex doesn't change, so we hoist
    them out of the per-candidate loop (~30+ recomputations in s08, ~10
    in s08b). Each term is a pure function of (wt.structure, wt.ligand,
    contact_cutoff, catalytic_positions) - bit-identical to the
    previous in-loop computation."""
    pocket_plddt: float
    contact_count: int
    catalytic_distances: Mapping[str, float]  # empty when no catalytics

    @classmethod
    def build(cls, wt_cx: Complex, *,
              catalytic_positions: Sequence[int] = (),
              contact_cutoff: float = 6.0) -> "WTDeltaCache":
        return cls(
            pocket_plddt=pocket_plddt(wt_cx.structure, wt_cx.ligand.atoms),
            contact_count=len(
                residue_ligand_contacts(
                    wt_cx.structure, wt_cx.ligand.atoms, contact_cutoff,
                )
            ),
            catalytic_distances=(
                catalytic_distances(
                    wt_cx.structure, wt_cx.ligand.atoms, catalytic_positions,
                ) if catalytic_positions else {}
            ),
        )


def boltz_delta_features(
    mut_cx: Complex,
    wt_cx: Complex,
    *,
    catalytic_positions: Sequence[int] = (),
    contact_cutoff: float = 6.0,
    wt_cache: Optional[WTDeltaCache] = None,
) -> Dict[str, float]:
    """Delta features mut - wt. Pass a pre-built `wt_cache` (one per
    run) to skip re-evaluating WT-side pocket_plddt / contact_count /
    catalytic_distances on every candidate; the result is numerically
    identical to computing them inline."""
    cache = wt_cache or WTDeltaCache.build(
        wt_cx, catalytic_positions=catalytic_positions,
        contact_cutoff=contact_cutoff,
    )

    d: Dict[str, float] = {}
    for k in _METRICS:
        mv = mut_cx.metrics.get(k)
        wv = wt_cx.metrics.get(k)
        if mv is not None and wv is not None:
            d[f"d_{k}"] = round(float(mv) - float(wv), 4)

    d["d_pocket_plddt"] = round(
        pocket_plddt(mut_cx.structure, mut_cx.ligand.atoms)
        - cache.pocket_plddt,
        4,
    )

    if catalytic_positions:
        mut_cat = catalytic_distances(
            mut_cx.structure, mut_cx.ligand.atoms, catalytic_positions
        )
        diffs = [
            mut_cat[k] - cache.catalytic_distances[k]
            for k in cache.catalytic_distances if k in mut_cat
        ]
        d["d_key_distance"] = round(
            sum(abs(x) for x in diffs) / len(diffs), 4
        ) if diffs else 0.0

    mut_n = len(
        residue_ligand_contacts(mut_cx.structure, mut_cx.ligand.atoms,
                                contact_cutoff)
    )
    d["d_contact_count"] = float(mut_n - cache.contact_count)
    return d
