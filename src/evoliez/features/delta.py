"""WT - mutant delta features (user spec section 6).

Relative change vs the wild type is often more informative than absolute
Boltz values. All deltas are FEATURES, never labels.
"""

from __future__ import annotations

from typing import Dict, Sequence

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


def boltz_delta_features(
    mut_cx: Complex,
    wt_cx: Complex,
    *,
    catalytic_positions: Sequence[int] = (),
    contact_cutoff: float = 6.0,
) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for k in _METRICS:
        mv = mut_cx.metrics.get(k)
        wv = wt_cx.metrics.get(k)
        if mv is not None and wv is not None:
            d[f"d_{k}"] = round(float(mv) - float(wv), 4)

    d["d_pocket_plddt"] = round(
        pocket_plddt(mut_cx.structure, mut_cx.ligand.atoms)
        - pocket_plddt(wt_cx.structure, wt_cx.ligand.atoms),
        4,
    )

    if catalytic_positions:
        wt_cat = catalytic_distances(
            wt_cx.structure, wt_cx.ligand.atoms, catalytic_positions
        )
        mut_cat = catalytic_distances(
            mut_cx.structure, mut_cx.ligand.atoms, catalytic_positions
        )
        diffs = [
            mut_cat[k] - wt_cat[k] for k in wt_cat if k in mut_cat
        ]
        d["d_key_distance"] = round(
            sum(abs(x) for x in diffs) / len(diffs), 4
        ) if diffs else 0.0

    wt_n = len(
        residue_ligand_contacts(wt_cx.structure, wt_cx.ligand.atoms,
                                contact_cutoff)
    )
    mut_n = len(
        residue_ligand_contacts(mut_cx.structure, mut_cx.ligand.atoms,
                                contact_cutoff)
    )
    d["d_contact_count"] = float(mut_n - wt_n)
    return d
