"""Negative design (user §4).

Recommending only "good" mutations inflates false positives. We explicitly
penalise structurally / mechanistically risky mutations. The objective is
NOT "maximise binding" but: maintain fold + catalytic geometry + improve the
desired interaction + keep turnover-compatible (not over-tight) binding.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from evoliez.features.evolutionary import PositionFeature
from evoliez.features.mechanism import Mechanism, catalytic_geometry_deviation
from evoliez.types import Candidate

# hydrophobic residues; mutating a buried hydrophobic core to a charged/polar
# residue is destabilising.
_HYDROPHOBIC = set("AVLIMFWC")
_POLAR_CHARGED = set("DEKRNQHST")


def negative_penalties(
    cand: Candidate,
    *,
    wt_mech: Mechanism,
    mut_mech: Optional[Mechanism],
    position_features: Sequence[PositionFeature],
    catalytic_positions: Sequence[int],
    buried_fraction: float,
    docking_score: float,
    redocking_consistency: float,
) -> Dict[str, float]:
    cat = set(catalytic_positions)
    pf = {f.target_position: f for f in position_features
          if f.target_position is not None}

    catalytic_mut = 0.0
    conserved_motif = 0.0
    buried_core_polar = 0.0
    for m in cand.mutations:
        if m.position in cat or wt_mech.roles.get(m.position, {}).get("catalytic"):
            catalytic_mut += 1.0
        f = pf.get(m.position)
        if f and f.conservation_score >= 0.85:
            conserved_motif += (f.conservation_score - 0.85) / 0.15
        if (buried_fraction > 0.6 and m.wt in _HYDROPHOBIC
                and m.mut in _POLAR_CHARGED):
            buried_core_polar += 1.0

    geom = (
        catalytic_geometry_deviation(wt_mech, mut_mech)
        if mut_mech is not None else 0.0
    )
    # over-binding: very strong predicted binding can slow product release
    overbinding = max(0.0, (-docking_score) - 11.0) / 4.0
    pose_inversion = max(0.0, 1.0 - redocking_consistency)

    return {
        "neg_catalytic_mut": round(catalytic_mut, 4),
        "neg_conserved_motif": round(conserved_motif, 4),
        "neg_buried_core_polar": round(buried_core_polar, 4),
        "neg_catalytic_geometry": round(geom, 4),
        "neg_overbinding": round(overbinding, 4),
        "neg_pose_inversion": round(pose_inversion, 4),
    }
