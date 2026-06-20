"""Final multi-objective design score (spec section 16).

Deliberately additive and interpretable: the total is decomposed into named
positive contributions and penalties so every candidate carries a transparent
rationale rather than a black-box number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from evoliez.config import ScoreWeights
from evoliez.types import Candidate


@dataclass
class ScoreBreakdown:
    total: float
    contributions: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)


def compute_final_score(cand: Candidate, w: ScoreWeights) -> ScoreBreakdown:
    s = cand.scores
    contrib = {
        "ml_mutation": w.ml_mutation * s.get("ml_score", 0.0),
        "ligand_interaction_gain": w.ligand_interaction_gain
        * s.get("interaction_gain", 0.0),
        "msa_permissiveness": w.msa_permissiveness * s.get("msa_permissiveness", 0.0),
        "complex_confidence": w.complex_confidence * s.get("complex_confidence", 0.0),
        "redocking_consistency": w.redocking_consistency
        * s.get("redocking_consistency", 0.0),
        "key_contact_preservation": w.key_contact_preservation
        * s.get("key_contact_preservation", 0.0),
        "stability": w.stability * s.get("stability_score", 0.0),
        "md_lite": w.md_lite * s.get("md_lite_score", 0.0),
        "family_interaction": w.family_interaction
        * s.get("family_interaction_score", 0.0),
        # real per-mutant ΔBoltz: positive d_ligand_iptm = mutant binds the
        # design-target ligand better than WT. Set by s08b for the top-N; 0 for
        # candidates that only carry the proxy Δ -- so the expensive real Boltz
        # re-prediction now actually moves the final ranking.
        "mutant_boltz_gain": w.mutant_boltz_gain * s.get("d_ligand_iptm", 0.0),
        "gnn": w.gnn * s.get("gnn_score", 0.0),
        "catalytic_geometry_preservation": w.catalytic_geometry_preservation
        * s.get("ts_geometry_score", 0.0),
        "specificity_divergence": w.specificity_divergence_bonus
        * s.get("specificity_divergence", 0.0),
    }
    penalties = {
        "conservation": w.conservation_penalty * s.get("conservation_penalty", 0.0),
        "catalytic_geometry": w.catalytic_geometry_penalty
        * s.get("catalytic_geometry_penalty", 0.0),
        "ddg_stability": w.ddg_penalty * max(0.0, s.get("ddg_fold", 0.0)) / 3.0,
        "clash": w.clash_penalty * s.get("clash_score", 0.0),
        "docking_uncertainty": w.docking_uncertainty_penalty
        * s.get("docking_uncertainty", 0.0),
        "md_instability": w.md_instability_penalty * s.get("md_instability", 0.0),
        # negative design (user §4)
        "neg_catalytic_mut": w.neg_catalytic_mut
        * s.get("neg_catalytic_mut", 0.0),
        "neg_conserved_motif": w.neg_conserved_motif
        * s.get("neg_conserved_motif", 0.0),
        "neg_buried_core_polar": w.neg_buried_core_polar
        * s.get("neg_buried_core_polar", 0.0),
        "neg_catalytic_geometry": w.neg_catalytic_geometry
        * s.get("neg_catalytic_geometry", 0.0),
        "neg_overbinding": w.neg_overbinding * s.get("neg_overbinding", 0.0),
        "neg_pose_inversion": w.neg_pose_inversion
        * s.get("neg_pose_inversion", 0.0),
    }
    total = sum(contrib.values()) - sum(penalties.values())
    return ScoreBreakdown(
        total=round(total, 4),
        contributions={k: round(v, 4) for k, v in contrib.items()},
        penalties={k: round(v, 4) for k, v in penalties.items()},
    )


def rationale(cand: Candidate, breakdown: ScoreBreakdown) -> List[str]:
    """Human-readable 'why' lines (spec 16 example report)."""
    lines: List[str] = []
    d = cand.details
    if d.get("near_ligand_atom"):
        lines.append(f"located near ligand atom {d['near_ligand_atom']}")
    if d.get("msa_variable"):
        lines.append("variable in the MSA at this position")
    if d.get("family_observed"):
        lines.append(f"{d['family_observed']} observed at homologous positions")
    top = sorted(breakdown.contributions.items(), key=lambda kv: -kv[1])[:3]
    for name, val in top:
        if val > 0:
            lines.append(f"{name.replace('_', ' ')} contributes +{val:.2f}")
    worst = sorted(breakdown.penalties.items(), key=lambda kv: -kv[1])[:2]
    for name, val in worst:
        if val > 0.05:
            lines.append(f"risk: {name.replace('_', ' ')} penalty -{val:.2f}")
    return lines
