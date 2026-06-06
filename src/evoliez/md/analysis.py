"""Trajectory metrics, MD pass/fail, and the MD-lite score
(spec sections 15.7-15.9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List

from evoliez.adapters.openmm_engine import MDResult
from evoliez.config import ScoreWeights


@dataclass
class MDMetrics:
    ligand_rmsd_mean: float = 0.0
    ligand_rmsd_final: float = 0.0
    ligand_escape: bool = False
    pocket_rmsd_mean: float = 0.0
    contact_occupancy_mean: float = 0.0
    hbond_occupancy: float = 0.0
    catalytic_distance_mean: float = 0.0
    catalytic_distance_std: float = 0.0
    energy_drift: float = 0.0
    simulation_health_ok: bool = True
    md_lite_score: float = 0.0
    passed: bool = True
    failure_reasons: List[str] = field(default_factory=list)


# Default pass/fail thresholds (spec 15.8). Enzyme-class overrides belong in
# config; these are the conservative screening defaults.
LIGAND_RMSD_MAX = 4.5
POCKET_RMSD_MAX = 2.75
HBOND_MIN = 0.25
CAT_DIST_MAX = 7.0


def analyse(result: MDResult, weights: ScoreWeights) -> MDMetrics:
    m = MDMetrics()

    # ONLY skipped_parameterization is neutral: an optional small-molecule
    # FF was unavailable (common for metals/cofactors) - judge the candidate
    # on the other layers, don't penalise (expert review). Any OTHER skip
    # (e.g. skipped_no_full_atom_structure) means real MD never ran on this
    # candidate -> NOT a pass; surfacing it honestly is the difference
    # between "MD validated" and "MD silently didn't run".
    if result.status == "skipped_parameterization":
        m.passed = True
        m.md_lite_score = 0.0
        m.failure_reasons.append(f"MD {result.status}: "
                                 f"{result.failure_reason or 'n/a'}")
        return m
    if result.status.startswith("skipped"):
        m.passed = False
        m.simulation_health_ok = False
        m.md_lite_score = 0.0
        m.failure_reasons.append(f"MD {result.status}: "
                                 f"{result.failure_reason or 'n/a'}")
        return m

    if result.status == "failed" or result.integration_failed:
        m.simulation_health_ok = False
        m.passed = False
        m.failure_reasons.append(result.failure_reason or "simulation failed")
        m.md_lite_score = -5.0
        return m

    lig = result.ligand_rmsd_series or [0.0]
    pkt = result.pocket_rmsd_series or [0.0]
    m.ligand_rmsd_mean = round(mean(lig), 3)
    m.ligand_rmsd_final = round(lig[-1], 3)
    m.ligand_escape = lig[-1] > LIGAND_RMSD_MAX
    m.pocket_rmsd_mean = round(mean(pkt), 3)
    m.contact_occupancy_mean = round(
        mean(result.contact_occupancy.values()) if result.contact_occupancy else 0.0,
        3,
    )
    m.hbond_occupancy = round(result.hbond_occupancy, 3)
    m.energy_drift = round(result.energy_drift, 4)

    all_d: List[float] = [d for series in result.key_distances.values() for d in series]
    if all_d:
        m.catalytic_distance_mean = round(mean(all_d), 3)
        m.catalytic_distance_std = round(pstdev(all_d) if len(all_d) > 1 else 0.0, 3)

    # pass/fail (spec 15.8)
    if m.ligand_escape or m.ligand_rmsd_mean > LIGAND_RMSD_MAX:
        m.failure_reasons.append("ligand left pocket / high ligand RMSD")
    if m.pocket_rmsd_mean > POCKET_RMSD_MAX:
        m.failure_reasons.append("pocket backbone destabilised")
    if result.key_distances and m.catalytic_distance_mean > CAT_DIST_MAX:
        m.failure_reasons.append("catalytic geometry implausible")
    if m.hbond_occupancy < HBOND_MIN and result.hbond_occupancy is not None:
        m.failure_reasons.append("required H-bond occupancy too low")
    if result.energy_drift > 0.5:
        m.failure_reasons.append("energy drift / unstable integration")
        m.simulation_health_ok = False
    m.passed = not m.failure_reasons

    # MD-lite QUALITY score (spec 15.9): a RAW, weight-free measure of how well
    # the bound state held up. The ScoreWeights.md_lite weight is applied ONCE
    # downstream in ranking.score.compute_final_score - it must NOT be folded in
    # here as well (that squared the MD term at any non-1.0 weight). Integration
    # health is penalised separately and transparently via `md_instability`
    # (s10 -> md_instability_penalty), so it is NOT also subtracted here (that
    # double-counted an unhealthy run). `weights` is kept on the signature for
    # future enzyme-class threshold overrides.
    contact = m.contact_occupancy_mean
    keydist_stab = max(0.0, 1.0 - m.catalytic_distance_std / 2.0)
    cat_geom = max(0.0, 1.0 - max(0.0, m.catalytic_distance_mean - 3.5) / 5.0)
    lig_pen = max(0.0, m.ligand_rmsd_mean - 2.0) / 3.0
    pkt_pen = max(0.0, m.pocket_rmsd_mean - 1.5) / 2.0
    score = (
        0.30 * contact
        + 0.20 * keydist_stab
        + 0.20 * m.hbond_occupancy
        + 0.20 * cat_geom
        - 0.35 * lig_pen
        - 0.25 * pkt_pen
        - 0.50 * (1.0 if m.ligand_escape else 0.0)
    )
    m.md_lite_score = round(float(score), 4)
    return m


def to_json(metrics: MDMetrics) -> Dict[str, object]:
    return {
        "ligand_rmsd_mean": metrics.ligand_rmsd_mean,
        "ligand_rmsd_final": metrics.ligand_rmsd_final,
        "ligand_escape": metrics.ligand_escape,
        "pocket_rmsd_mean": metrics.pocket_rmsd_mean,
        "contact_occupancy_mean": metrics.contact_occupancy_mean,
        "hbond_occupancy": metrics.hbond_occupancy,
        "catalytic_distance_mean": metrics.catalytic_distance_mean,
        "catalytic_distance_std": metrics.catalytic_distance_std,
        "energy_drift": metrics.energy_drift,
        "simulation_health_ok": metrics.simulation_health_ok,
        "md_lite_score": metrics.md_lite_score,
        "passed": metrics.passed,
        "failure_reasons": metrics.failure_reasons,
    }
