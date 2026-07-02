"""EvidenceCard builder (ROADMAP_V3 D4 / V3-5).

Maps a candidate's existing provenance (the s09/s10 record dict — stability_score,
nac_occupancy, hbond_occupancy, docking_uncertainty, …) onto the six v3 evidence axes,
with a SCORE and a SEPARATELY-DERIVED CONFIDENCE per axis. The point of the split:

  * a geometry score can be HIGH while its confidence is LOW (Tier-C reference, no
    controls, sparse WT geometry) — and the report must then hedge, not boast;
  * uncertainty.score is "higher = worse".

Confidence is rule-based + reproducible (the §6 disposition table + the reference/
ensemble/simulation cards), never a hand-written label. Pure / numeric; light env.
"""
from __future__ import annotations

from typing import Optional

from evoliez.mechanism.cards import (
    ConfidenceModel, EvidenceAxis, EvidenceCard, ReferenceConfidenceCard,
    SimulationSetupCard,
)
from evoliez.mechanism.vocab import (
    HYPOTHESIS_GRADE, MODERATE_SCREENING, STRONG_SCREENING, UNCALIBRATED,
    confidence_from_score,
)


def _f(rec: dict, key: str):
    v = rec.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def _present(*vals) -> int:
    return sum(1 for v in vals if v is not None)


def _conf_label(base: float, *, reference_card: Optional[ReferenceConfidenceCard],
                sim_penalty: bool, extra_penalty: float = 0.0,
                n_signals: int = 1) -> ConfidenceModel:
    """Confidence is INDEPENDENT of the axis score (ROADMAP_V3 D4): it measures how much
    we trust the measurement, not its value. Built from the reference strength + data
    quality (number of agreeing signals), then a MULTIPLICATIVE penalty for a weak
    reference / unvalidated MD parameterization / sparse-geometry — so a penalty actually
    drags confidence down even when the score is high. ``base`` (the score) is accepted
    only for the legacy signature and deliberately unused."""
    ref_strength = (reference_card.claim_strength if reference_card else UNCALIBRATED)
    ref_component = {
        STRONG_SCREENING: 0.8, MODERATE_SCREENING: 0.55,
        HYPOTHESIS_GRADE: 0.35, UNCALIBRATED: 0.2,
    }.get(ref_strength, 0.2)
    data_quality = _clamp01(0.4 + 0.2 * n_signals)         # more signals -> better
    trust = (ref_component + data_quality) / 2.0
    factor = _clamp01(1.0 - extra_penalty)
    if sim_penalty:
        factor *= 0.6
    final = _clamp01(trust * factor)
    components = {
        "reference_confidence": round(ref_component, 3),
        "data_quality": round(data_quality, 3),
    }
    if extra_penalty:
        components["axis_penalty"] = round(_clamp01(1.0 - extra_penalty), 3)
    if sim_penalty:
        components["md_parameterization"] = 0.3
    return ConfidenceModel(method="rule_based_v1", components=components,
                           final_confidence=confidence_from_score(final))


def build_evidence_card(
    rec: dict, *, reference_card: Optional[ReferenceConfidenceCard] = None,
    sim_card: Optional[SimulationSetupCard] = None,
    wt_geometry_sparse: bool = False, ensemble_geometry_variance: float = 0.0,
    msa_neff: Optional[float] = None,
) -> EvidenceCard:
    """Build the per-candidate evidence card. Missing fields -> score 0 with LOW
    confidence (an honest "no evidence", never a silent pass)."""
    variant_id = str(rec.get("candidate_id") or rec.get("variant_id") or "unknown")
    sim_penalty = bool(sim_card and sim_card.md_confidence_penalty())

    # --- structural_viability: stability up, instability/drift down --------------------
    stab = _f(rec, "stability_score")
    inst = _f(rec, "md_instability")
    drift = _f(rec, "energy_drift")
    sv_score = 0.0
    if stab is not None:
        sv_score = _clamp01(0.5 + stab / 4.0)              # ddG-like; +ve stabilizing
    if inst is not None:
        sv_score = _clamp01(sv_score - 0.5 * inst) if stab is not None else _clamp01(1 - inst)
    structural_viability = EvidenceAxis(
        score=round(sv_score, 3),
        confidence=_conf_label(sv_score, reference_card=reference_card,
                               sim_penalty=False,
                               n_signals=_present(stab, inst, drift)).final_confidence,
        evidence=[k for k, v in (("stability_model_pass", stab), ("md_instability", inst),
                                 ("energy_drift", drift)) if v is not None])

    # --- evolutionary_tolerance -------------------------------------------------------
    cons = _f(rec, "conservation")
    msa = _f(rec, "msa_permissiveness") or _f(rec, "msa_freq")
    esm = _f(rec, "esm_permissive")
    et_vals = [v for v in (msa, esm, (1 - cons) if cons is not None else None) if v is not None]
    et_score = _clamp01(sum(et_vals) / len(et_vals)) if et_vals else 0.0
    neff_penalty = 0.0 if (msa_neff is None or msa_neff >= 30) else 0.3
    evolutionary_tolerance = EvidenceAxis(
        score=round(et_score, 3),
        confidence=_conf_label(et_score, reference_card=reference_card, sim_penalty=False,
                               extra_penalty=neff_penalty,
                               n_signals=_present(cons, msa, esm)).final_confidence,
        evidence=[k for k, v in (("PSSM_allowed", msa), ("esm_permissive", esm),
                                 ("conservation", cons)) if v is not None])

    # --- ligand_cofactor_competence ---------------------------------------------------
    hb = _f(rec, "hbond_occupancy")
    ig = _f(rec, "interaction_gain")
    redock = _f(rec, "redocking_consistency")
    lc_vals = [v for v in (hb, redock) if v is not None]
    lc_score = _clamp01(sum(lc_vals) / len(lc_vals)) if lc_vals else 0.0
    if ig is not None:
        lc_score = _clamp01(lc_score + 0.1 * ig) if lc_vals else _clamp01(0.5 + 0.1 * ig)
    ligand_cofactor_competence = EvidenceAxis(
        score=round(lc_score, 3),
        confidence=_conf_label(lc_score, reference_card=reference_card,
                               sim_penalty=sim_penalty,
                               n_signals=_present(hb, ig, redock)).final_confidence,
        evidence=[k for k, v in (("anchor_contacts_preserved", hb),
                                 ("interaction_gain", ig),
                                 ("redock_consistency", redock)) if v is not None])

    # --- substrate_positioning --------------------------------------------------------
    cat_d = _f(rec, "catalytic_distance_mean")
    pocket = _f(rec, "pocket_rmsd_mean")
    sp_score = 0.0
    if cat_d is not None:
        sp_score = _clamp01(1.0 - (cat_d - 3.0) / 4.0)    # closer to ~3 Å is better
    analog_penalty = 0.0 if (reference_card is None or reference_card.substrate_is_real) else 0.3
    substrate_positioning = EvidenceAxis(
        score=round(sp_score, 3),
        confidence=_conf_label(sp_score, reference_card=reference_card, sim_penalty=False,
                               extra_penalty=analog_penalty,
                               n_signals=_present(cat_d, pocket)).final_confidence,
        evidence=[k for k, v in (("near_reactive_site", cat_d),
                                 ("pocket_rmsd", pocket)) if v is not None])

    # --- reaction_geometry_accommodation (generalized NAC) ----------------------------
    nac = _f(rec, "nac_occupancy")
    nac_delta = _f(rec, "nac_delta_vs_wt")
    pen = _f(rec, "catalytic_geometry_penalty")
    # ΔNAC-vs-WT is THE catalytic-accommodation signal for triage: in a restrained
    # implicit-solvent screen the ABSOLUTE occupancy is uniformly low (~0.1) for every
    # candidate, so ranking on it hides the lead. The RELATIVE gain (mutant vs WT) is
    # what distinguishes a catalytic gainer (>0 better than WT, <0 worse; 0.5 neutral).
    # Absolute occupancy then only fine-tunes / feeds confidence (it is sparse -> low).
    # NOTE: catalytic_geometry_penalty is computed on the per-mutant Boltz pose (noisy /
    # contaminated per the anchored-validation finding), so it is NOT trusted as POSITIVE
    # reaction-geometry evidence — it only DOWNGRADES confidence below. The score uses the
    # real MD-derived NAC signal: ΔNAC vs WT (primary), then absolute occupancy.
    rg_score = 0.0
    if nac_delta is not None:
        rg_score = _clamp01(0.5 + 2.0 * nac_delta)
        if nac is not None:
            rg_score = _clamp01(0.85 * rg_score + 0.15 * nac)
    elif nac is not None:
        rg_score = _clamp01(nac)
    # no NAC data at all -> 0.0 (honest: no reaction-geometry evidence; never a Boltz-pose proxy)
    geo_penalty = 0.0
    geo_evidence = [k for k, v in (("nac_occupancy", nac), ("nac_delta_vs_wt", nac_delta),
                                   ("catalytic_geometry_penalty", pen)) if v is not None]
    if wt_geometry_sparse:
        geo_penalty = max(geo_penalty, 0.5); geo_evidence.append("wt_geometry_sparse")
    if ensemble_geometry_variance >= 0.1:
        geo_penalty = max(geo_penalty, 0.4); geo_evidence.append("ensemble_disagreement")
    if pen is not None and pen > 1.0:
        # a high Boltz-pose strain penalty lowers CONFIDENCE (pose is contaminated), not score
        geo_penalty = max(geo_penalty, 0.3); geo_evidence.append("boltz_pose_strain")
    reaction_geometry_accommodation = EvidenceAxis(
        score=round(rg_score, 3),
        confidence=_conf_label(rg_score, reference_card=reference_card,
                               sim_penalty=sim_penalty, extra_penalty=geo_penalty,
                               n_signals=_present(nac, nac_delta, pen)).final_confidence,
        reason=("score from a Tier-C reference / sparse WT geometry — diagnostic"
                if (geo_penalty or (reference_card and reference_card.tier in ("C", "D")))
                else ""),
        evidence=geo_evidence)

    # --- uncertainty (higher = worse) -------------------------------------------------
    dock_unc = _f(rec, "docking_uncertainty")
    d_iptm = _f(rec, "d_ligand_iptm")
    alt_pose = _f(rec, "alternative_pose_boltz")
    unc_signals = [v for v in (dock_unc, abs(d_iptm) if d_iptm is not None else None,
                               alt_pose) if v is not None]
    unc_score = _clamp01(sum(unc_signals) / len(unc_signals)) if unc_signals else 0.5
    # confidence in the uncertainty estimate is HIGH when many validators weigh in
    unc_conf = confidence_from_score(_clamp01(0.3 + 0.25 * _present(dock_unc, d_iptm, alt_pose)))
    uncertainty = EvidenceAxis(
        score=round(unc_score, 3), confidence=unc_conf,
        evidence=[k for k, v in (("docking_uncertainty", dock_unc),
                                 ("de_novo_pose_disagreement", d_iptm),
                                 ("alternative_pose_boltz", alt_pose)) if v is not None])

    overall = ConfidenceModel(method="rule_based_v1", components={
        "reference_confidence": round({STRONG_SCREENING: 0.8, MODERATE_SCREENING: 0.55,
                                       HYPOTHESIS_GRADE: 0.35, UNCALIBRATED: 0.2}.get(
            reference_card.claim_strength if reference_card else UNCALIBRATED, 0.2), 3),
        "geometry_penalty": round(_clamp01(1 - geo_penalty), 3),
        "md_parameterization": 0.3 if sim_penalty else 0.7,
    }).recompute()

    return EvidenceCard(
        variant_id=variant_id,
        structural_viability=structural_viability,
        evolutionary_tolerance=evolutionary_tolerance,
        ligand_cofactor_competence=ligand_cofactor_competence,
        substrate_positioning=substrate_positioning,
        reaction_geometry_accommodation=reaction_geometry_accommodation,
        uncertainty=uncertainty,
        confidence_model=overall,
    )
