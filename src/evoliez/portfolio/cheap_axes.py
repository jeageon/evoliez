"""All-candidate cheap seven-axis ledger (ROADMAP_V7 Phase V7-1).

Tier-0 of the multi-fidelity engine: score EVERY generated candidate on the cheap axes
BEFORE any docking / Boltz / MD is spent (ROADMAP_V7 §5). The input is the per-candidate
provenance a run already writes (``reports/provenance/generated_candidates.json`` etc.),
a list of plain dicts carrying sequence / MSA / mutation-identity / static-pocket features.

This module maps those cheap features onto five of the seven ledger axes
(``ledger.CHEAP_AXES``); the reaction-geometry and portfolio axes are panel-level /
expensive and stay DEFERRED here (``missing=True``, band ``deferred``) until the GPU tiers
and the builder fill them. Design invariants we obey (ROADMAP_V7 §2):

  * **Cheap-only.** Every quantity read into an axis SCORE is checked against
    ``ml.learnability`` — a field is admitted only if ``learnability.is_cheap`` and never if
    it is an expensive label source (nac / pose_gate / md_* / binding_dg / rbfe / ml_score …).
    ``assert_no_leakage`` gives the mechanical guarantee, exactly as ``learnability.assess``
    does for the triage model, so a cheap axis can never become a redundant copy of a gate.
  * **Missing is deferred, not zero.** An axis with no cheap ligand-proximity information is
    marked ``missing`` (deferred), never scored 0 — a candidate is never sunk by an honest gap.
    Structural / evolutionary / uncertainty cheap priors are *expected* to exist, so when
    they are absent we emit a present-but-weak ``unresolved`` axis (score 0, explicit
    ``no_cheap_*_signal`` provenance), which is a scored-weak signal, not an uncomputed gap.
  * **Uncalibrated bands.** A cheap score is scored but NOT yet calibrated against a null, so
    every scored axis here carries band ``unresolved`` — ``portfolio.stats`` promotes it to a
    significant band later. Rank never selects; only a calibrated band does.
  * **Mechanism-generic.** No enzyme / residue names are hardcoded. Roles are resolved from
    the passed ``MechanismSpec`` (its ``catalytic_residues`` seqids) and from record role tags
    (``ligand_roles``), matched positionally against the mutation string.
  * **Deterministic.** Pure rule-based scoring; identical output for identical input + seed.

Light-env safe: depends only on the ledger contract, ``ml.learnability`` (numpy), and the
shared confidence vocabulary — no rdkit / torch / openmm. The ``EvidenceCard`` adapter
imports its type lazily so this module imports even where the mechanism cards are unused.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

from evoliez.ml import learnability
from evoliez.mechanism.vocab import (
    CONF_HIGH, CONF_LOW, CONF_LOW_TO_MEDIUM, CONF_MEDIUM,
)
from evoliez.portfolio.ledger import (
    AXIS_EVOLUTIONARY, AXIS_LIGAND, AXIS_MECHANISM, AXIS_PORTFOLIO,
    AXIS_REACTION_GEOMETRY, AXIS_STRUCTURAL, AXIS_UNCERTAINTY, BAND_CONTROL,
    BAND_DEFERRED, BAND_UNRESOLVED, CLAIM_L0_HYPOTHESIS, CLAIM_L1_SCREENING,
    TIER_CHEAP, TIER_FOCUSED, TIER_GPU_BROAD, AxisEvidenceV7, EvidenceLedgerV7,
)

# generator -> prior on how explicitly the design encodes a mechanism hypothesis. These are
# PIPELINE-method names (not enzyme names) so the map stays target-generic. An active-site /
# combinatorial design encodes an explicit hypothesis; a marginal statistical pick less so.
_GENERATOR_HYPOTHESIS_PRIOR: Dict[str, float] = {
    "multipoint": 0.70,
    "multipoint_active_site": 0.80,
    "ligandmpnn": 0.70,
    "chemistry_rules": 0.50,
    "chemistry": 0.50,
    "msa": 0.35,
    "conservation": 0.35,
    "evolutionary": 0.35,
}
_GENERATOR_PRIOR_DEFAULT = 0.40

# soft length scales for the cheap geometric priors (Angstrom / count). Deliberately loose —
# a cheap static distance is a proximity PRIOR, never a NAC; the real reaction-geometry axis
# (distance AND angle, kept separate) is computed at a higher tier.
_LIGAND_SHELL_A = 8.0
_CONTACT_SATURATION = 6.0
# near-attack occupancy that counts as "full access" for the reaction-geometry axis. A
# restrained implicit-solvent screen samples the in-line conformation only rarely, so full
# credit at a modest fraction; a 0 occupancy still maps to 0 (no access), so a short distance
# alone can never band the axis (V7 §3 Axis 4).
_NAC_ACCESS_REFERENCE = 0.2
_COMBINATORIAL_SATURATION = 3.0   # (order-1) that saturates the combinatorial-hypothesis term
_MUTATION_LOAD_SATURATION = 4.0   # (n_mut-1) that saturates the uncertainty mutation-load term


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _num(rec: dict, key: str) -> Optional[float]:
    """Numeric getter mirroring ``learnability._as_float``: bool -> 1/0, number -> float,
    everything else (None / strings) -> missing."""
    v = rec.get(key)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _confidence(n_signals: int) -> str:
    """Rule-based confidence by count of independent cheap signals that fired."""
    if n_signals >= 3:
        return CONF_HIGH
    if n_signals == 2:
        return CONF_MEDIUM
    if n_signals == 1:
        return CONF_LOW_TO_MEDIUM
    return CONF_LOW


def _weaker_confidence(a: str, b: str) -> str:
    rank = {CONF_HIGH: 0, CONF_MEDIUM: 1, CONF_LOW_TO_MEDIUM: 2, CONF_LOW: 3}
    return a if rank.get(a, 3) >= rank.get(b, 3) else b


def _cheap_axis(axis: str, score: float, provenance: List[str], n_signals: int) -> AxisEvidenceV7:
    """A scored (but uncalibrated) cheap axis: band ``unresolved``, tier-0, hypothesis claim."""
    return AxisEvidenceV7(
        axis=axis, score=_clamp01(score), band=BAND_UNRESOLVED,
        confidence=_confidence(n_signals), provenance=list(provenance),
        claim_ceiling=CLAIM_L0_HYPOTHESIS, tier=TIER_CHEAP, missing=False,
    )


def _weak_axis(axis: str, reason: str) -> AxisEvidenceV7:
    """Present-but-weak: a cheap prior that is expected to exist but is absent for this row.
    Scored 0 / ``unresolved`` (NOT ``missing``) — a weak signal, not an uncomputed gap."""
    return AxisEvidenceV7(axis=axis, score=0.0, band=BAND_UNRESOLVED, confidence=CONF_LOW,
                          provenance=[reason], claim_ceiling=CLAIM_L0_HYPOTHESIS,
                          tier=TIER_CHEAP, missing=False)


def _deferred_ligand_axis() -> AxisEvidenceV7:
    """No cheap ligand-proximity information -> DEFERRED (an honest gap, never a zero)."""
    return AxisEvidenceV7(axis=AXIS_LIGAND, score=0.0, band=BAND_DEFERRED, missing=True,
                          confidence=CONF_LOW, provenance=["deferred_no_cheap_ligand_proximity"],
                          claim_ceiling=CLAIM_L0_HYPOTHESIS, tier=TIER_CHEAP)


# --------------------------------------------------------------------------------------
# per-axis cheap scorers.  Each returns (axis, set_of_learnability_cheap_keys_it_read).
# Only learnability-tracked keys go in the returned set (guarded by assert_no_leakage);
# pipeline metadata (generator / multipoint_order / role tags) is provenance-only.
# --------------------------------------------------------------------------------------

def _structural_axis(rec: dict) -> Tuple[AxisEvidenceV7, Set[str]]:
    signals: List[float] = []
    prov: List[str] = []
    used: Set[str] = set()
    ddg = _num(rec, "ddg_fold")
    if ddg is not None:                     # lower ddG (more stabilizing) -> higher score
        signals.append(_clamp01(0.5 - ddg / 4.0))
        prov.append("ddg_fold"); used.add("ddg_fold")
    stab = _num(rec, "stability_score")     # assumed already a [0,1] higher-better prior
    if stab is not None:
        signals.append(_clamp01(stab))
        prov.append("stability_score"); used.add("stability_score")
    if not signals:
        return _weak_axis(AXIS_STRUCTURAL, "no_cheap_structural_signal"), used
    return _cheap_axis(AXIS_STRUCTURAL, sum(signals) / len(signals), prov, len(signals)), used


def _evolutionary_axis(rec: dict) -> Tuple[AxisEvidenceV7, Set[str]]:
    signals: List[float] = []
    prov: List[str] = []
    used: Set[str] = set()
    perm = _num(rec, "msa_permissiveness")
    perm_key = "msa_permissiveness"
    if perm is None:
        perm = _num(rec, "msa_freq"); perm_key = "msa_freq"
    if perm is not None:
        signals.append(_clamp01(perm)); prov.append(perm_key); used.add(perm_key)
    esm = _num(rec, "esm_permissive")
    if esm is not None:
        signals.append(_clamp01(esm)); prov.append("esm_permissive"); used.add("esm_permissive")
    cons = _num(rec, "conservation")
    if cons is not None:                    # low conservation -> more tolerant
        signals.append(_clamp01(1.0 - cons)); prov.append("conservation"); used.add("conservation")
    sub = _num(rec, "subfamily_specific")
    if sub is not None:
        signals.append(_clamp01(sub)); prov.append("subfamily_specific"); used.add("subfamily_specific")
    if not signals:
        return _weak_axis(AXIS_EVOLUTIONARY, "no_cheap_evolutionary_signal"), used
    return _cheap_axis(AXIS_EVOLUTIONARY, sum(signals) / len(signals), prov, len(signals)), used


def _ligand_axis(rec: dict) -> Tuple[AxisEvidenceV7, Set[str]]:
    signals: List[float] = []
    prov: List[str] = []
    used: Set[str] = set()
    # any 'distance_to_<role>' key is a generic proximity prior (closer shell -> higher score)
    for key in sorted(rec):
        if key.startswith("distance_to_") and learnability.is_cheap(key):
            d = _num(rec, key)
            if d is not None:
                signals.append(_clamp01(1.0 - d / _LIGAND_SHELL_A))
                prov.append(key); used.add(key)
    nc = _num(rec, "n_contacts")
    if nc is not None:
        signals.append(_clamp01(nc / _CONTACT_SATURATION))
        prov.append("n_contacts"); used.add("n_contacts")
    if not signals:
        return _deferred_ligand_axis(), used
    return _cheap_axis(AXIS_LIGAND, sum(signals) / len(signals), prov, len(signals)), used


def _mutation_positions(mutation: str) -> Set[int]:
    return {int(m) for m in re.findall(r"\d+", mutation or "")}


def _catalytic_positions(mechanism) -> Set[int]:
    """Seqids of the mechanism's catalytic residues, generically (e.g. 'ARG290' -> 290).
    Empty when no mechanism is supplied."""
    out: Set[int] = set()
    for cr in (getattr(mechanism, "catalytic_residues", None) or []):
        res = getattr(cr, "residue", None)
        if res is None and isinstance(cr, dict):
            res = cr.get("residue")
        for m in re.findall(r"\d+", str(res or "")):
            out.add(int(m))
    return out


def _mechanism_axis(rec: dict, mechanism, deconvolution_of: Optional[str]
                    ) -> Tuple[AxisEvidenceV7, Set[str]]:
    """A multipoint active-site design, a role-tagged mutation, or a deconvolution probe
    encodes a more explicit mechanism hypothesis than a lone statistical pick."""
    prov: List[str] = []
    used: Set[str] = set()
    n_signals = 0

    gen = str(rec.get("generator") or "")
    if gen:
        gen_score = _GENERATOR_HYPOTHESIS_PRIOR.get(gen, _GENERATOR_PRIOR_DEFAULT)
        prov.append("generator"); n_signals += 1
    else:
        gen_score = 0.0

    order = _num(rec, "multipoint_order")           # pipeline metadata, not a learnability feat
    if order is not None:
        combo = order; prov.append("multipoint_order"); n_signals += 1
    else:
        combo = _num(rec, "n_mutations")            # cheap feature fallback
        if combo is not None:
            prov.append("n_mutations"); used.add("n_mutations"); n_signals += 1
    mp = _clamp01((combo - 1.0) / _COMBINATORIAL_SATURATION) if combo is not None else 0.0

    role_hit = False
    if rec.get("ligand_roles"):
        role_hit = True; prov.append("ligand_roles"); n_signals += 1
    elif _mutation_positions(str(rec.get("mutation_string") or rec.get("mutation") or "")) \
            & _catalytic_positions(mechanism):
        role_hit = True; prov.append("catalytic_residue_match"); n_signals += 1

    deconv_bonus = 0.0
    if deconvolution_of or rec.get("deconvolution_of"):
        deconv_bonus = 0.15; prov.append("deconvolution_of"); n_signals += 1

    if not prov:
        return _weak_axis(AXIS_MECHANISM, "no_cheap_mechanism_signal"), used
    score = 0.45 * gen_score + 0.25 * mp + 0.30 * (1.0 if role_hit else 0.0) + deconv_bonus
    return _cheap_axis(AXIS_MECHANISM, score, prov, n_signals), used


def _uncertainty_axis(rec: dict) -> Tuple[AxisEvidenceV7, Set[str]]:
    """Higher score == MORE uncertain (learning value): sparse MSA, high mutation load, low
    subfamily support. Never rejects anything — it only marks what is worth learning about."""
    signals: List[float] = []
    prov: List[str] = []
    used: Set[str] = set()
    gap = _num(rec, "gap_freq")
    if gap is not None:                     # sparser MSA column -> more uncertain
        signals.append(_clamp01(gap)); prov.append("gap_freq"); used.add("gap_freq")
    nmut = _num(rec, "n_mutations")
    if nmut is not None:                    # more simultaneous mutations -> more uncertain
        signals.append(_clamp01((nmut - 1.0) / _MUTATION_LOAD_SATURATION))
        prov.append("n_mutations"); used.add("n_mutations")
    sub = _num(rec, "subfamily_specific")
    if sub is not None:                     # low subfamily support -> more uncertain
        signals.append(_clamp01(1.0 - sub)); prov.append("subfamily_specific"); used.add("subfamily_specific")
    if not signals:
        return _weak_axis(AXIS_UNCERTAINTY, "no_cheap_uncertainty_signal"), used
    return _cheap_axis(AXIS_UNCERTAINTY, sum(signals) / len(signals), prov, len(signals)), used


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------

def build_cheap_ledgers(records: List[dict], *, mechanism=None, seed: int = 0
                        ) -> List[EvidenceLedgerV7]:
    """Build one Tier-0 seven-axis ledger per candidate record.

    The five ``CHEAP_AXES`` are scored from cheap features only; ``reaction_geometry_access``
    and ``portfolio_calibration_controls`` are left DEFERRED (auto-filled by the ledger). Every
    scored axis carries band ``unresolved`` (awaiting calibration by ``portfolio.stats``).
    Deterministic given ``seed`` (scoring is pure rule-based; ``seed`` is accepted for API
    parity with the tiers that do sample). Raises via ``assert_no_leakage`` if any expensive
    field ever reaches a cheap score.
    """
    _ = seed  # scoring is deterministic; no RNG is used at this tier
    ledgers: List[EvidenceLedgerV7] = []
    all_cheap_keys: Set[str] = set()

    for rec in records:
        variant_id = str(rec.get("candidate_id") or rec.get("variant_id") or "")
        mutation = str(rec.get("mutation_string") or rec.get("mutation") or "")
        generator = str(rec.get("generator") or "")
        deconv_of = rec.get("deconvolution_of")

        structural, k1 = _structural_axis(rec)
        evolutionary, k2 = _evolutionary_axis(rec)
        ligand, k3 = _ligand_axis(rec)
        mechanism_ax, k4 = _mechanism_axis(rec, mechanism, deconv_of)
        uncertainty, k5 = _uncertainty_axis(rec)
        all_cheap_keys |= (k1 | k2 | k3 | k4 | k5)

        led = EvidenceLedgerV7(
            variant_id=variant_id, mutation=mutation, generator=generator,
            axes={
                AXIS_STRUCTURAL: structural,
                AXIS_EVOLUTIONARY: evolutionary,
                AXIS_LIGAND: ligand,
                AXIS_MECHANISM: mechanism_ax,
                AXIS_UNCERTAINTY: uncertainty,
                # reaction_geometry + portfolio auto-fill as deferred by the ledger validator
            },
            deconvolution_of=(str(deconv_of) if deconv_of else None),
            tier_reached=TIER_CHEAP,
            tier_rationale="tier0_cheap_all_candidate_scoring",
        )
        ledgers.append(led)

    # mechanical guarantee: nothing expensive (a label source) ever fed a cheap SCORE.
    learnability.assert_no_leakage(sorted(all_cheap_keys))
    return ledgers


def ledger_from_evidence_card(card, *, rec: dict, is_control: bool = False,
                              deconvolution_of: Optional[str] = None) -> EvidenceLedgerV7:
    """Adapter: lift an existing ``mechanism.cards.EvidenceCard`` (6 axes) onto the V7 seven.

    Axis mapping (ROADMAP_V7 §3):
      * structural_viability / evolutionary_tolerance                -> same-named V7 axes
      * ligand_cofactor_competence                                   -> ligand_cofactor_metal_accommodation
      * substrate_positioning + reaction_geometry_accommodation      -> reaction_geometry_access,
        with the distance and angle observables kept SEPARATE in ``detail``
        (``{'distance': ..., 'angle_nac': ...}``) and the axis score conservatively the MIN of
        the two — a short distance never inflates the axis on its own (V7 §3 Axis 4).
      * uncertainty                                                  -> uncertainty_learning_value
      * mechanism_hypothesis_consistency                             -> derived from ``rec``
      * portfolio_calibration_controls                               -> control membership if a
        control, else DEFERRED (panel-level; the builder fills it).

    ``EvidenceCard`` is imported lazily so this module loads even without the mechanism cards.
    """
    from evoliez.mechanism.cards import EvidenceCard  # lazy: light-env safe, kept decoupled
    if not isinstance(card, EvidenceCard):
        raise TypeError(f"expected mechanism.cards.EvidenceCard, got {type(card)!r}")

    def _axis(v7_axis: str, ax, extra_prov: Optional[List[str]] = None) -> AxisEvidenceV7:
        prov = list(ax.evidence) + list(extra_prov or [])
        return AxisEvidenceV7(axis=v7_axis, score=_clamp01(ax.score), band=BAND_UNRESOLVED,
                              confidence=ax.confidence, provenance=prov,
                              claim_ceiling=CLAIM_L0_HYPOTHESIS, tier=TIER_CHEAP, missing=False)

    sp = card.substrate_positioning
    rg = card.reaction_geometry_accommodation
    # Reaction-geometry ACCESS = distance AND near-attack angle BOTH favourable (V7 §3 Axis 4).
    # The angle term is the ABSOLUTE near-attack occupancy when the record carries it: a 0
    # occupancy means the in-line conformation was never sampled, so a short distance ALONE must
    # not band the axis. (build_evidence_card's rg.score is delta-vs-WT-based and maps ΔNAC=0 to
    # a neutral 0.5 — which would let distance drive the axis when the site never reaches NAC;
    # that is the exact conflation §3 Axis 4 forbids.) Absolute occupancy is scaled by a modest
    # reference (a restrained implicit screen samples the near-attack only rarely), and ΔNAC-vs-WT
    # is retained in ``detail`` as a separate catalytic-gain observable, never folded into access.
    abs_occ = rec.get("nac_occupancy")
    if isinstance(abs_occ, (int, float)) and not isinstance(abs_occ, bool):
        angle_score = _clamp01(float(abs_occ) / _NAC_ACCESS_REFERENCE)
        angle_prov = ["nac_occupancy_absolute"]
    else:
        angle_score = rg.score                                  # no absolute occupancy -> card score
        angle_prov = list(rg.evidence)
    reaction_geometry = AxisEvidenceV7(
        axis=AXIS_REACTION_GEOMETRY,
        score=_clamp01(min(sp.score, angle_score)),    # both must hold; distance != NAC
        band=BAND_UNRESOLVED,
        confidence=_weaker_confidence(sp.confidence, rg.confidence),
        provenance=(["substrate_positioning_distance", "near_attack_angle_occupancy"]
                    + list(sp.evidence) + angle_prov),
        detail={"distance": float(sp.score), "angle_nac_access": float(angle_score),
                "nac_delta_vs_wt_score": float(rg.score)},
        claim_ceiling=CLAIM_L0_HYPOTHESIS, tier=TIER_CHEAP, missing=False,
    )

    mechanism_ax, _ = _mechanism_axis(rec, None, deconvolution_of)

    axes = {
        AXIS_STRUCTURAL: _axis(AXIS_STRUCTURAL, card.structural_viability),
        AXIS_EVOLUTIONARY: _axis(AXIS_EVOLUTIONARY, card.evolutionary_tolerance),
        AXIS_LIGAND: _axis(AXIS_LIGAND, card.ligand_cofactor_competence),
        AXIS_REACTION_GEOMETRY: reaction_geometry,
        AXIS_MECHANISM: mechanism_ax,
        AXIS_UNCERTAINTY: _axis(AXIS_UNCERTAINTY, card.uncertainty),
    }

    control_role = rec.get("control_role") or rec.get("control_type")
    if is_control:
        axes[AXIS_PORTFOLIO] = AxisEvidenceV7(
            axis=AXIS_PORTFOLIO, score=1.0, band=BAND_CONTROL, confidence=CONF_LOW,
            provenance=[f"control:{control_role}" if control_role else "control"],
            claim_ceiling=CLAIM_L0_HYPOTHESIS, tier=TIER_CHEAP, missing=False)
    # else: portfolio auto-fills as deferred (panel-level; the builder assigns it)

    led = EvidenceLedgerV7(
        variant_id=str(card.variant_id),
        mutation=str(rec.get("mutation_string") or rec.get("mutation") or ""),
        generator=str(rec.get("generator") or ""),
        axes=axes,
        is_control=bool(is_control),
        control_role=(str(control_role) if (is_control and control_role) else None),
        deconvolution_of=(str(deconvolution_of) if deconvolution_of else None),
        tier_reached=TIER_CHEAP,
        tier_rationale="lifted_from_evidence_card",
    )
    led.recompute_overall_band()
    return led


# fields whose presence marks that a candidate reached an expensive tier. MD keys upgrade
# structural/ligand/uncertainty with real physics; a VALID NAC (a real occupancy/delta, not the
# 'invalid_cosubstrate_diffused'/null case) additionally fills the reaction-geometry axis.
_MD_EVIDENCE_KEYS = ("md_instability", "md_lite_score", "nac_occupancy", "nac_delta_vs_wt")
_NAC_KEYS = ("nac_occupancy", "nac_delta_vs_wt")
_S09_EVIDENCE_KEYS = ("redocking_consistency", "docking_uncertainty",
                      "catalytic_distance_mean", "hbond_occupancy", "pocket_rmsd_mean")


def _has_any(rec: dict, keys) -> bool:
    return any(rec.get(k) is not None for k in keys)


def _has_valid_nac(rec: dict) -> bool:
    """A real (numeric) ABSOLUTE near-attack occupancy — the reaction-geometry ACCESS measure.
    Requires ``nac_occupancy`` specifically: ΔNAC-vs-WT alone is NOT sufficient (Fable review —
    ΔNAC=0 maps to a neutral 0.5, which let a short DISTANCE band the axis with no angle support,
    the exact §3 Axis-4 conflation). A null / absent occupancy leaves the axis DEFERRED (an
    honest gap), never a fabricated 0 that would pollute the null."""
    v = rec.get("nac_occupancy")
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def enrich_expensive_axes(records: List[dict], ledgers: List[EvidenceLedgerV7], *,
                          mechanism=None) -> int:
    """Fill/upgrade the physics axes for the SUBSET of candidates that reached the expensive
    tiers (ROADMAP_V7 §5 Tier 1-3), from the run's real s09 (docking/stability) + s10 (MD/NAC)
    provenance. This is what turns the DEFERRED reaction-geometry axis into real evidence — for
    the MD subset only — so the bands over it are honestly subset-level (§4.2 / Gate 5).

    For each candidate whose record carries MD or s09 evidence, an ``EvidenceCard`` is built and
    lifted to the V7 seven axes; the physics axes (reaction-geometry when MD is present, plus
    real structural / ligand / uncertainty) REPLACE the thin cheap axes, tagged with the
    fidelity tier that produced them and ``subset_level=True``. Evolutionary + mechanism axes
    are left as the cheap ledger set them (homology / design-intent do not change with MD).
    Mutates ``ledgers`` in place; returns the number of candidates enriched.
    """
    from evoliez.ranking.evidence_card import build_evidence_card  # lazy: light-env safe

    by_id: Dict[str, EvidenceLedgerV7] = {led.variant_id: led for led in ledgers}
    n_enriched = 0
    for rec in records:
        cid = str(rec.get("candidate_id") or rec.get("variant_id") or "")
        led = by_id.get(cid)
        if led is None:
            continue
        has_md = _has_any(rec, _MD_EVIDENCE_KEYS)
        has_s09 = _has_any(rec, _S09_EVIDENCE_KEYS)
        if not (has_md or has_s09):
            continue
        card = build_evidence_card(rec)
        card_led = ledger_from_evidence_card(card, rec=rec)
        tier = TIER_FOCUSED if has_md else TIER_GPU_BROAD
        # reaction-geometry ONLY when the MD produced a VALID NAC (a real occupancy/delta). An
        # s09-only candidate, or an MD whose co-substrate diffused (null NAC), leaves the axis
        # DEFERRED — a short docking distance is never a near-attack conformation (V7 §3 Axis 4).
        merge = [AXIS_STRUCTURAL, AXIS_LIGAND, AXIS_UNCERTAINTY]
        if has_md and _has_valid_nac(rec):
            merge.append(AXIS_REACTION_GEOMETRY)
        for axis in merge:
            src = card_led.axes[axis]
            if src.missing:
                continue
            src.tier = tier
            src.subset_level = True
            src.claim_ceiling = CLAIM_L1_SCREENING   # real physics -> screening-level ceiling
            src.provenance = list(src.provenance) + [f"enriched_{tier}"]
            led.axes[axis] = src
        led.tier_reached = tier
        led.tier_rationale = (
            f"enriched from {'MD/NAC (s10)' if has_md else 's09 docking/stability'} evidence")
        n_enriched += 1
    return n_enriched
