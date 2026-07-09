"""V7 mechanism-ranked portfolio builder (ROADMAP_V7 Phase V7-6, §8).

The builder is the last hop between calibrated evidence and a wet-lab plate. It does NOT
re-rank candidates by a scalar; it reads the seven-axis evidence bands (already filled by
``portfolio.bands``) and lays out a claim-safe, sub-50 experimental panel across the V7
lanes (ROADMAP_V7 §7/§8):

  * ``wt_parental``       — the parental reference, always first (calibration baseline).
  * ``strong_significant``— candidates whose OVERALL band is strong/significant.
  * ``cross_axis_consensus`` — candidates promoted to the consensus band.
  * ``deconvolution``     — single/pairwise probes of the strongest MULTIPOINT hypotheses,
                            so a multipoint's evidence can be attributed to its parts.
  * ``single_site_probe`` — clean single substitutions in any band.
  * ``control``           — scalar-top / geometry-weak / low-signal / random controls that
                            calibrate the panel (they are design controls, not hypotheses).
  * ``uncertainty_probe`` — the highest-disagreement candidates, chosen for learning value.

Design invariants (ROADMAP_V7 §2/§8):

  * Bands over top-N. Lane membership follows the calibrated ``overall_band`` / consensus
    set, never a raw scalar rank. The panel is still useful when NO candidate reached a
    strong band — it fills from consensus / exploratory / controls.
  * Missing evidence never sinks a candidate; a deferred axis simply does not add support.
  * Claims stay bounded. Every ``reason_to_test`` is assembled from claim-safe phrasing
    only ("prioritized for experimental testing", "screening-level evidence", "hypothesis-
    grade mechanism probe", ...); the panel ``claim_ceiling`` never exceeds ``L1_screening``
    pre-wet-lab. ClaimGuard (``ranking.claim_guard.assert_clean``) must pass on the joined
    reasons.
  * Mechanism-generic. No enzyme name / residue list is hardcoded; lanes are resolved from
    the ledger contract, the supplied controls, and the ';'-joined mutation strings.
  * Deterministic given a fixed seed and a fixed bundle (ties break on ``variant_id``).

Pure schema + stdlib; imports in the light env (no numpy needed to build a panel).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

from evoliez.experimental.deconvolution import deconvolution_set
from evoliez.portfolio.ledger import (
    AXIS_UNCERTAINTY, BAND_CONSENSUS, BAND_CONTROL, BAND_SIGNIFICANT, BAND_STRONG,
    BAND_UNRESOLVED, CLAIM_L0_HYPOTHESIS, CLAIM_L1_SCREENING, LANE_CONSENSUS, LANE_CONTROL,
    LANE_DECONVOLUTION, LANE_PROTECTED, LANE_SINGLE_SITE, LANE_STRONG, LANE_UNCERTAINTY,
    LANE_WT, ALL_LANES, EvidenceLedgerV7, LedgerBundle, band_rank, panel_layer, strongest_band,
)

if TYPE_CHECKING:                               # sibling module (portfolio.controls); the
    from evoliez.portfolio.controls import ControlSpec  # builder only duck-types it.

_WT_CONTROL_TYPE = "wt_parental"
_VALID_PANEL_SIZES = (24, 32, 48)

# ROADMAP_V7 §8 default lane composition per panel budget. Each row sums to the panel size
# (WT included), so filling every lane up to its quota can never exceed the budget, and the
# protected lanes (WT / control / deconvolution) always keep their reserved seats. Smaller
# budgets trim uncertainty + consensus hardest and preserve controls + deconvolution.
_LANE_QUOTA = {
    48: {LANE_WT: 1, LANE_STRONG: 16, LANE_CONSENSUS: 7, LANE_DECONVOLUTION: 9,
         LANE_SINGLE_SITE: 5, LANE_CONTROL: 6, LANE_UNCERTAINTY: 4},
    32: {LANE_WT: 1, LANE_STRONG: 11, LANE_CONSENSUS: 5, LANE_DECONVOLUTION: 6,
         LANE_SINGLE_SITE: 3, LANE_CONTROL: 4, LANE_UNCERTAINTY: 2},
    24: {LANE_WT: 1, LANE_STRONG: 8, LANE_CONSENSUS: 4, LANE_DECONVOLUTION: 4,
         LANE_SINGLE_SITE: 2, LANE_CONTROL: 3, LANE_UNCERTAINTY: 2},
}

# When a custom over-subscribed quota forces a trim, cut from these lanes first (weakest
# member first); WT / control / deconvolution / mechanism-protected keep their reserved seats.
_TRIM_ORDER = (LANE_UNCERTAINTY, LANE_CONSENSUS, LANE_SINGLE_SITE, LANE_STRONG)

# Mandatory caveat when the run produced NO statistical evidence band (reviewer directive):
# stated on the panel + surfaced in the CSV so an experimenter can never mistake a calibration
# panel for a computationally selected lead set. Claim-safe (no activity/kcat/validated-lead).
_NO_BAND_CAVEAT = (
    "No candidate reached a strong / significant / consensus statistical evidence band in this "
    "run; this panel is designed for calibration and mechanism probing, not as a computationally "
    "selected lead set.")


class _Base(BaseModel):
    model_config = {"extra": "forbid"}


class PortfolioVariant(_Base):
    """One seat on the experimental panel (shared V7 contract shape)."""

    variant_id: str
    mutation: str
    lane: str
    overall_band: str
    reason_to_test: str
    is_control: bool = False
    control_role: Optional[str] = None
    deconvolution_of: Optional[str] = None
    significant_axes: List[str] = Field(default_factory=list)
    tier_reached: str = "tier0_cheap"


class Portfolio(_Base):
    """The assembled sub-50 panel (shared V7 contract shape)."""

    panel_size: int
    target_id: str = ""
    mechanism_class: str = ""
    variants: List[PortfolioVariant] = Field(default_factory=list)
    lane_counts: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    claim_ceiling: str = CLAIM_L0_HYPOTHESIS

    def as_rows(self) -> List[dict]:
        """Flat, report-friendly rows (one dict per variant, contract key order)."""
        return [
            {
                "variant_id": v.variant_id,
                "mutation": v.mutation,
                "lane": v.lane,
                "panel_layer": panel_layer(v.lane),   # statistical band vs protected vs probe
                "overall_band": v.overall_band,
                "reason_to_test": v.reason_to_test,
                "is_control": v.is_control,
                "control_role": v.control_role,
                "deconvolution_of": v.deconvolution_of,
                "significant_axes": list(v.significant_axes),
                "tier_reached": v.tier_reached,
            }
            for v in self.variants
        ]


# --- helpers -------------------------------------------------------------------------
def default_lane_quota(panel_size: int) -> Dict[str, int]:
    """The ROADMAP_V7 §8 lane composition for a 24 / 32 / 48-seat panel (WT included)."""
    if panel_size not in _LANE_QUOTA:
        raise ValueError(f"panel_size must be one of {_VALID_PANEL_SIZES}, got {panel_size}")
    return dict(_LANE_QUOTA[panel_size])


def _dedup_key(mutation: str, variant_id: str) -> str:
    """First-lane-wins dedup key. Distinct empty mutations must not collapse together."""
    m = (mutation or "").strip()
    return m if m else f"__id__::{variant_id}"


def _min_q(led: EvidenceLedgerV7) -> float:
    """Smallest finite axis q-value on a ledger (missing / NaN -> +inf, i.e. weakest)."""
    best = float("inf")
    for ax in led.axes.values():
        q = ax.q_value
        if q is not None and q == q:            # not None and not NaN
            best = min(best, float(q))
    return best


def _is_multipoint(mutation: str) -> bool:
    return ";" in (mutation or "")


def _significant(led: EvidenceLedgerV7) -> List[str]:
    return led.significant_axes()


# --- reason_to_test sentences (claim-safe: allowed vocab ONLY) -----------------------
def _reason_wt() -> str:
    return ("Wild-type parental reference: baseline calibration control for the "
            "experimental panel.")


def _reason_protected(mutation: str, *, is_deconv: bool, parent: str = "") -> str:
    if is_deconv:
        return (f"Mechanism-protected deconvolution probe of the expert hypothesis {parent}: "
                f"hypothesis-grade mechanism probe, included regardless of statistical bands to "
                f"attribute the hypothesis to its component substitutions; prioritized for "
                f"experimental testing.")
    return (f"Mechanism-protected expert hypothesis: hypothesis-grade mechanism probe, forced "
            f"into the panel independent of computational evidence bands so the curated "
            f"hypothesis is experimentally tested; prioritized for experimental testing.")


def _reason_strong(led: EvidenceLedgerV7) -> str:
    axes = _significant(led)
    across = ", ".join(axes) if axes else "the calibrated axes"
    return (f"Prioritized for experimental testing: statistically supported evidence band "
            f"({led.overall_band}) with multi-axis support across {across}.")


def _reason_consensus(led: EvidenceLedgerV7) -> str:
    axes = list(led.consensus_axes) or _significant(led)
    across = ", ".join(axes) if axes else "the calibrated axes"
    return (f"Prioritized for experimental testing: cross-axis consensus screening-level "
            f"evidence across {across}.")


def _reason_deconvolution(parent: PortfolioVariant) -> str:
    return (f"Deconvolution probe of multipoint hypothesis {parent.mutation} "
            f"({parent.variant_id}): hypothesis-grade mechanism probe to attribute the "
            f"evidence band to its component substitutions.")


def _reason_single_site(led: EvidenceLedgerV7) -> str:
    return (f"Single-substitution probe carrying screening-level evidence "
            f"(band {led.overall_band}); prioritized for experimental testing.")


def _reason_control(control_type: str) -> str:
    return (f"Panel calibration control ({control_type}): included to calibrate the "
            f"evidence bands, not selected as a prioritized hypothesis.")


def _reason_uncertainty(led: EvidenceLedgerV7) -> str:
    score = led.axes[AXIS_UNCERTAINTY].score
    return (f"Uncertainty probe (uncertainty score {score:.2f}): highest model disagreement "
            f"remaining in the panel; learning objective is to reduce that disagreement and "
            f"calibrate the uncertainty-learning axis for the next round.")


def _reason_exploratory(led: EvidenceLedgerV7) -> str:
    return ("Exploratory single-site probe: no calibrated significant band, included for "
            "design-space coverage so the first wet-lab round can estimate axis-level "
            "enrichment; screening-level at most, prioritized for experimental testing.")


def _cheap_composite(led: EvidenceLedgerV7) -> float:
    """A coarse higher-is-better ranker over the non-missing cheap axes (uncertainty inverted),
    used ONLY to order EXPLORATORY fill picks — never to define a band (bands come from the
    calibrated stats, never a scalar)."""
    from evoliez.portfolio.ledger import HIGHER_IS_WORSE_AXES, SEVEN_AXES
    vals = []
    for ax in SEVEN_AXES:
        ev = led.axes[ax]
        if ev.missing:
            continue
        vals.append((1.0 - ev.score) if ax in HIGHER_IS_WORSE_AXES else ev.score)
    return sum(vals) / len(vals) if vals else 0.0


# --- the builder ---------------------------------------------------------------------
def _place_protected(picks: List[PortfolioVariant], protected: List[str],
                     do_deconv: bool, by_mutation: Dict[str, EvidenceLedgerV7],
                     take) -> int:
    """Place the mechanism-protected hypotheses (Layer 1). Each protected mutation uses its real
    ledger when the universe generated it, else a SYNTHESISED deferred probe (variant_id
    ``protected::<mut>``, all axes deferred — no fabricated evidence). With ``do_deconv`` each
    multipoint hypothesis also injects its single/pairwise components. Returns the count placed."""
    n = 0
    for raw in protected:
        parent = (raw or "").strip()
        if not parent:
            continue
        members = [(parent, False, "")]
        if do_deconv and _is_multipoint(parent):
            members += [(s.strip(), True, parent) for s in deconvolution_set(parent)
                        if s.strip() and s.strip() != parent]
        for mut, is_deconv, par in members:
            led = by_mutation.get(mut)
            vid = led.variant_id if led is not None else f"protected::{mut}"
            if not take(mut, vid):
                continue
            picks.append(PortfolioVariant(
                variant_id=vid, mutation=mut, lane=LANE_PROTECTED,
                overall_band=(led.overall_band if led is not None else BAND_UNRESOLVED),
                reason_to_test=_reason_protected(mut, is_deconv=is_deconv, parent=par),
                deconvolution_of=(par or None),
                significant_axes=(_significant(led) if led is not None else []),
                tier_reached=(led.tier_reached if led is not None else "tier0_cheap")))
            n += 1
    return n


def build_portfolio(bundle: LedgerBundle, *, panel_size: int = 48,
                    controls: Optional[List["ControlSpec"]] = None,
                    lane_quota: Optional[Dict[str, int]] = None,
                    protected: Optional[List[str]] = None,
                    protected_deconvolution: bool = True,
                    seed: int = 0) -> Portfolio:
    """Assemble a mechanism-ranked, claim-safe sub-50 panel from calibrated evidence bands.

    ``panel_size`` must be 24 / 32 / 48. Lanes are filled in the ROADMAP_V7 §8 order (WT,
    strong/significant, consensus, deconvolution, single-site, controls, uncertainty),
    deduped by mutation (first lane wins) and capped per lane by ``lane_quota`` (a partial
    override merges onto :func:`default_lane_quota`). If a custom quota over-subscribes the
    budget, uncertainty + weak-consensus seats are trimmed first; WT, controls, and
    deconvolution probes are preserved. Deterministic given ``bundle`` + ``seed``.
    """
    if panel_size not in _VALID_PANEL_SIZES:
        raise ValueError(f"panel_size must be one of {_VALID_PANEL_SIZES}, got {panel_size}")

    quota = default_lane_quota(panel_size)
    if lane_quota:
        quota.update({k: int(v) for k, v in lane_quota.items()})

    controls = list(controls or [])
    ledgers = list(bundle.ledgers)
    by_mutation: Dict[str, EvidenceLedgerV7] = {}
    for led in ledgers:                          # first occurrence wins the mutation string
        by_mutation.setdefault((led.mutation or "").strip(), led)

    picks: List[PortfolioVariant] = []
    taken: set = set()                           # dedup keys already placed on the panel

    def _take(mutation: str, variant_id: str) -> bool:
        key = _dedup_key(mutation, variant_id)
        if key in taken:
            return False
        taken.add(key)
        return True

    # 1) WT parental control (always first). Sourced from a wt_parental ControlSpec if
    #    present, else synthesized so the calibration baseline is never silently absent.
    wt_spec = next((c for c in controls if getattr(c, "control_type", "") == _WT_CONTROL_TYPE),
                   None)
    wt_id = getattr(wt_spec, "variant_id", "") or "WT"
    wt_mut = getattr(wt_spec, "mutation", "") or "WT"
    if _take(wt_mut, wt_id):
        picks.append(PortfolioVariant(
            variant_id=wt_id, mutation=wt_mut, lane=LANE_WT, overall_band=BAND_CONTROL,
            reason_to_test=_reason_wt(), is_control=True, control_role=_WT_CONTROL_TYPE))

    # 1.5) MECHANISM-PROTECTED hypotheses (Layer 1, reviewer breakthrough) — forced in BEFORE the
    #      statistical lanes and never trimmed, so a hard target with 0 significant bands still
    #      tests the curated hypotheses. Each protected mutation uses its real ledger if the
    #      universe generated it, else a synthesised deferred probe (no fabricated evidence);
    #      with protected_deconvolution, each multipoint's single/pairwise components are added.
    n_protected = _place_protected(
        picks, protected or [], protected_deconvolution, by_mutation, _take)

    # 2) strong / significant overall band (rank by band, then q, then id).
    strong = sorted(
        (l for l in ledgers if l.overall_band in (BAND_STRONG, BAND_SIGNIFICANT)),
        key=lambda l: (band_rank(l.overall_band), _min_q(l), l.variant_id))
    for led in strong[: quota.get(LANE_STRONG, 0)]:
        if _take(led.mutation, led.variant_id):
            picks.append(PortfolioVariant(
                variant_id=led.variant_id, mutation=led.mutation, lane=LANE_STRONG,
                overall_band=led.overall_band, reason_to_test=_reason_strong(led),
                significant_axes=_significant(led), tier_reached=led.tier_reached))

    # 3) cross-axis consensus band (rank by consensus breadth, then q, then id).
    consensus = sorted(
        (l for l in ledgers if l.overall_band == BAND_CONSENSUS),
        key=lambda l: (-len(l.consensus_axes), _min_q(l), l.variant_id))
    for led in consensus[: quota.get(LANE_CONSENSUS, 0)]:
        if _take(led.mutation, led.variant_id):
            picks.append(PortfolioVariant(
                variant_id=led.variant_id, mutation=led.mutation, lane=LANE_CONSENSUS,
                overall_band=led.overall_band, reason_to_test=_reason_consensus(led),
                significant_axes=_significant(led), tier_reached=led.tier_reached))

    # 4) deconvolution of the strongest MULTIPOINT hypotheses. Expand each parent into its
    #    single/pairwise components and include the ones that EXIST as candidates.
    multipoints = sorted(
        (l for l in ledgers if _is_multipoint(l.mutation)),
        key=lambda l: (band_rank(l.overall_band), _min_q(l), l.variant_id))
    deconv_cap = quota.get(LANE_DECONVOLUTION, 0)
    n_deconv = 0
    for parent in multipoints:
        if n_deconv >= deconv_cap:
            break
        parent_view = PortfolioVariant(
            variant_id=parent.variant_id, mutation=parent.mutation, lane=LANE_DECONVOLUTION,
            overall_band=parent.overall_band, reason_to_test="placeholder")
        for sub in deconvolution_set(parent.mutation):
            if n_deconv >= deconv_cap:
                break
            if sub.strip() == (parent.mutation or "").strip():
                continue                         # the full multipoint is not its own probe
            child = by_mutation.get(sub.strip())
            if child is None:
                continue                         # probe not synthesized as a candidate
            if not _take(child.mutation, child.variant_id):
                continue                         # already placed in an earlier lane
            picks.append(PortfolioVariant(
                variant_id=child.variant_id, mutation=child.mutation,
                lane=LANE_DECONVOLUTION, overall_band=child.overall_band,
                reason_to_test=_reason_deconvolution(parent_view),
                deconvolution_of=parent.variant_id,
                significant_axes=_significant(child), tier_reached=child.tier_reached))
            n_deconv += 1

    # 5) clean single-substitution probes in ANY band.
    singles = sorted(
        (l for l in ledgers if not _is_multipoint(l.mutation)),
        key=lambda l: (band_rank(l.overall_band), _min_q(l), l.variant_id))
    n_single = 0
    single_cap = quota.get(LANE_SINGLE_SITE, 0)
    for led in singles:
        if n_single >= single_cap:
            break
        if _take(led.mutation, led.variant_id):
            picks.append(PortfolioVariant(
                variant_id=led.variant_id, mutation=led.mutation, lane=LANE_SINGLE_SITE,
                overall_band=led.overall_band, reason_to_test=_reason_single_site(led),
                significant_axes=_significant(led), tier_reached=led.tier_reached))
            n_single += 1

    # 6) remaining (non-WT) design controls.
    n_control = 0
    control_cap = quota.get(LANE_CONTROL, 0)
    for c in controls:
        if n_control >= control_cap:
            break
        ctype = getattr(c, "control_type", "")
        if ctype == _WT_CONTROL_TYPE:
            continue                             # WT already placed in lane 1
        cid = getattr(c, "variant_id", "")
        cmut = getattr(c, "mutation", "")
        if _take(cmut, cid):
            picks.append(PortfolioVariant(
                variant_id=cid, mutation=cmut, lane=LANE_CONTROL, overall_band=BAND_CONTROL,
                reason_to_test=_reason_control(ctype or "control"), is_control=True,
                control_role=ctype or "control"))
            n_control += 1

    # 7) highest-uncertainty candidates as learning probes (skip deferred uncertainty).
    uncertain = sorted(
        (l for l in ledgers if not l.axes[AXIS_UNCERTAINTY].missing),
        key=lambda l: (-float(l.axes[AXIS_UNCERTAINTY].score), l.variant_id))
    n_unc = 0
    unc_cap = quota.get(LANE_UNCERTAINTY, 0)
    for led in uncertain:
        if n_unc >= unc_cap:
            break
        if _take(led.mutation, led.variant_id):
            picks.append(PortfolioVariant(
                variant_id=led.variant_id, mutation=led.mutation, lane=LANE_UNCERTAINTY,
                overall_band=led.overall_band, reason_to_test=_reason_uncertainty(led),
                significant_axes=_significant(led), tier_reached=led.tier_reached))
            n_unc += 1

    # 8) EXPLORATORY fill: when the significance-driven lanes underfill the budget (thin /
    #    uniform evidence, e.g. a hard target where no axis reaches a band), use the remaining
    #    seats for the best untaken SINGLE-site candidates so the panel is still a useful
    #    axis-enrichment calibration set (ROADMAP_V7 §7 / Gate 7). Interpretable singles only —
    #    an unbanded multipoint cannot be attributed, so it adds no calibration value here.
    remaining = panel_size - len(picks)
    if remaining > 0:
        explor = sorted(
            (l for l in ledgers if not _is_multipoint(l.mutation)
             and _dedup_key(l.mutation, l.variant_id) not in taken),
            key=lambda l: (-_cheap_composite(l), l.variant_id))
        for led in explor[:remaining]:
            if _take(led.mutation, led.variant_id):
                picks.append(PortfolioVariant(
                    variant_id=led.variant_id, mutation=led.mutation, lane=LANE_SINGLE_SITE,
                    overall_band=led.overall_band, reason_to_test=_reason_exploratory(led),
                    significant_axes=_significant(led), tier_reached=led.tier_reached))

    # Global budget guard: with the default quota (which sums to panel_size) this is a
    # no-op; a custom over-subscribed quota trims uncertainty + weak consensus first and
    # preserves WT / controls / deconvolution.
    picks = _trim_to_budget(picks, panel_size)

    lane_counts = OrderedDict((lane, 0) for lane in ALL_LANES)
    for v in picks:
        lane_counts[v.lane] = lane_counts.get(v.lane, 0) + 1

    ceiling = _panel_claim_ceiling(picks)
    n_statistical = lane_counts.get(LANE_STRONG, 0) + lane_counts.get(LANE_CONSENSUS, 0)
    notes = [
        f"panel_size={panel_size}; seed={seed}; placed={len(picks)}",
        "lane_quota=" + ", ".join(f"{k}:{quota.get(k, 0)}" for k in ALL_LANES),
        "claim ceiling never exceeds L1_screening pre-wet-lab (ROADMAP_V7 §8/§11).",
    ]
    if n_protected:
        notes.append(f"{n_protected} mechanism-protected hypothesis probe(s) forced in (Layer 1).")
    if n_statistical == 0:
        notes.insert(0, _NO_BAND_CAVEAT)
    return Portfolio(
        panel_size=panel_size, target_id=bundle.target_id,
        mechanism_class=bundle.mechanism_class, variants=picks,
        lane_counts=dict(lane_counts), notes=notes, claim_ceiling=ceiling)


def _trim_to_budget(picks: List[PortfolioVariant], panel_size: int) -> List[PortfolioVariant]:
    if len(picks) <= panel_size:
        return picks
    excess = len(picks) - panel_size
    drop_ids: set = set()
    for lane in _TRIM_ORDER:
        if excess <= 0:
            break
        members = [v for v in picks if v.lane == lane]   # weakest last (sorted strongest-first)
        while members and excess > 0:
            drop_ids.add(id(members.pop()))
            excess -= 1
    trimmed = [v for v in picks if id(v) not in drop_ids]
    # Last-resort hard cap (should never trigger with protected lanes alone).
    return trimmed[:panel_size]


def _panel_claim_ceiling(picks: List[PortfolioVariant]) -> str:
    if not picks:
        return CLAIM_L0_HYPOTHESIS
    strongest = strongest_band(*(v.overall_band for v in picks))
    if strongest in (BAND_STRONG, BAND_SIGNIFICANT, BAND_CONSENSUS):
        return CLAIM_L1_SCREENING           # never above L1_screening pre-wet-lab
    return CLAIM_L0_HYPOTHESIS


__all__ = ["PortfolioVariant", "Portfolio", "build_portfolio", "default_lane_quota"]
