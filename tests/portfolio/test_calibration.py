"""V7 wet-lab calibration loop (ROADMAP_V7 Phase V7-7).

These tests build the calibration inputs DIRECTLY from the ledger contract (setting axis
scores by hand) and duck-type the portfolio with ``SimpleNamespace`` — no sibling builder
is imported. They assert the two calibration outputs: an axis that scored the hits enriches
above the null, hits below the selection cut are counted per lane, and the claim level +
round-2 acquisition stay calibrated and deterministic.
"""
from types import SimpleNamespace

import pytest

from evoliez.portfolio.calibration import (
    CLAIM_L0, CLAIM_L1, CLAIM_L2, AxisEnrichment, CalibrationReport,
    LaneFalseNegative, axis_enrichment, build_calibration_report,
    hits_by_variant_from_labels, hits_from_assay_file, lane_false_negative,
    round2_from_calibration,
)
from evoliez.portfolio.ledger import (
    AXIS_STRUCTURAL, AXIS_UNCERTAINTY, LANE_STRONG, LANE_UNCERTAINTY,
    AxisEvidenceV7, EvidenceLedgerV7, LedgerBundle,
)

N = 12
HIT_IDS = {f"v{i:02d}" for i in range(8, 12)}  # top-4 structural scorers are the hits


def _led(i: int) -> EvidenceLedgerV7:
    """A ledger whose STRUCTURAL score rises with i (v11 highest) and whose UNCERTAINTY
    score is high for the low structural scorers (so hits have low uncertainty)."""
    struct = i / (N - 1)
    unc = 1.0 - struct
    return EvidenceLedgerV7(
        variant_id=f"v{i:02d}",
        mutation=f"A{i + 1}G",
        axes={
            AXIS_STRUCTURAL: AxisEvidenceV7(axis=AXIS_STRUCTURAL, score=struct),
            AXIS_UNCERTAINTY: AxisEvidenceV7(axis=AXIS_UNCERTAINTY, score=unc),
        },
    )


def _bundle() -> LedgerBundle:
    return LedgerBundle(run_id="r", ledgers=[_led(i) for i in range(N)], n_candidates=N)


def _hits() -> dict:
    return {f"v{i:02d}": (f"v{i:02d}" in HIT_IDS) for i in range(N)}


# --- axis enrichment ------------------------------------------------------------------

def test_structural_axis_enriches_hits():
    ae = axis_enrichment(_bundle(), _hits(), ks=(4, 8))
    by_axis = {a.axis: a for a in ae}
    s = by_axis[AXIS_STRUCTURAL]
    # top-4 by structural are exactly the 4 hits -> enrichment = (4/4)/(4/12) = 3
    assert s.enrichment_top_k[4] is not None and s.enrichment_top_k[4] > 1.0
    assert s.enrichment_top_k[8] is not None and s.enrichment_top_k[8] > 1.0
    assert s.auc is not None and s.auc > 0.9
    assert s.n == N and s.n_hits == len(HIT_IDS)


def test_default_ks_has_enrichment_at_some_k():
    # default ks=(8,16,32): only k=8 is valid for n=12, and it must show enrichment > 1
    ae = axis_enrichment(_bundle(), _hits())
    s = {a.axis: a for a in ae}[AXIS_STRUCTURAL]
    assert s.enrichment_top_k[8] is not None and s.enrichment_top_k[8] > 1.0
    assert s.enrichment_top_k[16] is None  # k > n -> honest None, not a fabricated value


def test_uncertainty_axis_oriented_to_good_direction():
    # hits have LOW uncertainty; oriented (negated) uncertainty must predict hits, AUC > 0.5
    u = {a.axis: a for a in axis_enrichment(_bundle(), _hits(), ks=(4,))}[AXIS_UNCERTAINTY]
    assert u.auc is not None and u.auc > 0.5


def test_deferred_axis_is_honest_empty_not_zero():
    # ligand axis was never set -> auto-deferred -> n == 0, enrichment all None
    from evoliez.portfolio.ledger import AXIS_LIGAND
    lig = {a.axis: a for a in axis_enrichment(_bundle(), _hits(), ks=(4,))}[AXIS_LIGAND]
    assert lig.n == 0 and lig.n_hits == 0
    assert all(v is None for v in lig.enrichment_top_k.values())
    assert lig.auc is None


# --- lane false-negative accounting ---------------------------------------------------

def _portfolio(order):
    """Duck-typed portfolio: order is a list of (variant_id, lane)."""
    return SimpleNamespace(
        variants=[SimpleNamespace(variant_id=vid, lane=lane) for vid, lane in order])


def test_lane_false_negative_counts_hits_below_cut():
    # two lanes; put both hits of the uncertainty lane BELOW the cut, strong-lane hit above
    order = [
        ("v11", LANE_STRONG),        # rank 1 (hit, above cut)
        ("v00", LANE_STRONG),        # rank 2 (miss)
        ("v08", LANE_UNCERTAINTY),   # rank 3 (hit, below cut=2)
        ("v09", LANE_UNCERTAINTY),   # rank 4 (hit, below cut=2)
        ("v01", LANE_UNCERTAINTY),   # rank 5 (miss)
    ]
    hits = {"v11": True, "v00": False, "v08": True, "v09": True, "v01": False}
    lanes = {l.lane: l for l in lane_false_negative(_portfolio(order), hits, cutoff_rank=2)}
    assert lanes[LANE_STRONG].n == 2 and lanes[LANE_STRONG].n_hits == 1
    assert lanes[LANE_STRONG].hits_below_cut == 0        # its only hit is rank 1
    assert lanes[LANE_UNCERTAINTY].n == 3 and lanes[LANE_UNCERTAINTY].n_hits == 2
    assert lanes[LANE_UNCERTAINTY].hits_below_cut == 2   # both hits ranked below cut


def test_lane_false_negative_ignores_unknown_labels():
    order = [("v11", LANE_STRONG), ("vX", LANE_STRONG)]
    lanes = lane_false_negative(_portfolio(order), {"v11": True}, cutoff_rank=1)
    assert len(lanes) == 1 and lanes[0].n == 1  # vX has no label -> not counted


# --- report assembly + claim levels ---------------------------------------------------

def test_empty_results_is_l0_uncalibrated():
    rep = build_calibration_report(_bundle(), _portfolio([]), {})
    assert rep.claim_level == CLAIM_L0
    assert rep.n == 0 and rep.n_hits == 0
    assert rep.axis_enrichment == [] and rep.lane_false_negative == []
    assert any("L0" in note for note in rep.notes)


def test_results_without_controls_is_l1_screened():
    rep = build_calibration_report(_bundle(), _portfolio([]), _hits())
    assert rep.claim_level == CLAIM_L1
    assert rep.n == N and rep.n_hits == len(HIT_IDS)
    # the structural axis enrichment must have been surfaced in the report
    s = {a.axis: a for a in rep.axis_enrichment}[AXIS_STRUCTURAL]
    assert s.enrichment_top_k[8] is not None and s.enrichment_top_k[8] > 1.0


def test_controls_and_replicates_reach_l2():
    rep = build_calibration_report(
        _bundle(), _portfolio([]), _hits(), has_controls=True, replicates=2)
    assert rep.claim_level == CLAIM_L2


def test_controls_without_replicates_stays_l1():
    rep = build_calibration_report(
        _bundle(), _portfolio([]), _hits(), has_controls=True, replicates=1)
    assert rep.claim_level == CLAIM_L1


def test_report_accepts_list_of_tuples():
    tuples = [(f"v{i:02d}", f"v{i:02d}" in HIT_IDS) for i in range(N)]
    rep = build_calibration_report(_bundle(), _portfolio([]), tuples)
    assert rep.n == N and rep.n_hits == len(HIT_IDS)
    assert rep.claim_level == CLAIM_L1


# --- round-2 acquisition --------------------------------------------------------------

def test_round2_excludes_tested_and_prioritizes_enriched_axis():
    bundle = _bundle()
    rep = build_calibration_report(bundle, _portfolio([]), _hits())
    props = round2_from_calibration(rep, bundle, size=4, tested=HIT_IDS)
    ids = {p["variant_id"] for p in props}
    assert ids.isdisjoint(HIT_IDS)                     # never re-proposes tested variants
    assert len(props) == 4
    assert all(set(p) == {"variant_id", "mutation", "reason"} for p in props)
    # highest untested structural scorer (v07) should be an exploit pick
    exploit = [p for p in props if "experimental testing" in p["reason"]]
    assert any(p["variant_id"] == "v07" for p in exploit)
    # an explore slice is reserved for the most-uncertain candidate
    assert any("explore" in p["reason"] for p in props)


def test_round2_falls_back_when_no_axis_enriched():
    # a bundle whose hits are UNCORRELATED with any axis -> no enrichment -> cheap-axis prior
    bundle = _bundle()
    flat_hits = {f"v{i:02d}": (i % 2 == 0) for i in range(N)}
    rep = build_calibration_report(bundle, _portfolio([]), flat_hits)
    props = round2_from_calibration(rep, bundle, size=3)
    assert len(props) == 3  # still populated via fallback prior


def test_round2_deterministic():
    bundle = _bundle()
    rep = build_calibration_report(bundle, _portfolio([]), _hits())
    a = round2_from_calibration(rep, bundle, size=5, tested=HIT_IDS)
    b = round2_from_calibration(rep, bundle, size=5, tested=HIT_IDS)
    assert a == b


def test_full_report_is_deterministic():
    r1 = build_calibration_report(_bundle(), _portfolio([]), _hits())
    r2 = build_calibration_report(_bundle(), _portfolio([]), _hits())
    assert r1.model_dump() == r2.model_dump()


# --- assay-label join (reuses ml.assay_label) -----------------------------------------

def test_hits_from_assay_file_joins_by_mutation(tmp_path):
    # bundle mutations are A9G..A12G for v08..v11 (see _led); write an activity assay CSV
    csv = tmp_path / "assay.csv"
    csv.write_text(
        "mutation,label_type,value,source\n"
        "A9G,activity,2.0,wetlab\n"     # -> v08, high
        "A10G,activity,2.5,wetlab\n"    # -> v09, high
        "A1G,activity,0.1,wetlab\n"     # -> v00, low
        "A2G,activity,0.2,wetlab\n"     # -> v01, low
    )
    hits = hits_from_assay_file(csv, _bundle())
    assert hits["v08"] is True and hits["v09"] is True
    assert hits["v00"] is False and hits["v01"] is False
    # unmeasured variants are absent (not silently False-labelled)
    assert "v05" not in hits


def test_hits_activity_only_filter_drops_non_activity_readouts(tmp_path):
    # A9G is a STABILITY readout, A10G an ACTIVITY readout; the default activity_only filter
    # must keep only the activity variant when deciding hits (route via the CSV loader so the
    # test depends only on the calibration module + ledger, not on ml.assay_label directly).
    csv = tmp_path / "assay.csv"
    csv.write_text(
        "mutation,label_type,value,source\n"
        "A9G,stability,9.0,wetlab\n"
        "A10G,activity,9.0,wetlab\n"
    )
    hits = hits_from_assay_file(csv, _bundle())
    assert "v09" in hits            # A10G is activity -> kept
    assert "v08" not in hits        # A9G is stability -> filtered out
    # hits_by_variant_from_labels is the underlying join used by hits_from_assay_file
    assert callable(hits_by_variant_from_labels)
