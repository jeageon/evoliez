"""V6-4: lock the CAR portfolio's claim-safety invariants on the committed
deliverables. A non-converged PMF forbids ranking, so the panel must stay a
claim-safe L0 mechanism-probe portfolio — every card L0, the doc ClaimGuard-clean.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CARDS = REPO / "outputs/car_v6/evidence_cards.json"
PLATE = REPO / "outputs/car_v6/wetlab_plate_24.csv"
DOC = REPO / "docs/car_v6/car_candidate_portfolio.md"

pytestmark = pytest.mark.skipif(not CARDS.exists(),
                                reason="run scripts/build_car_v6_portfolio.py first")


def test_panel_size_16_to_24():
    cards = json.loads(CARDS.read_text())
    assert 16 <= len(cards) <= 24


def test_every_card_is_L0_uncalibrated():
    cards = json.loads(CARDS.read_text())
    assert all(c["claim_level"] == "L0_uncalibrated" for c in cards)
    # experimental_calibration must be empty of real calibration -> stays L0
    assert all(c["experimental_calibration"]["evidence"] == ["no_wetlab_calibration"]
               for c in cards)


def test_no_ranking_score_leaks_into_plate():
    # the plate is a probe panel, not a ranking: no numeric final/activity score column
    header = PLATE.read_text().splitlines()[0].lower()
    for banned in ("final_score", "activity", "kcat", "rank", "recommendation"):
        assert banned not in header


def test_doc_and_plate_are_claimguard_clean():
    cg = pytest.importorskip("evoliez.ranking.claim_guard")
    from evoliez.ranking.claim_guard import ClaimProvenance
    cg.assert_report_clean(DOC.read_text(), ClaimProvenance())
    cg.assert_report_clean(PLATE.read_text(), ClaimProvenance())
