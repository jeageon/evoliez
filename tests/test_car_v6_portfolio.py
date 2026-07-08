"""V6-4: lock the CAR portfolio's claim-safety invariants on the committed
deliverables. A non-converged PMF forbids ranking, so the panel must stay a
claim-safe L0 mechanism-probe portfolio — every card L0, the doc ClaimGuard-clean.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CARDS = REPO / "outputs/car_v6/evidence_cards.json"
PLATE = REPO / "outputs/car_v6/mechanism_probe_plate.csv"
DOC = REPO / "docs/car_v6/car_candidate_portfolio.md"


def _plate_header(text: str) -> str:
    # skip leading '#' caveat comment line(s) to reach the real column header
    for line in text.splitlines():
        if not line.startswith("#"):
            return line.lower()
    return ""

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
    header = _plate_header(PLATE.read_text())
    for banned in ("final_score", "activity", "kcat", "rank", "recommendation"):
        assert banned not in header


def test_plate_carries_claim_boundary_caveat():
    # the artifact must self-document its claim boundary (reviewer requirement)
    first = PLATE.read_text().splitlines()[0]
    assert first.startswith("#") and "hypothesis-grade" in first.lower()
    assert "not lead validation" in first.lower() or "not a validated" in first.lower()


def test_doc_and_plate_are_claimguard_clean():
    cg = pytest.importorskip("evoliez.ranking.claim_guard")
    from evoliez.ranking.claim_guard import ClaimProvenance
    cg.assert_report_clean(DOC.read_text(), ClaimProvenance())
    cg.assert_report_clean(PLATE.read_text(), ClaimProvenance())
