from evoliez.ranking.evidence_card_v4 import EvidenceCardV4, EvidenceAxisV4
from evoliez.reports.v4_candidate_card import render_candidate_cards


def test_high_score_low_confidence_rendering():
    card = EvidenceCardV4(
        candidate_id="x",
        mutation="A1B",
        role="probe",
        reaction_geometry_accommodation=EvidenceAxisV4(score=0.9, confidence="low"),
        pose_uncertainty=EvidenceAxisV4(score=0.8, confidence="low"),
        experimental_calibration=EvidenceAxisV4(
            score=0.0,
            confidence="low",
            evidence=["no_wetlab_calibration"],
            reason="claims remain L0 until assay data are ingested",
        ),
    )
    report = render_candidate_cards([card])
    assert "0.900 (low)" in report
