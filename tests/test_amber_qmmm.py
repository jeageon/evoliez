"""V6-5 unit tests for the QM/MM-lite pure helpers (mask, sander &qmmm deck,
energy parse). The sander/sqm run is server-only."""
from evoliez.amber.qmmm import (
    QmmmLiteSpec, mdin_qmmm_min, parse_qmmm_energy, qmmask_from_residues,
)


def test_qmmask_from_residues():
    assert qmmask_from_residues(["LIG", "ATP", "MG"]) == ":LIG,ATP,MG"


def test_mdin_qmmm_deck():
    spec = QmmmLiteSpec(qm_residues=["LIG", "ATP", "MG"], qm_charge=-3,
                        qm_theory="PM6", min_steps=300)
    deck = mdin_qmmm_min(spec, qmmask_from_residues(spec.qm_residues))
    assert "ifqnt=1" in deck                    # QM/MM enabled
    assert "&qmmm" in deck
    assert "qmmask=':LIG,ATP,MG'" in deck
    assert "qmcharge=-3" in deck                # 3-HP −1 + ATP −4 + Mg +2
    assert "qm_theory='PM6'" in deck
    assert "restraintmask='@CA,C,N,O'" in deck  # backbone held, core relaxes


def test_parse_qmmm_energy():
    txt = ("   NSTEP       ENERGY          RMS \n"
           "      1      -1.2345E+05     5.0E+00\n"
           "...\n   FINAL RESULTS\n"
           "   NSTEP       ENERGY          RMS \n"
           "    300      -1.3010E+05     9.9E-02\n")
    e = parse_qmmm_energy(txt)
    assert e is not None and abs(e - (-1.3010e5)) < 1.0
