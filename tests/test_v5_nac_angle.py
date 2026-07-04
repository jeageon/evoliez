"""ROADMAP_V5 V5-1 — O->P (adenylation) near-attack angle.

For a non-hydride O->P attack the transferring atom IS the donor heavy atom, so the legacy
3-point donor--transfer--acceptor angle is degenerate (NaN) and forces occupancy 0 for EVERY
frame regardless of geometry (the CAR dynamic-NAC failure). With a resolved leaving atom the
angle becomes the real in-line O_nuc--Palpha--O_leaving.
"""
import numpy as np
import pytest

from evoliez.md.nac import ReactiveSpec, nac_from_frames, nac_from_subframes


def _ospec():
    # O attacks P (no H transfer); near-attack O...P <= 3.6 Å and in-line angle >= 150 deg.
    return ReactiveSpec(
        donor_smarts="[OX1-]", acceptor_smarts="[PX4]([OX2][CX4])",
        transfer_is_h=False, distance_max=3.6, angle_min=150.0,
        label="3HP_to_ATP_alphaP")


def _inline_frame():
    # rows = [donor_heavy(O_nuc), transfer(==O_nuc), acceptor(Palpha), leaving(O_leaving)]
    o_nuc = [-3.2, 0.0, 0.0]         # 3.2 Å from Palpha
    p_alpha = [0.0, 0.0, 0.0]
    o_leaving = [1.6, 0.0, 0.0]      # opposite side -> O_nuc--Palpha--O_leaving == 180 deg
    return np.array([o_nuc, o_nuc, p_alpha, o_leaving], float)


def test_old_3atom_path_is_degenerate_nan():
    # WITHOUT a leaving atom the O->P angle is the degenerate donor==transfer vertex -> NaN
    fr = _inline_frame()
    res = nac_from_frames([fr], donor_heavy=0, transfer=1, acceptor=2, spec=_ospec())
    assert np.isnan(res.angle_mean)      # regression lock: this is the bug
    assert res.occupancy == 0.0          # NaN >= 150 is always False


def test_leaving_atom_gives_real_inline_angle_and_occupancy():
    fr = _inline_frame()
    res = nac_from_frames([fr], donor_heavy=0, transfer=1, acceptor=2, spec=_ospec(), leaving=3)
    assert not np.isnan(res.angle_mean)
    assert res.angle_mean == pytest.approx(180.0, abs=1.0)
    assert res.distance_min == pytest.approx(3.2, abs=0.05)   # O_nuc -> Palpha
    assert res.occupancy == 1.0                                # in-line + within 3.6 Å


def test_bent_geometry_is_not_reaction_competent():
    # O_leaving perpendicular -> ~90 deg angle -> below angle_min 150 -> not reactive
    fr = np.array([[-3.2, 0, 0], [-3.2, 0, 0], [0, 0, 0], [0, 1.6, 0]], float)
    res = nac_from_frames([fr], 0, 1, 2, _ospec(), leaving=3)
    assert res.angle_mean == pytest.approx(90.0, abs=1.0)
    assert res.occupancy == 0.0


def test_nac_from_subframes_autodetects_leaving_from_width():
    # a 4-row subframe must be treated as the O->P case (leaving at row 3)
    res4 = nac_from_subframes([_inline_frame()], _ospec())
    assert not np.isnan(res4.angle_mean) and res4.occupancy == 1.0
    # a 3-row subframe (hydride-style) keeps the legacy 3-point angle
    res3 = nac_from_subframes([_inline_frame()[:3]], _ospec())
    assert np.isnan(res3.angle_mean)


def test_identify_leaving_topology():
    Chem = pytest.importorskip("rdkit.Chem")
    from evoliez.md.nac import identify_leaving
    # methyl triphosphate-like: C-O-P(=O)(O)-O-P... ; the alpha-P's bridging O to the beta-P
    mol = Chem.MolFromSmiles("COP(=O)(O)OP(=O)(O)O")
    assert mol is not None
    # find the alpha-P (the P bonded to the ester O off the carbon)
    alpha_p = None
    for a in mol.GetAtoms():
        if a.GetSymbol() == "P":
            for nb in a.GetNeighbors():
                if nb.GetSymbol() == "O" and any(n.GetSymbol() == "C" for n in nb.GetNeighbors()):
                    alpha_p = a.GetIdx()
            if alpha_p == a.GetIdx():
                break
    assert alpha_p is not None
    lv = identify_leaving(mol, alpha_p)
    assert lv is not None
    # the leaving O must bridge the alpha-P to a SECOND P
    o = mol.GetAtomWithIdx(lv)
    assert o.GetSymbol() == "O"
    assert any(n.GetSymbol() == "P" and n.GetIdx() != alpha_p for n in o.GetNeighbors())
