"""Unit tests for the NAC / catalytic-power core (evoliez.md.nac).

The geometry/occupancy core is pure NumPy (always tested); the reactive-atom
identification needs RDKit (skipped where unavailable, e.g. the light venv).
"""
import numpy as np
import pytest

from evoliez.md.nac import (FDH_HYDRIDE, ReactiveSpec, identify_acceptor,
                            identify_donor, nac_from_frames)

_SPEC = ReactiveSpec(donor_smarts="", acceptor_smarts="",
                     distance_max=3.5, angle_min=150.0)


def test_nac_distance_and_angle_both_required():
    # atoms 0=donor_heavy, 1=transfer(H), 2=acceptor
    reactive = np.array([[0, 0, 0], [1.1, 0, 0], [3.9, 0, 0]], float)   # d2.8 ang180
    far = np.array([[0, 0, 0], [1.1, 0, 0], [11.1, 0, 0]], float)       # d10.0
    bent = np.array([[0, 0, 0], [1.1, 0, 0], [1.1, 2.8, 0]], float)     # d2.8 ang90
    res = nac_from_frames([reactive, far, bent], 0, 1, 2, _SPEC)
    assert res.n_frames == 3
    assert res.n_reactive == 1                  # only the close+linear frame
    assert abs(res.occupancy - 1 / 3) < 1e-3
    assert res.distance_min == 2.8
    assert res.angle_mean == 150.0              # (180+180+90)/3


def test_nac_all_and_none_reactive():
    near = np.array([[0, 0, 0], [1.1, 0, 0], [3.6, 0, 0]], float)       # d2.5 ang180
    assert nac_from_frames([near, near], 0, 1, 2, _SPEC).occupancy == 1.0
    far = np.array([[0, 0, 0], [1.1, 0, 0], [11, 0, 0]], float)
    assert nac_from_frames([far], 0, 1, 2, _SPEC).occupancy == 0.0


def test_nac_empty_frames():
    res = nac_from_frames([], 0, 1, 2, _SPEC)
    assert res.occupancy == 0.0 and res.n_frames == 0


_NADP = ("NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP([O-])(=O)OP([O-])(=O)OC[C@H]2O"
         "[C@@H](n3cnc4c3ncnc4N)[C@H](O)[C@@H]2OP([O-])([O-])=O)[C@@H](O)[C@H]1O")


def test_fdh_reactive_atom_identification():
    Chem = pytest.importorskip("rdkit.Chem")
    nadp = Chem.AddHs(Chem.MolFromSmiles(_NADP))
    acc = identify_acceptor(nadp, FDH_HYDRIDE)
    assert acc is not None
    a = nadp.GetAtomWithIdx(acc)
    assert a.GetSymbol() == "C" and a.GetIsAromatic()
    # the nicotinamide C4: not bonded to the ring n+
    npos = {x.GetIdx() for x in nadp.GetAtoms()
            if x.GetSymbol() == "N" and x.GetFormalCharge() == 1}
    assert not any(n.GetIdx() in npos for n in a.GetNeighbors())

    fmt = Chem.AddHs(Chem.MolFromSmiles("[O-]C=O"))
    don = identify_donor(fmt, FDH_HYDRIDE)
    assert don is not None and "transfer" in don
    assert fmt.GetAtomWithIdx(don["transfer"]).GetSymbol() == "H"
