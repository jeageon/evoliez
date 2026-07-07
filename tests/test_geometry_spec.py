"""Unit tests for the generic functional-geometry layer."""
import numpy as np

from evoliez.md import geometry_spec as gs
from evoliez.md.geometry_spec import (
    GeometryTerm, GeometryTermResult, evaluate_geometry_term, spec_satisfaction,
)


def test_distance_occupancy(monkeypatch):
    monkeypatch.setattr(gs, "_resolve_global", lambda mb, s, i: {"A": 0, "B": 1}[s])
    frames = [np.array([[0, 0, 0], [2.0, 0, 0], [0, 0, 0]]),   # d=2.0
              np.array([[0, 0, 0], [5.0, 0, 0], [0, 0, 0]])]   # d=5.0
    t = GeometryTerm(kind="distance", a_smarts="A", b_smarts="B", distance_max=3.0)
    r = evaluate_geometry_term(frames, None, t)
    assert r.status == "ok" and r.n_frames == 2
    assert abs(r.occupancy - 0.5) < 1e-9          # 1 of 2 within 3.0 A
    assert abs(r.min - 2.0) < 1e-6


def test_angle_occupancy(monkeypatch):
    monkeypatch.setattr(gs, "_resolve_global",
                        lambda mb, s, i: {"A": 0, "B": 1, "C": 2}[s])
    frames = [np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]])]     # angle at b = 90 deg
    t = GeometryTerm(kind="angle", a_smarts="A", b_smarts="B", c_smarts="C",
                     angle_min=80, angle_max=100)
    r = evaluate_geometry_term(frames, None, t)
    assert r.status == "ok" and abs(r.mean - 90) < 1e-3 and r.occupancy == 1.0


def test_missing_atom_skips(monkeypatch):
    monkeypatch.setattr(gs, "_resolve_global", lambda mb, s, i: None)
    t = GeometryTerm(kind="distance", a_smarts="X", b_smarts="Y")
    r = evaluate_geometry_term([np.zeros((3, 3))], None, t)
    assert r.status == "skipped_missing_atoms"


def test_spec_satisfaction_is_min_and_all_required():
    terms = [GeometryTerm(kind="distance", label="d"),
             GeometryTerm(kind="angle", label="a")]
    ok = [GeometryTermResult("d", "distance", status="ok", occupancy=1.0),
          GeometryTermResult("a", "angle", status="ok", occupancy=0.5)]
    assert abs(spec_satisfaction(ok, terms) - 0.5) < 1e-9   # min of the terms
    # a single unresolved term -> 0.0 (a missing partner is not satisfaction)
    bad = [ok[0], GeometryTermResult("a", "angle", status="skipped_missing_atoms")]
    assert spec_satisfaction(bad, terms) == 0.0


def test_real_smarts_resolution():
    from rdkit import Chem
    mol = Chem.AddHs(Chem.MolFromSmiles("[O-]C=O"))   # formate
    mb = [(mol, list(range(mol.GetNumAtoms())))]
    c_idx = gs._resolve_global(mb, "[CX3]", 0)
    assert c_idx is not None
    assert mol.GetAtomWithIdx(c_idx).GetSymbol() == "C"
