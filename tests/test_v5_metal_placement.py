"""ROADMAP_V5 V5-2 — Mg2+ bridging placement geometry (the pure, testable core; the OpenMM
System build consumes the resulting PDB on the server)."""
import numpy as np

from evoliez.md.metal_placement import (
    MG_COORD_DIST, bridging_metal_position, insert_mg_into_pdb, mg_hetatm_line)


def test_bridges_both_anions_at_coordination_distance():
    # two O atoms 3.0 Å apart (closer than 2*2.1) -> Mg equidistant at ~2.1 Å from each
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 0.0, 0.0])
    m = bridging_metal_position(a, b)
    da = float(np.linalg.norm(m - a))
    db = float(np.linalg.norm(m - b))
    assert abs(da - db) < 1e-6                             # symmetric bridge
    assert abs(da - MG_COORD_DIST) < 0.05                 # ~coordination distance from each


def test_deterministic_for_wt_and_mutant_same_inputs():
    a, b, ref = [0, 0, 0], [3, 0, 0], [0, 5, 0]
    m1 = bridging_metal_position(a, b, reference=ref)
    m2 = bridging_metal_position(a, b, reference=ref)
    assert np.allclose(m1, m2)                              # identical placement -> valid ΔNAC


def test_reference_picks_the_side_away_from_pocket():
    a, b = [0, 0, 0], [3, 0, 0]
    # reference on +y -> ion placed on -y (away from the reference/pocket bulk)
    m = bridging_metal_position(a, b, reference=[0, 10, 0])
    assert m[1] < 0


def test_far_apart_falls_back_to_midpoint():
    # O–O 10 Å apart: can't bridge at 2.1 Å -> midpoint best effort (flagged by caller)
    m = bridging_metal_position([0, 0, 0], [10, 0, 0])
    assert np.allclose(m, [5, 0, 0])


def test_coincident_atoms_do_not_crash():
    m = bridging_metal_position([1, 1, 1], [1, 1, 1])
    assert np.allclose(m, [1, 1, 1])


def test_mg_hetatm_and_insertion():
    line = mg_hetatm_line([1.234, 5.678, 9.012])
    assert line.startswith("HETATM") and "MG" in line and "MG2+" in line.replace(" ", "")[-6:] or "MG" in line
    pdb = "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n"
    out = insert_mg_into_pdb(pdb, [1.0, 2.0, 3.0])
    lines = out.splitlines()
    # MG inserted BEFORE END
    assert any(l.startswith("HETATM") and "MG" in l for l in lines)
    assert lines.index([l for l in lines if l.startswith("HETATM")][0]) < lines.index("END")
