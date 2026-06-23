"""NAC-4 productive placement: building the formate explicitly off the acceptor ring face
must yield a genuinely near-attack geometry (close + linear) that passes the NAC placement
gate -- replacing Boltz's random formate pose that skipped 9/12 candidates."""
import numpy as np

from evoliez.md.cosubstrate_placement import (place_cosubstrate, productive_geometry,
                                              ring_normal)
from evoliez.md.nac import ReactiveSpec, nac_from_frames

SPEC = ReactiveSpec(donor_smarts="x", acceptor_smarts="y", distance_max=3.5, angle_min=150.0)


def test_placed_formate_is_near_attack():
    acc = np.zeros(3)
    normal = np.array([0.0, 0.0, 1.0])
    # arbitrary starting formate coords; atoms 0=C(donor) 1=H(transfer) 2,3=O
    formate0 = np.array([[5, 5, 5], [5, 5, 6], [6, 5, 5], [5, 6, 5]], float)
    new = place_cosubstrate(formate0, donor_heavy=0, transfer=1, acceptor_pos=acc, normal=normal)
    C, H = new[0], new[1]
    assert abs(np.linalg.norm(H - acc) - 2.8) < 1e-6          # H 2.8 Å off the face
    # feed the 3 reacting atoms [donor_heavy, transfer, acceptor] to the NAC geometry
    res = nac_from_frames([np.array([C, H, acc])], 0, 1, 2, SPEC)
    assert res.distance_min < 3.0 and res.angle_mean > 175    # close + ~linear
    assert res.occupancy == 1.0                               # reaction-competent
    assert res.distance_initial <= 4.0 and res.angle_initial >= 120  # passes placement gate


def test_carboxylate_geometry_is_sane():
    # the two O's should be ~1.25 Å from C and roughly 120° apart (planar sp2)
    p_H, p_C, p_O1, p_O2 = productive_geometry(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert abs(np.linalg.norm(p_O1 - p_C) - 1.25) < 1e-6
    assert abs(np.linalg.norm(p_O2 - p_C) - 1.25) < 1e-6
    v1, v2 = p_O1 - p_C, p_O2 - p_C
    ang = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    assert 110 < ang < 130


def test_face_selection_follows_hint():
    acc = np.zeros(3)
    normal = np.array([0.0, 0.0, 1.0])
    f0 = np.zeros((4, 3))
    up = place_cosubstrate(f0, 0, 1, acc, normal, face_toward=np.array([0, 0, 10.0]))
    dn = place_cosubstrate(f0, 0, 1, acc, normal, face_toward=np.array([0, 0, -10.0]))
    assert up[1][2] > 0 and dn[1][2] < 0                      # H follows the catalytic face


def test_ring_normal_of_planar_hexagon():
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang), np.zeros(6)], axis=1)   # unit hexagon in xy
    n = ring_normal(ring, list(range(6)))
    assert abs(abs(n[2]) - 1.0) < 1e-6                        # normal is ±z
