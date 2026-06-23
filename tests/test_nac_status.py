"""NAC validity accounting (NAC-1/2): the catalytic NAC must distinguish a genuinely
non-reactive-but-retained co-substrate from one that simply DIFFUSED away or was
mis-placed. Only a retained, reactive-geometry NAC may feed ranking; a diffused/
mis-placed formate yields nac_occupancy = None (never a fake "low reactivity")."""
import math

import numpy as np
import pytest

from evoliez.md.nac import (
    NAC_INVALID_DIFFUSED, NAC_SKIP_PLACEMENT, NAC_VALID, ReactiveSpec,
    nac_from_frames, nac_status_is_valid)


def _frames(distances, angles):
    """3-atom (donor_heavy, transfer, acceptor) frames at the given transfer->acceptor
    distance (Å) and donor_heavy-transfer-acceptor angle (deg)."""
    out = []
    for d, th in zip(distances, angles):
        acc = np.array([0.0, 0.0, 0.0])
        tr = np.array([d, 0.0, 0.0])
        dh = tr + 1.1 * np.array([-math.cos(math.radians(th)), math.sin(math.radians(th)), 0.0])
        out.append(np.array([dh, tr, acc]))
    return out


SPEC = ReactiveSpec(donor_smarts="x", acceptor_smarts="y",
                    distance_max=3.5, angle_min=150.0)


def test_retained_but_unaligned_is_valid_with_zero_occupancy():
    # formate stays in the pocket (~3.4 Å) but never reaches the reactive angle
    r = nac_from_frames(_frames([3.4] * 50, [74] * 50), 0, 1, 2, SPEC)
    assert r.status == NAC_VALID
    assert r.retention_fraction == 1.0 and not r.escape
    assert r.occupancy == 0.0
    assert r.occupancy_or_none == 0.0          # valid -> usable


def test_diffused_cosubstrate_is_invalid_and_occupancy_none():
    # starts close (3.4 Å) then diffuses to >10 Å -> occupancy is not interpretable
    r = nac_from_frames(_frames(list(np.linspace(3.4, 12.0, 50)), [60] * 50), 0, 1, 2, SPEC)
    assert r.status == NAC_INVALID_DIFFUSED
    assert r.escape and r.retention_fraction < 0.8
    assert r.occupancy_or_none is None         # excluded from ranking


def test_bad_initial_placement_is_skipped():
    # formate placed ~9 Å away the whole time (Boltz mis-placement)
    r = nac_from_frames(_frames([9.0] * 50, [122] * 50), 0, 1, 2, SPEC)
    assert r.status == NAC_SKIP_PLACEMENT
    assert r.occupancy_or_none is None


def test_genuinely_reactive_is_valid_nonzero():
    # retained AND linear, close -> real productive occupancy
    r = nac_from_frames(_frames([2.9] * 50, [160] * 50), 0, 1, 2, SPEC)
    assert r.status == NAC_VALID
    assert r.occupancy_or_none == 1.0


def test_status_validity_helper():
    assert nac_status_is_valid(NAC_VALID)
    assert not nac_status_is_valid(NAC_INVALID_DIFFUSED)
    assert not nac_status_is_valid(NAC_SKIP_PLACEMENT)
    assert not nac_status_is_valid(None)


def test_to_json_exposes_status_and_gates():
    r = nac_from_frames(_frames([3.4] * 50, [74] * 50), 0, 1, 2, SPEC)
    j = r.to_json()
    for k in ("nac_status", "nac_occupancy", "retention_fraction", "escape",
              "distance_initial", "angle_initial", "occupancy_retained"):
        assert k in j
    assert j["nac_occupancy"] == 0.0           # valid -> the gated value
