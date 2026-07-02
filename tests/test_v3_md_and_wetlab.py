"""V3-9: MD adaptive early-stop + wet-lab recalibration (ROADMAP_V3 D10)."""
import numpy as np

from evoliez.md.early_stop import (
    EarlyStopConfig, early_stop_record, evaluate_early_stop,
)
from evoliez.ranking.wetlab_feedback import WetLabResult, recalibrate_from_wetlab


# --- early stop ----------------------------------------------------------------------
def _frames_with_ligand_at(distances):
    """Build frames: atom 0 = anchor at origin; atom 1 = single-atom ligand at (d,0,0)."""
    return [np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]) for d in distances]


def test_early_stop_disabled_by_default():
    frames = _frames_with_ligand_at([20, 21, 22, 23, 24])
    v = evaluate_early_stop(frames, ligand_atom_idxs=[1], anchor_atom_idx=0)
    assert v.stop is False and v.reason == "not_evaluated"


def test_early_stop_on_ligand_escape():
    frames = _frames_with_ligand_at([4, 7, 10, 13, 16])   # monotonic escape
    v = evaluate_early_stop(frames, ligand_atom_idxs=[1], anchor_atom_idx=0,
                            cfg=EarlyStopConfig(enabled=True, escape_distance_A=12.0))
    assert v.stop is True and v.reason == "ligand_escaped_pocket"
    rec = early_stop_record(v)
    assert rec["md_lite_status"] == "early_stopped"
    assert rec["structural_viability_override"] == 0.0
    assert rec["uncertainty_override"] == 1.0


def test_no_stop_when_ligand_retained():
    frames = _frames_with_ligand_at([4, 3.5, 4, 3.8, 4])  # stays in pocket
    v = evaluate_early_stop(frames, ligand_atom_idxs=[1], anchor_atom_idx=0,
                            cfg=EarlyStopConfig(enabled=True, escape_distance_A=12.0))
    assert v.stop is False


def test_early_stop_on_network_collapse():
    # ligand retained, but catalytic contacts all broken
    frames = [np.array([[0, 0, 0], [4, 0, 0], [0, 0, 0], [20, 0, 0]], float) for _ in range(6)]
    contacts = [(2, 3, 3.5)]   # atom2-atom3 should be < 3.5 but is 20 Å
    v = evaluate_early_stop(frames, ligand_atom_idxs=[1], anchor_atom_idx=0,
                            catalytic_contacts=contacts,
                            cfg=EarlyStopConfig(enabled=True, min_contact_fraction=0.2))
    assert v.stop is True and v.reason == "catalytic_network_collapsed"
    assert v.contact_fraction == 0.0


# --- wet-lab recalibration -----------------------------------------------------------
def test_recalibration_unlocks_only_on_replicated_active():
    # actives exist but not replicated -> claims stay locked
    res = [WetLabResult("A1", active=True, replicated=False),
           WetLabResult("A2", active=False, replicated=True)]
    rc = recalibrate_from_wetlab(res, {"A1": 3.0})
    assert rc.wetlab_replicated is False
    assert rc.n_active == 1
    assert any("stay locked" in n for n in rc.notes)

    res2 = [WetLabResult("A1", active=True, replicated=True),
            WetLabResult("A3", active=True, replicated=True)]
    rc2 = recalibrate_from_wetlab(res2, {"A1": 3.0, "A3": 3.2})
    assert rc2.wetlab_replicated is True
    assert rc2.geometry_center == 3.1                    # median of actives' geometry
    assert rc2.claim_provenance_update()["wetlab_replicated"] is True


def test_recalibration_handles_no_geometry():
    rc = recalibrate_from_wetlab([WetLabResult("A1", active=True, replicated=True)])
    assert rc.geometry_center is None
    assert any("kernel unchanged" in n for n in rc.notes)
