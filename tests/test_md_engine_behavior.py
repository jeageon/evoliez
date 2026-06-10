"""Behavioral guards for the headline real-MD fixes (ultra-review G3,
reviewer #23/#24/#25). These replace source-string introspection with tests
that FAIL if the exact bugs are reintroduced. Pure helpers - no OpenMM/GPU."""

from evoliez.adapters.openmm_engine import (
    _LITE_MAX_STEPS,
    MDResult,
    _ca_restraint_tier,
    _production_nsteps,
)
from evoliez.config import MDConfig, ScoreWeights
from evoliez.md.analysis import analyse


# --- #23: production step count (was 1000x short; actual_ns echoed back) --- #
def test_production_nsteps_matches_simulated_time():
    cfg = MDConfig(protocol_level=3, production_ns=1.0, timestep_fs=2.0)
    nsteps, actual_ns = _production_nsteps(cfg)
    assert nsteps == 500_000              # 1 ns / 2 fs = 5e5 steps, NOT ~500
    assert abs(actual_ns - 1.0) < 1e-9


def test_lite_rungs_capped_and_actual_ns_reflects_steps():
    cfg = MDConfig(protocol_level=1, production_ns=1.0, timestep_fs=2.0)
    nsteps, actual_ns = _production_nsteps(cfg)
    assert nsteps == _LITE_MAX_STEPS                       # L1/L2 capped
    # actual_ns is derived FROM nsteps, never echoed as production_ns (=1.0)
    assert abs(actual_ns - _LITE_MAX_STEPS * 2.0 / 1e6) < 1e-9
    assert actual_ns < 1.0


def test_nsteps_has_a_floor():
    cfg = MDConfig(protocol_level=3, production_ns=0.0, timestep_fs=2.0)
    nsteps, _ = _production_nsteps(cfg)
    assert nsteps == 50


# --- #25: pocket backbone is FREE (the unfreeze-pocket fix) --- #
def test_restraint_tiers_leave_pocket_backbone_free():
    # POCKET_NM=0.8, SHELL_NM=1.2 (nm from the ligand centroid)
    assert _ca_restraint_tier(0.5) == "free"     # inside pocket -> NOT restrained
    assert _ca_restraint_tier(1.0) == "weak"     # active-site shell
    assert _ca_restraint_tier(2.0) == "strong"   # distant backbone held firmly
    assert _ca_restraint_tier(None) == "strong"  # apo (no ligand): hold backbone


def test_pocket_boundary_is_inclusive_free():
    assert _ca_restraint_tier(0.8) == "free"     # d <= POCKET_NM is free


# --- #24: md_lite RAW score must be weight-invariant (no squared MD term) --- #
def _passing_result() -> MDResult:
    return MDResult(
        candidate_id="c", status="ok", protocol_level=3,
        solvent_mode="implicit", simulation_time_ns=1.0,
        ligand_rmsd_series=[0.5, 0.6, 0.7],
        pocket_rmsd_series=[0.3, 0.4, 0.5],
        hbond_occupancy=0.6, contact_occupancy={"a": 0.8}, energy_drift=0.01,
    )


def test_md_lite_score_is_weight_invariant():
    r = _passing_result()
    s1 = analyse(r, ScoreWeights(md_lite=1.0)).md_lite_score
    s3 = analyse(r, ScoreWeights(md_lite=3.0)).md_lite_score
    assert s1 == s3        # md_lite weight is applied ONCE downstream, not folded in
    assert s1 != 0.0       # the passing-run scoring block actually executed
