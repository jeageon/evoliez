"""V6-2 unit tests for the Amber GPU scheduler's pure planning/analysis helpers.

The pmemd.cuda fan-out itself is server-only (exercised by scripts/run_amber_md_car.sh);
here we lock restart detection, reactive-mask extraction from a V6-1 manifest, the
cpptraj deck, and the occupancy summarization (access distance vs in-line angle
kept separate).
"""
from pathlib import Path

from evoliez.adapters.amber_scheduler import (
    production_nsteps, reactive_cpptraj_script, reactive_masks_from_manifest,
    stage_plan, stages_remaining, summarize_reactive, _col, _energy_drift,
    _mdin_min1_exp, _mdin_min2_exp,
)

MANIFEST = {
    "fingerprint": "abc",
    "reactive_atoms": [
        {"role": "O_nuc", "mask": ":LIG@O2", "resid": 722},
        {"role": "P_alpha", "mask": ":ATP@P1", "resid": 723},
        {"role": "O_leaving", "mask": ":ATP@O5", "resid": 723},
        {"role": "metal", "mask": ":MG@MG", "resid": 721},
        {"role": "catalytic", "mask": ":268", "resid": 268, "orig_resid": 268},
    ],
    "reactive_spec": {"distance_max": 3.6},
}


def test_production_nsteps():
    assert production_nsteps(0.2, 2.0) == 100_000
    assert production_nsteps(0.04, 2.0) == 20_000


def test_stage_plan():
    assert stage_plan("explicit") == ["min", "heat", "equil", "prod"]
    assert stage_plan("implicit") == ["min", "heat", "prod"]


def test_stages_remaining_restart_safe(tmp_path):
    # nothing done -> all stages
    assert stages_remaining(tmp_path, "explicit", 1) == ["min", "heat", "equil", "prod"]
    # min + heat restarts present -> resume at equil
    (tmp_path / "min.rst").write_text("x")
    (tmp_path / "heat.rst").write_text("x")
    assert stages_remaining(tmp_path, "explicit", 1) == ["equil", "prod"]
    # equil done + prod nc present -> nothing remaining
    (tmp_path / "equil.rst").write_text("x")
    (tmp_path / "prod_0.nc").write_text("traj")
    assert stages_remaining(tmp_path, "explicit", 1) == []
    # 2 replicas: only prod_0 present -> prod still remaining
    (tmp_path / "prod_1.nc").unlink(missing_ok=True)
    assert stages_remaining(tmp_path, "explicit", 2) == ["prod"]


def test_reactive_masks_from_manifest():
    masks = reactive_masks_from_manifest(MANIFEST)
    assert masks["O_nuc"] == ":LIG@O2"
    assert masks["P_alpha"] == ":ATP@P1"
    assert masks["O_leaving"] == ":ATP@O5"
    assert masks["metal"] == ":MG@MG"
    assert masks["catalytic_268"] == ":268"


def test_reactive_cpptraj_script():
    masks = reactive_masks_from_manifest(MANIFEST)
    deck = reactive_cpptraj_script(masks, "explicit", ["prod_0.nc"])
    assert "autoimage" in deck and "strip :WAT,Na+,Cl-,K+" in deck
    assert "distance d_onuc_pa :LIG@O2 :ATP@P1" in deck
    assert "angle a_inline :LIG@O2 :ATP@P1 :ATP@O5" in deck
    assert "distance d_mg_onuc :MG@MG :LIG@O2" in deck
    assert "distance d_mg_pa :MG@MG :ATP@P1" in deck


def test_col_parse():
    dat = "#Frame d_onuc_pa a_inline\n1 3.2 160.0\n2 4.1 140.0\n"
    assert _col(dat, "d_onuc_pa") == [3.2, 4.1]
    assert _col(dat, "a_inline") == [160.0, 140.0]
    assert _col(dat, "missing") == []


def test_summarize_reactive_separates_access_and_angle():
    # 4 frames: distances 3.0,3.5,4.0,5.0 ; angles 160,155,140,120
    reactive = ("#Frame d_onuc_pa a_inline d_mg_onuc d_mg_pa\n"
                "1 3.0 160 2.1 2.2\n2 3.5 155 2.3 2.0\n"
                "3 4.0 140 4.5 2.1\n4 5.0 120 5.0 5.0\n")
    masks = {"O_nuc": ":LIG@O2", "P_alpha": ":ATP@P1", "O_leaving": ":ATP@O5",
             "metal": ":MG@MG"}
    s = summarize_reactive(reactive, [1.0, 1.2, 1.5, 2.0], [0.8, 0.9, 1.0, 1.1],
                           masks, near_attack_A=3.6, angle_min=150.0, mg_coord_A=2.8)
    # access distance: 2/4 frames <= 3.6
    assert s["access_distance"]["near_attack_occupancy"] == 0.5
    assert s["access_distance"]["series_min_A"] == 3.0
    # in-line angle: 2/4 frames >= 150 (kept SEPARATE from access)
    assert s["inline_angle"]["productive_angle_occupancy"] == 0.5
    # Mg bridge retained (both <=2.8) only in frames 1,2 -> 0.5
    assert s["mg_retention"]["bridge_occupancy"] == 0.5
    assert s["ligand_retention"] == 1.0     # all lig rmsd <= 5.0


def test_energy_drift_ignores_pmemd_averages_block(tmp_path):
    # real frames drift ~1.5%; pmemd's trailing AVERAGES/RMS-FLUCT lines (Etot=612)
    # must NOT be treated as the last frame (that read ~100% and flagged unstable).
    (tmp_path / "prod_0.out").write_text(
        " NSTEP = 1\n Etot   =   -176895.7181  EKtot =  44466.8\n"
        " NSTEP = 100\n Etot   =   -174203.0624  EKtot =  44944.1\n"
        "      A V E R A G E S   O V E R     100 S T E P S\n"
        " Etot   =   -175000.0000  EKtot =  44500.0\n"
        "      R M S  F L U C T U A T I O N S\n"
        " Etot   =       612.0861  EKtot =    271.3\n")
    drift = _energy_drift(tmp_path)
    assert drift < 0.05, f"drift {drift} should be ~1.5%, not poisoned by the summary"


def test_robust_min_mdin_decks():
    m1 = _mdin_min1_exp()
    assert "imin=1" in m1 and "ntr=1" in m1 and "ntmin=1" in m1
    assert "restraintmask='@CA,C,N,O'" in m1 and "restraint_wt=10.0" in m1
    assert "ntc=1, ntf=1" in m1        # no SHAKE during clash relaxation
    m2 = _mdin_min2_exp()
    assert "restraint_wt=2.0" in m2    # weaker restraint in stage 2
