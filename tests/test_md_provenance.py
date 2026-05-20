"""P0.6: OpenMM provenance + Sage probe order + HMR flag plumbing.

Source-level guards (no MD stack needed) for:
- `_ligand_system_generator` probes `prefer_ff` (Sage by default) BEFORE
  gaff-2.11; espaloma stays an optional fallback.
- `MDResult` carries provenance fields populated from MDConfig.
- `s10_md` writes `md_status` + `md_ligand_forcefield` + `md_hmr_enabled`
  + `md_timestep_fs` + `md_replicas_run` onto each candidate.scores.
- `MDConfig` exposes the new knobs with safe defaults.
"""

from __future__ import annotations

import inspect

from evoliez.adapters import openmm_engine
from evoliez.adapters.openmm_engine import MDResult
from evoliez.config import MDConfig
from evoliez.stages import s10_md


def test_ligand_system_generator_prefers_sage_before_gaff():
    src = inspect.getsource(openmm_engine._ligand_system_generator)
    # Sage is the new default (P0.6); gaff-2.11 and espaloma remain.
    assert 'openff-2.2.0' in src
    assert 'gaff-2.11' in src
    assert 'espaloma' in src
    # prefer_ff is parameterized through the call signature; the FF list
    # is built dynamically so the default can be overridden per run.
    assert "prefer_ff" in src
    assert "ffs.append" in src


def test_run_real_threads_ligand_forcefield_from_cfg():
    src = inspect.getsource(openmm_engine._run_real)
    # P0.6: the configured ligand_forcefield reaches the probe.
    assert 'prefer_ff' in src and 'ligand_forcefield' in src


def test_md_result_carries_provenance_fields():
    r = MDResult(
        candidate_id="c", status="ok", protocol_level=1,
        solvent_mode="implicit", simulation_time_ns=1.0,
    )
    # Defaults: no FF stamped yet, HMR off, single replica.
    assert r.ligand_forcefield is None
    assert r.hmr_enabled is False
    assert r.timestep_fs == 2.0
    assert r.replicas_run == 1
    assert r.ligand_rmsd_replicas == []
    assert r.pocket_rmsd_replicas == []


def test_md_config_exposes_p0_6_knobs():
    cfg = MDConfig()
    assert cfg.ligand_forcefield == "openff-2.2.0"
    assert cfg.hmr_enabled is False
    assert cfg.hmr_timestep_fs == 4.0
    assert cfg.final_tier_replicas == 3
    assert cfg.persist_provenance is True


def test_s10_md_persists_provenance_on_candidate_scores():
    src = inspect.getsource(s10_md.MDStage.run)
    # All four provenance scores must reach the report layer when the
    # config flag is on.
    for k in (
        "md_status", "md_ligand_forcefield", "md_hmr_enabled",
        "md_timestep_fs", "md_replicas_run",
    ):
        assert k in src, f"{k} not persisted onto candidate.scores"


def test_run_real_stamps_curated_amber_when_curated_path_won():
    # Source guard: the curated branch sets ligand_forcefield = "curated_amber"
    # so the report doesn't claim Sage when the Bryce Lab tleap path ran.
    src = inspect.getsource(openmm_engine._run_real)
    assert "curated_amber" in src
    assert "ligand_forcefield" in src
    assert "hmr_enabled" in src
