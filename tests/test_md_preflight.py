"""P0 C — MD preflight at s01 (before s08b).

Verifies that `run_md_preflight`:
  - short-circuits on curated cofactors (no probe needed; MD will take
    the tleap path);
  - short-circuits when MD is disabled in config;
  - degrades gracefully when openmmforcefields / OpenFF aren't
    importable (the light mac venv);
  - writes results to the same disk sidecar that s10's per-candidate
    `_ligand_system_generator` consults, so every candidate becomes a
    cache hit.

The tests mock out openmmforcefields / OpenFF where needed so the suite
runs in the light venv. End-to-end "real probe primes real s10 cache"
correctness is covered indirectly by the existing
test_md_ligand_probe_cache.py round-trip tests.
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from evoliez.adapters import md_preflight as mdp
from evoliez.adapters.openmm_engine import (
    _LIGAND_PROBE_CACHE_MEM,
    _clear_ligand_probe_cache,
    _ligand_probe_cache_path,
    _probe_cache_key,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    _clear_ligand_probe_cache()
    yield
    _clear_ligand_probe_cache()


def test_md_disabled_short_circuits(tmp_path):
    """When MD is off in config, the preflight should be free."""
    result = mdp.run_md_preflight(
        smiles="CCO", md_dir=tmp_path / "md", md_enabled=False,
    )
    assert result.status == "skipped_md_disabled"
    # No sidecar written, no md_dir spelunking.
    assert not (tmp_path / "md" / ".ligand_probe_cache.json").exists()


def test_no_smiles_short_circuits(tmp_path):
    """Defensive: empty SMILES → skip cleanly, don't crash."""
    result = mdp.run_md_preflight(
        smiles="", md_dir=tmp_path / "md", md_enabled=True,
    )
    assert result.status == "skipped_no_smiles"


def test_curated_cofactor_short_circuits(tmp_path, monkeypatch):
    """When `lookup_by_smiles` finds a curated spec AND
    `resolved_amber_files()` returns a path tuple (files exist on
    disk), the preflight reports curated_available without ever
    touching OpenFF — MD takes the tleap path."""
    fake_spec = MagicMock()
    fake_spec.name = "NADP+"
    fake_spec.amber_residue_name = "NAP"
    fake_spec.resolved_amber_files.return_value = (
        tmp_path / "NAP.lib",
        tmp_path / "NAP.frcmod",
    )
    with patch.object(mdp, "lookup_by_smiles", return_value=fake_spec):
        # Even with OpenFF imports failing, the curated path doesn't care.
        with patch.object(mdp, "_build_offmol",
                          side_effect=AssertionError(
                              "shouldn't reach OpenFF on curated path")):
            result = mdp.run_md_preflight(
                smiles="<NADP+ smiles>", md_dir=tmp_path / "md",
            )
    assert result.status == "curated_available"
    assert result.ff_used == "amber:NAP"


def test_no_md_libs_degrades_to_skip(tmp_path, monkeypatch):
    """Light mac venv: OpenFF not importable → defer to s10."""
    # `_build_offmol` returns None on ImportError.
    with patch.object(mdp, "lookup_by_smiles", return_value=None):
        with patch.object(mdp, "_build_offmol", return_value=None):
            result = mdp.run_md_preflight(
                smiles="CCO", md_dir=tmp_path / "md",
            )
    assert result.status == "skipped_no_md_libs"
    assert "openff" in (result.reason or "").lower() or result.reason is None


def test_ok_probe_writes_disk_sidecar(tmp_path):
    """Successful probe should:
      - return status="ok"
      - write the `(smiles, ff)` entry to the disk sidecar at
        `md_dir/.ligand_probe_cache.json` so s10's per-candidate
        `_ligand_system_generator` sees the cached OK and skips its
        own probe.
    """
    md_dir = tmp_path / "md"

    # Stub OpenFF Molecule with a stable canonical SMILES.
    fake_mol = MagicMock()
    fake_mol.to_smiles.return_value = "CCO"
    fake_mol.to_topology.return_value.to_openmm.return_value = "topology"

    # Inject a fake openmmforcefields.generators.SystemGenerator that
    # always succeeds. We do this via sys.modules so the inline import
    # inside `run_md_preflight` picks it up.
    fake_sg_module = types.ModuleType("openmmforcefields.generators")

    class _FakeSG:
        def __init__(self, **kw):
            pass

        def create_system(self, *a, **kw):
            return "system"

    fake_sg_module.SystemGenerator = _FakeSG
    fake_omff = types.ModuleType("openmmforcefields")
    fake_omff.generators = fake_sg_module

    # Inject openmm.app stub if not present.
    fake_app = types.ModuleType("openmm.app")
    fake_app.HBonds = "HBonds"
    fake_app.CutoffNonPeriodic = "CutoffNonPeriodic"
    fake_openmm = types.ModuleType("openmm")
    fake_openmm.app = fake_app

    sys_mods_patch = {
        "openmmforcefields": fake_omff,
        "openmmforcefields.generators": fake_sg_module,
    }
    # Only add openmm stubs if the real ones aren't there (we don't want
    # to mask a real install on the server side).
    if "openmm" not in sys.modules:
        sys_mods_patch["openmm"] = fake_openmm
        sys_mods_patch["openmm.app"] = fake_app

    with patch.object(mdp, "lookup_by_smiles", return_value=None):
        with patch.object(mdp, "_build_offmol", return_value=fake_mol):
            with patch.dict(sys.modules, sys_mods_patch):
                result = mdp.run_md_preflight(
                    smiles="CCO", md_dir=md_dir,
                    prefer_ff="openff-2.2.0",
                )

    assert result.status == "ok"
    assert result.ff_used == "openff-2.2.0"

    # Sidecar must contain the (smiles, ff) entry tagged ok.
    sidecar = md_dir / ".ligand_probe_cache.json"
    assert sidecar.exists(), "preflight didn't prime the disk sidecar"
    data = json.loads(sidecar.read_text())
    assert data[_probe_cache_key("CCO", "openff-2.2.0")] == "ok"


def test_unsupported_probe_writes_disk_sidecar(tmp_path):
    """When every FF in the priority list raises, the result is
    'unsupported' and EACH `(smiles, ff)` lands in the sidecar as
    'unsupported' — so s10 can short-circuit each FF without re-probing.
    """
    md_dir = tmp_path / "md"
    fake_mol = MagicMock()
    fake_mol.to_smiles.return_value = "weird-smiles"
    fake_mol.to_topology.return_value.to_openmm.return_value = "topology"

    fake_sg_module = types.ModuleType("openmmforcefields.generators")

    class _AlwaysFailSG:
        def __init__(self, **kw):
            raise RuntimeError("sqm hard-failed (mocked)")

    fake_sg_module.SystemGenerator = _AlwaysFailSG
    fake_omff = types.ModuleType("openmmforcefields")
    fake_omff.generators = fake_sg_module

    fake_app = types.ModuleType("openmm.app")
    fake_app.HBonds = "HBonds"
    fake_app.CutoffNonPeriodic = "CutoffNonPeriodic"
    fake_openmm = types.ModuleType("openmm")
    fake_openmm.app = fake_app

    sys_mods_patch = {
        "openmmforcefields": fake_omff,
        "openmmforcefields.generators": fake_sg_module,
    }
    if "openmm" not in sys.modules:
        sys_mods_patch["openmm"] = fake_openmm
        sys_mods_patch["openmm.app"] = fake_app

    with patch.object(mdp, "lookup_by_smiles", return_value=None):
        with patch.object(mdp, "_build_offmol", return_value=fake_mol):
            with patch.dict(sys.modules, sys_mods_patch):
                result = mdp.run_md_preflight(
                    smiles="weird-smiles", md_dir=md_dir,
                    prefer_ff="openff-2.2.0",
                )

    assert result.status == "unsupported"
    sidecar = md_dir / ".ligand_probe_cache.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    # At LEAST the preferred FF must be marked unsupported. The exact
    # set depends on which FFs were in the priority list at test time
    # (gaff-2.11 always; espaloma only if installed).
    assert data[_probe_cache_key("weird-smiles", "openff-2.2.0")] == "unsupported"
    assert data[_probe_cache_key("weird-smiles", "gaff-2.11")] == "unsupported"


def test_preflight_sidecar_lives_in_md_dir_not_under_preflight(tmp_path):
    """`_ligand_probe_cache_path(workdir)` returns `workdir.parent /
    .ligand_probe_cache.json`. The preflight uses a synthetic
    `md_dir/_preflight` workdir so the sidecar lands in md_dir — exactly
    where s10's per-candidate code (workdir = md_dir/<candidate_id>)
    will look for it. This test locks that placement."""
    md_dir = tmp_path / "md"
    md_dir.mkdir()
    synthetic = md_dir / "_preflight"
    path = _ligand_probe_cache_path(synthetic)
    assert path == md_dir / ".ligand_probe_cache.json"

    # And a real per-candidate workdir resolves to the SAME path.
    cand_path = _ligand_probe_cache_path(md_dir / "cand_001")
    assert cand_path == path, (
        "preflight sidecar and per-candidate sidecar must be the same file"
    )


# --- Source-level guards on s01 -----------------------------------------


def test_s01_imports_run_md_preflight():
    """A future refactor must not silently delete the s01 hook."""
    import inspect

    from evoliez.stages import s01_input_preprocess as s01
    src = inspect.getsource(s01)
    assert "from evoliez.adapters.md_preflight import run_md_preflight" in src
    assert "run_md_preflight(" in src
    # Must persist the status to ctx.meta so downstream stages + report
    # can read it.
    assert "md_preflight_status" in src
