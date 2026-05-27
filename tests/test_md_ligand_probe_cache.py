"""P0 B — ligand-level probe-deduplication cache.

The openmmforcefields disk JSON (`ff_cache_<ff>.json`) saves the charge
calculation per canonical SMILES, but the *probe* call
`sg.create_system(off_lig.to_topology().to_openmm(), molecules=[off_lig])`
still runs once per candidate to verify the ligand is parameterizable.
With `mdcfg.subprocess_isolation=true` each fresh process repeats the
probe — wasted work on N candidates of the same ligand.

This cache memoises probe outcome by `(canonical_smiles, ff_name)`. The
in-memory dict catches in-process calls; a JSON sidecar at
`workdir.parent/.ligand_probe_cache.json` carries the result across
subprocess boundaries.

These tests exercise the helpers directly (without depending on a real
OpenFF Molecule / openmmforcefields install) so they run in the light
local venv — production correctness is verified by the server smoke.
"""

from __future__ import annotations

import inspect
import json

import pytest

from evoliez.adapters import openmm_engine
from evoliez.adapters.openmm_engine import (
    _canonical_smiles_for_cache,
    _clear_ligand_probe_cache,
    _hydrate_probe_cache,
    _LIGAND_PROBE_CACHE_MEM,
    _ligand_probe_cache_path,
    _persist_probe_result,
    _probe_cache_key,
)


@pytest.fixture(autouse=True)
def _reset_probe_cache():
    """Each test starts with an empty in-memory cache."""
    _clear_ligand_probe_cache()
    yield
    _clear_ligand_probe_cache()


def test_probe_cache_path_lives_one_dir_up_from_candidate_workdir(tmp_path):
    """Same parent dir as `ff_cache_*.json` so the disk sidecar is shared
    across all candidates of one MD run (incl. subprocess-isolated)."""
    cand = tmp_path / "md" / "cand_001"
    cand.mkdir(parents=True)
    p = _ligand_probe_cache_path(cand)
    assert p.parent == cand.parent
    assert p.name == ".ligand_probe_cache.json"


def test_probe_cache_key_uses_pipe_separator():
    """`|` is illegal inside SMILES so it can't accidentally collide with
    a ligand whose name happens to contain the FF token."""
    k = _probe_cache_key("CC(=O)O", "gaff-2.11")
    assert k == "CC(=O)O|gaff-2.11"


def test_persist_and_hydrate_round_trip(tmp_path):
    """Writing 'ok' to disk and rehydrating in a fresh process should
    restore the entry — that's what makes the cache work across the
    subprocess-isolation boundary."""
    disk = tmp_path / ".ligand_probe_cache.json"
    key = _probe_cache_key("OC", "openff-2.2.0")

    _persist_probe_result(disk, key, "ok")
    assert disk.exists()
    assert json.loads(disk.read_text())[key] == "ok"

    # Simulate a fresh process: clear in-memory + force re-hydrate.
    _clear_ligand_probe_cache()
    _hydrate_probe_cache(disk)
    assert _LIGAND_PROBE_CACHE_MEM[key] == "ok"


def test_persist_merges_with_existing_disk_state(tmp_path):
    """Concurrent candidates (parallel subprocesses) caching different
    keys shouldn't overwrite each other. The persist helper reads the
    current disk file before writing back."""
    disk = tmp_path / ".ligand_probe_cache.json"
    k1 = _probe_cache_key("CCO", "gaff-2.11")
    k2 = _probe_cache_key("CCN", "openff-2.2.0")

    _persist_probe_result(disk, k1, "ok")
    _persist_probe_result(disk, k2, "unsupported")

    data = json.loads(disk.read_text())
    assert data == {k1: "ok", k2: "unsupported"}


def test_hydrate_is_idempotent_per_disk_path(tmp_path):
    """Hydrating twice should not double-load entries or refetch the file
    on the second call — `_PROBE_CACHE_DISK_LOADED` tracks visited paths."""
    disk = tmp_path / ".ligand_probe_cache.json"
    disk.write_text(json.dumps({"OC|gaff-2.11": "ok"}))
    _hydrate_probe_cache(disk)
    # Modify disk; second hydrate should NOT pick up the change (path
    # already marked loaded). That's the contract: hydrate is a per-process
    # one-shot bootstrap, not a polling reload.
    disk.write_text(json.dumps({"OC|gaff-2.11": "ok", "C|gaff-2.11": "ok"}))
    _hydrate_probe_cache(disk)
    assert "OC|gaff-2.11" in _LIGAND_PROBE_CACHE_MEM
    assert "C|gaff-2.11" not in _LIGAND_PROBE_CACHE_MEM


def test_hydrate_tolerates_corrupt_disk_file(tmp_path):
    """A half-written sidecar (e.g. crashed during a non-atomic legacy
    write) must not crash the next MD run."""
    disk = tmp_path / ".ligand_probe_cache.json"
    disk.write_text("{not valid json")
    _hydrate_probe_cache(disk)
    # No entries restored, but no exception either.
    assert _LIGAND_PROBE_CACHE_MEM == {}


def test_persist_uses_atomic_rename(tmp_path):
    """Tmp file must not survive after persist completes (atomic-rename
    moves it into place). Otherwise a crashed candidate could leave
    `<sidecar>.json.tmp` files cluttering the MD dir."""
    disk = tmp_path / ".ligand_probe_cache.json"
    _persist_probe_result(disk, "OC|gaff-2.11", "ok")
    leftovers = list(tmp_path.glob(".ligand_probe_cache.json*"))
    # Only the final file, no .tmp residue.
    assert [p.name for p in leftovers] == [".ligand_probe_cache.json"]


def test_canonical_smiles_safe_on_failure():
    """A non-OpenFF object passed in (e.g. a future refactor breaking the
    duck-type) must return empty string, NOT raise — the caller treats
    empty key as 'cache disabled for this ligand', which is the safe
    fallback (we'll re-probe instead of silently skipping)."""
    class _NotOpenFF:
        def to_smiles(self, *_, **__):
            raise RuntimeError("not a real molecule")
    assert _canonical_smiles_for_cache(_NotOpenFF()) == ""


def test_clear_resets_both_memory_and_disk_loaded(tmp_path):
    """`_clear_ligand_probe_cache` is the test/teardown hook. After a
    clear, re-hydrating from the same disk path must work again."""
    disk = tmp_path / ".ligand_probe_cache.json"
    disk.write_text(json.dumps({"OC|gaff-2.11": "ok"}))
    _hydrate_probe_cache(disk)
    assert _LIGAND_PROBE_CACHE_MEM["OC|gaff-2.11"] == "ok"

    _clear_ligand_probe_cache()
    assert _LIGAND_PROBE_CACHE_MEM == {}

    # Same disk path is "fresh" again (re-hydrate works).
    _hydrate_probe_cache(disk)
    assert _LIGAND_PROBE_CACHE_MEM["OC|gaff-2.11"] == "ok"


# --- Source-level contract guards (so a future refactor can't silently
# drop the cache hookup from one of the two generator functions). -----


def test_gasteiger_generator_consults_probe_cache():
    src = inspect.getsource(openmm_engine._gasteiger_charge_system_generator)
    assert "_ligand_probe_cache_path" in src, (
        "Gasteiger probe must check the disk-sidecar cache"
    )
    assert "_LIGAND_PROBE_CACHE_MEM" in src, (
        "Gasteiger probe must check in-memory cache"
    )
    assert "_persist_probe_result" in src, (
        "Gasteiger probe must persist its outcome"
    )


def test_ligand_system_generator_consults_probe_cache():
    src = inspect.getsource(openmm_engine._ligand_system_generator)
    assert "_ligand_probe_cache_path" in src
    assert "_LIGAND_PROBE_CACHE_MEM" in src
    assert "_persist_probe_result" in src
    # Unsupported caching: the per-FF loop must skip a known-bad FF
    # without re-probing.
    assert 'cached_status == "unsupported"' in src or (
        '"unsupported"' in src and "continue" in src
    )


def test_ligand_system_generator_skips_probe_on_cached_ok():
    """The OK path must NOT call sg.create_system again when the cache
    says we've already proven this (SMILES, FF) is parameterizable."""
    src = inspect.getsource(openmm_engine._ligand_system_generator)
    assert 'cached_status == "ok"' in src
    assert "skipping probe" in src
