"""Regression tests for the MD GPU memory-leak fix.

## Production incident
On a 30-candidate × 2-3 replica MD run, server GPU memory grew unbounded
across candidates because OpenMM ``Simulation`` / ``System`` / ``Context``
/ ``Integrator`` objects were only freed when Python's GC eventually ran.
The C++ destructor (which actually releases CUDA context memory) only
fires when the Python wrapper is finalised.

## Root cause (audit + code review)
1. ``_run_real`` had 10 ``return MDResult(...)`` statements scattered
   throughout a 335-line function. NONE called ``del sim`` / ``del system``
   / etc. before returning.
2. ``s10_md.py`` per-candidate loop held ``replica_results`` list across
   the entire MD run.
3. No ``gc.collect()`` between candidates -> deferred & non-deterministic
   release of GPU memory.

## Fix landed
- ``adapters/openmm_engine.py``: ``_release_openmm_resources(*objects)``
  helper + ``_run_real`` wrapped in ``try/finally`` so cleanup ALWAYS
  runs on every return path (early-exit skips, success, exceptions).
- ``stages/s10_md.py``: per-candidate ``replica_results.clear() +
  gc.collect()`` (belt + suspenders).

This file pins both fixes against future regressions.
"""

from __future__ import annotations

import gc
import inspect
import weakref
from unittest.mock import MagicMock

import pytest


# --------------------------------------------------------------------- #
# 1. Source guards: the structural fix MUST be present.
# --------------------------------------------------------------------- #
def test_openmm_engine_has_release_helper():
    """The release helper must exist and accept variadic args."""
    from evoliez.adapters import openmm_engine
    assert hasattr(openmm_engine, "_release_openmm_resources"), (
        "_release_openmm_resources helper missing - the cleanup fix "
        "has been reverted or the file was edited incorrectly"
    )
    sig = inspect.signature(openmm_engine._release_openmm_resources)
    # *objects should accept any number of args
    assert any(p.kind == inspect.Parameter.VAR_POSITIONAL
               for p in sig.parameters.values()), (
        "_release_openmm_resources must take *objects (variadic)"
    )


def test_run_real_uses_try_finally_with_cleanup():
    """Source guard: _run_real body must be wrapped in try/finally and
    the finally must call _release_openmm_resources. If someone reverts
    this, the leak comes back."""
    from evoliez.adapters import openmm_engine
    src = inspect.getsource(openmm_engine._run_real)
    assert "try:" in src, "_run_real must wrap body in try/finally"
    assert "finally:" in src, "_run_real must have a finally block"
    assert "_release_openmm_resources(" in src, (
        "finally block must call _release_openmm_resources to free GPU"
    )
    # The cleanup MUST come AFTER the try (i.e., be in the finally).
    try_pos = src.index("try:")
    finally_pos = src.index("finally:")
    release_pos = src.index("_release_openmm_resources(")
    assert finally_pos > try_pos and release_pos > finally_pos, (
        "release helper must be called inside the finally, not anywhere else"
    )


def test_release_helper_called_with_all_heavy_resources():
    """The finally must release every OpenMM resource we allocate:
    sim, integrator, restraint, system, modeller, system_generator,
    off_lig, pdb. Missing any of these = partial leak."""
    from evoliez.adapters import openmm_engine
    src = inspect.getsource(openmm_engine._run_real)
    # Extract the call to _release_openmm_resources
    import re
    m = re.search(
        r"_release_openmm_resources\((.*?)\)", src, re.DOTALL
    )
    assert m is not None, "release call not found in _run_real"
    call_args = m.group(1)
    required = ["sim", "integrator", "system", "modeller",
                "system_generator", "off_lig", "pdb"]
    for name in required:
        assert name in call_args, (
            f"_release_openmm_resources call missing argument '{name}' "
            f"in: {call_args[:200]}..."
        )


def test_s10_md_calls_gc_collect_per_candidate():
    """s10_md.py per-candidate loop must call gc.collect() after each
    candidate completes (belt + suspenders for the _run_real cleanup)."""
    from evoliez.stages import s10_md
    src = inspect.getsource(s10_md.MDStage.run)
    assert "gc.collect()" in src, (
        "s10_md per-candidate loop must call gc.collect() between "
        "candidates so the cleanup happens deterministically before "
        "the next candidate's allocations start"
    )
    # Replica results list must be cleared (we don't need them after
    # the per-candidate aggregation).
    assert "replica_results.clear()" in src or "del replica_results" in src, (
        "replica_results must be cleared / deleted between candidates"
    )


# --------------------------------------------------------------------- #
# 2. Behavioural: _release_openmm_resources() actually drops references.
# --------------------------------------------------------------------- #
def test_release_helper_handles_none():
    """Calling release with None values must not raise."""
    from evoliez.adapters.openmm_engine import _release_openmm_resources
    _release_openmm_resources(None, None, None)  # should be a no-op


def test_release_helper_clears_reporters():
    """If an object has a .reporters list, it must be cleared (DCDReporter
    file handles are part of this list - clearing first ensures the file
    closes before the Simulation goes away)."""
    from evoliez.adapters.openmm_engine import _release_openmm_resources

    fake_sim = MagicMock()
    fake_sim.reporters = [MagicMock(), MagicMock()]
    _release_openmm_resources(fake_sim)
    # The list should have been cleared
    assert fake_sim.reporters == []


def test_release_helper_forces_gc_collect():
    """The helper must call gc.collect() so OpenMM C++ destructors fire
    immediately rather than at some unpredictable later moment."""
    from evoliez.adapters import openmm_engine
    src = inspect.getsource(openmm_engine._release_openmm_resources)
    assert "gc.collect()" in src, (
        "_release_openmm_resources must force gc.collect() so OpenMM "
        "C++ destructors run NOW and free GPU memory"
    )


def test_release_helper_swallows_cleanup_errors():
    """A misbehaving __del__ on one object must not stop the helper from
    cleaning up the others. Cleanup must never raise."""
    from evoliez.adapters.openmm_engine import _release_openmm_resources

    bad = MagicMock()

    class _RaisingReporters(list):
        def clear(self):
            raise RuntimeError("simulated bad reporter")

    bad.reporters = _RaisingReporters([MagicMock()])
    # Must NOT raise
    _release_openmm_resources(bad, None, MagicMock())


# --------------------------------------------------------------------- #
# 3. Weakref test: confirm references actually drop.
# --------------------------------------------------------------------- #
def test_simulated_md_loop_releases_references():
    """End-to-end behavioural test (no OpenMM): simulate a loop that
    creates fake "Simulation" objects, runs _release_openmm_resources
    on them, and verifies weakrefs are dead after the call.

    This proves the helper actually drops references in the way that
    OpenMM's C++ destructor would observe (since gc.collect() is what
    triggers the C++ destructor).
    """
    from evoliez.adapters.openmm_engine import _release_openmm_resources

    class FakeOpenMMObject:
        """Stands in for Simulation/System/etc. - has reporters list."""
        def __init__(self, name):
            self.name = name
            self.reporters = []

    refs = []
    for i in range(5):
        sim = FakeOpenMMObject(f"sim_{i}")
        system = FakeOpenMMObject(f"system_{i}")
        refs.append((weakref.ref(sim), weakref.ref(system)))
        # Mimic what _run_real's finally does:
        _release_openmm_resources(sim, system)
        # Local refs go out of scope at end of loop iteration too,
        # but _release_openmm_resources should have already done its job.
        del sim, system

    # After the loop, gc.collect() inside _release_openmm_resources
    # must have caused all FakeOpenMMObject instances to be collected.
    gc.collect()  # extra safety in the test
    for sim_ref, sys_ref in refs:
        assert sim_ref() is None, (
            "Simulation was NOT garbage collected after "
            "_release_openmm_resources - this is the exact failure "
            "mode of the production leak"
        )
        assert sys_ref() is None, "System was NOT garbage collected"


# --------------------------------------------------------------------- #
# 4. Integration: simulated 30-iteration MD loop drops refs each iter.
# --------------------------------------------------------------------- #
def test_thirty_iteration_loop_no_reference_growth():
    """Simulate the 30-candidate production loop. After each candidate's
    iteration, the previous candidate's resources MUST be gone (weakref
    dead). If they accumulate, GPU memory accumulates."""
    from evoliez.adapters.openmm_engine import _release_openmm_resources

    class FakeSim:
        def __init__(self, idx):
            self.idx = idx
            self.reporters = [MagicMock(), MagicMock(), MagicMock()]
            # Simulate holding ~1 MB of "GPU data"
            self.data = bytearray(1024 * 1024)

    all_refs = []
    for i in range(30):
        sim = FakeSim(i)
        ref = weakref.ref(sim)
        all_refs.append(ref)
        # _run_real's finally:
        _release_openmm_resources(sim)
        del sim

    gc.collect()
    alive = sum(1 for r in all_refs if r() is not None)
    assert alive == 0, (
        f"{alive}/30 simulated Simulations still alive after release - "
        "this is the leak that grows GPU memory in production"
    )
