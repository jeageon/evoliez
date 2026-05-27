"""D-minimal: phase-specific MD log markers.

The expert audit's diagnosis: TEM-1 third PASS is unexplained because
the worker.py only emitted `worker_start / run_md_start / run_md_done`.
Without phase-specific log markers, a future hang in (say)
`SystemGenerator.create_system` or `Simulation.__init__` is
indistinguishable from a hang in `am1bcc charge calc`. This commit
adds 12 phase markers inside `_run_real`; the test below locks the
contract so a future refactor can't silently drop them.
"""

from __future__ import annotations

import inspect

from evoliez.adapters import openmm_engine


_EXPECTED_PHASES = [
    "run_real_start",
    "openmm_imported",
    "topology_loaded",
    "ligand_build_start",
    "ligand_build_done",
    "parameterization_start",
    # parameterization_done is templated as f"parameterization_done_{_ff}"
    "parameterization_done",
    "protein_prep_start",
    "modeller_built",
    "create_system_start",
    "create_system_done",
    "context_create_start",
    "context_create_done",
    "minimize_start",
    "minimize_done",
    "production_start",
    "production_done",
    # run_real_done is f"run_real_done_{status}"
    "run_real_done",
]


def test_run_real_emits_all_phase_markers():
    src = inspect.getsource(openmm_engine._run_real)
    # Accept both literal `_phase("name"` and f-string `_phase(f"name`
    # (used for the templated `parameterization_done_{_ff}` and
    # `run_real_done_{status}` markers so the status suffix lands in
    # the log line without an extra emit).
    missing = [
        p for p in _EXPECTED_PHASES
        if f'_phase("{p}' not in src and f'_phase(f"{p}' not in src
    ]
    assert not missing, f"missing phase markers in _run_real: {missing}"


def test_phase_helper_uses_canonical_prefix():
    """Worker.py and openmm_engine.py must use the same `[evoliez-md-phase]`
    prefix so `grep '\\[evoliez-md-phase\\]'` finds both subprocess-
    internal and worker-boundary markers in the same log."""
    src = inspect.getsource(openmm_engine)
    assert '_PHASE_PREFIX = "[evoliez-md-phase]"' in src


def test_curated_path_also_emits_phase_markers():
    """When the curated AMBER cofactor path is taken (NADP+ on a server
    with Bryce Lab files), we want phase markers from that branch too."""
    src = inspect.getsource(openmm_engine._run_real)
    assert '_phase("curated_param_start"' in src
    assert '_phase("curated_param_done"' in src
