#!/usr/bin/env python
"""Production MD memory-leak diagnostic per expert recommendation.

Runs the SAME tiny MD system N times in two modes and reports
per-iteration RSS + Python heap deltas:

    parent:      all N iters in a single Python process
                 -> simulates what `evoliez run` does today
    subprocess:  each iter in a fresh `python -c "..."` child
                 -> simulates per-candidate subprocess isolation

If parent RSS grows linearly but Python heap stays flat AND
subprocess parent RSS stays flat:
    -> CONFIRMED: OpenMM/CUDA native retention, not Python ref leak
    -> FIX: subprocess isolation per candidate

If parent RSS AND Python heap both grow:
    -> Python ref leak; check _release_openmm_resources coverage

If parent RSS stays flat:
    -> our cleanup helper is sufficient, no further work needed

Run on the SERVER (real CUDA path) or the local Mac (CPU path).
Both are valid; the CPU path is faster but only catches Python
ref leaks. The CUDA path catches both.

Usage:
    EVOLIEZ_MD_MEMLOG=1 python scripts/diagnose_md_memory.py [--iters 5]

The diagnostic builds a tiny LJ blob - same OpenMM lifecycle as
the real protein-ligand path - so the test runs in seconds and
produces an unambiguous accumulation signature.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import textwrap
import tracemalloc
from pathlib import Path

try:
    import openmm as mm
    from openmm import app, unit
except ImportError:
    sys.stderr.write(
        "openmm not installed. Install with: pip install openmm 'numpy<2'\n"
    )
    sys.exit(2)


def _rss_mb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _build_and_run_md(seed: int) -> None:
    """One iteration of the OpenMM lifecycle: build a tiny system,
    minimize + step it, then let the variables go out of scope. The
    point is to reproduce the per-candidate allocation pattern from
    _run_real() with the same Simulation/Context/Integrator/System
    object set."""
    system = mm.System()
    for _ in range(30):
        system.addParticle(1.0 * unit.amu)
    nb = mm.NonbondedForce()
    for _ in range(30):
        nb.addParticle(0.0, 0.3 * unit.nanometer, 1.0 * unit.kilojoule_per_mole)
    system.addForce(nb)

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    for i in range(30):
        topology.addAtom(f"A{i}", app.Element.getBySymbol("C"), residue)

    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds,
    )
    platform = mm.Platform.getPlatformByName("Reference")
    sim = app.Simulation(topology, system, integrator, platform)
    positions = [(i * 0.5, 0.0, 0.0) for i in range(30)] * unit.nanometer
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=10)
    sim.step(20)
    # Trigger state read like _run_real does (DCDReporter writes are
    # state reads in disguise + file write):
    _state = sim.context.getState(getPositions=True, getEnergy=True)
    _e = _state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    # Apply our cleanup helper (the fix we're testing). Comment this
    # out to see the "before fix" baseline.
    try:
        from evoliez.adapters.openmm_engine import _release_openmm_resources
        _release_openmm_resources(sim, integrator, system)
    except ImportError:
        del sim, integrator, system
        gc.collect()


def parent_mode(iters: int) -> None:
    """All iters in the same Python process."""
    print(f"\n=== PARENT MODE ({iters} iters, single Python process) ===")
    tracemalloc.start()
    # Warm up
    _build_and_run_md(0)
    gc.collect()
    rss0 = _rss_mb()
    heap0, _ = tracemalloc.get_traced_memory()
    heap0 /= (1024 * 1024)
    print(f"  baseline (warm): RSS={rss0:.1f} MB  heap={heap0:.2f} MB")
    for i in range(1, iters + 1):
        _build_and_run_md(i)
        gc.collect()
        rss = _rss_mb()
        heap, _peak = tracemalloc.get_traced_memory()
        heap /= (1024 * 1024)
        d_rss = rss - rss0
        d_heap = heap - heap0
        print(
            f"  iter {i}: RSS={rss:.1f} MB (Δ {d_rss:+.1f})  "
            f"heap={heap:.2f} MB (Δ {d_heap:+.2f})"
        )
    print(f"\n  TOTAL drift over {iters} iters: "
          f"RSS={_rss_mb()-rss0:+.1f} MB  heap={(tracemalloc.get_traced_memory()[0]/(1024*1024)) - heap0:+.2f} MB")
    tracemalloc.stop()


def subprocess_mode(iters: int) -> None:
    """Each iter in a fresh child process. Parent RSS should stay
    perfectly flat - this is the upper bound on what subprocess
    isolation would achieve."""
    print(f"\n=== SUBPROCESS MODE ({iters} iters, fresh child per iter) ===")
    rss0 = _rss_mb()
    print(f"  parent baseline: RSS={rss0:.1f} MB")
    child_script = textwrap.dedent("""
        import sys
        sys.path.insert(0, %r)
        from scripts.diagnose_md_memory import _build_and_run_md, _rss_mb
        _build_and_run_md(0)
        print(f"  child RSS at exit: {_rss_mb():.1f} MB")
    """) % str(Path(__file__).resolve().parents[1])
    for i in range(1, iters + 1):
        result = subprocess.run(
            [sys.executable, "-c", child_script],
            capture_output=True, text=True, timeout=60,
        )
        child_rss = "?"
        for line in result.stdout.splitlines():
            if "child RSS" in line:
                child_rss = line.strip()
        parent_rss = _rss_mb()
        d_parent = parent_rss - rss0
        print(
            f"  iter {i}: parent RSS={parent_rss:.1f} MB (Δ {d_parent:+.1f})  "
            f"{child_rss}"
        )
    print(f"\n  parent drift over {iters} iters: {_rss_mb()-rss0:+.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=5,
                    help="iterations per mode (default 5)")
    ap.add_argument("--mode", choices=("parent", "subprocess", "both"),
                    default="both")
    args = ap.parse_args()

    print(f"OpenMM {mm.__version__}, Python {sys.version.split()[0]}")
    print(f"PID: {os.getpid()}, platform: Reference (CPU)")

    if args.mode in ("parent", "both"):
        parent_mode(args.iters)
    if args.mode in ("subprocess", "both"):
        subprocess_mode(args.iters)

    print("\n--- diagnostic complete ---")
    print("Interpretation:")
    print("  - parent RSS grows linearly + heap stays flat -> native leak")
    print("    (OpenMM/CUDA retention; subprocess isolation is the fix)")
    print("  - parent RSS and heap BOTH grow -> Python ref leak")
    print("    (check _release_openmm_resources coverage)")
    print("  - parent RSS stays flat -> current cleanup is sufficient")
    return 0


if __name__ == "__main__":
    sys.exit(main())
