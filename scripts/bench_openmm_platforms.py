#!/usr/bin/env python
"""Standalone OpenMM platform benchmark (no evoliez deps).

Builds a protein-only implicit-solvent system (Amber ff14SB + GBSA obc2 - the
same force-field family s10 uses) and runs the SAME short production MD on each
available platform (CUDA / OpenCL / CPU), reporting wall-time and ns/day.

Protein-only is a faithful proxy for s10's per-candidate MD cost: the protein's
thousands of atoms dominate the force evaluation; the small ligand is negligible
for a platform-speed comparison. Run in an env whose OpenMM CUDA build matches
the driver (openmm-cuda124). Usage:

    python bench_openmm_platforms.py <protein.pdb> [nsteps]
"""
import sys
import time

import openmm as mm
import openmm.app as app
from openmm import unit
import pdbfixer

PDB = sys.argv[1]
NSTEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
TS_FS = 2.0
WARMUP = 100

# --- build the protein-only system once (shared across platforms) -----------
fixer = pdbfixer.PDBFixer(filename=PDB)
fixer.removeHeterogens(False)                 # drop ligand/ions/water
fixer.findMissingResidues()
fixer.missingResidues = {}                    # don't model gaps, just cap atoms
fixer.findMissingAtoms()
fixer.addMissingAtoms()
ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
modeller = app.Modeller(fixer.topology, fixer.positions)
modeller.addHydrogens(ff)
system = ff.createSystem(
    modeller.topology,
    nonbondedMethod=app.CutoffNonPeriodic,
    nonbondedCutoff=2.0 * unit.nanometer,
    constraints=app.HBonds,
)

# Weak CA positional restraints keep the short benchmark MD from exploding
# (Boltz coords + addHydrogens carry some strain; s10 restrains likewise). The
# restraint is one cheap term per CA - negligible vs nonbonded+GBSA on ~6k
# atoms - so the platform-to-platform timing stays a fair comparison.
restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
restraint.addGlobalParameter("k", 1000.0)     # kJ/mol/nm^2
for _pn in ("x0", "y0", "z0"):
    restraint.addPerParticleParameter(_pn)
_pos_nm = modeller.positions.value_in_unit(unit.nanometer)
for _atom in modeller.topology.atoms():
    if _atom.name == "CA":
        _x, _y, _z = _pos_nm[_atom.index]
        restraint.addParticle(_atom.index, [_x, _y, _z])
system.addForce(restraint)

natoms = system.getNumParticles()
print(f"system: {natoms} atoms | {NSTEPS} steps @ {TS_FS} fs "
      f"({NSTEPS * TS_FS / 1000:.1f} ps) | {restraint.getNumParticles()} CA restraints\n")

# Minimize ONCE on the fastest available platform -> a stable starting
# structure reused by every platform (so each benchmarks the SAME relaxed coords
# and none blows up). Capped iterations: with CA restraints this is plenty to
# clear addHydrogens clashes, and an uncapped full minimize can run for minutes.
_avail = [mm.Platform.getPlatform(i).getName()
          for i in range(mm.Platform.getNumPlatforms())]
_minname = ("OpenCL" if "OpenCL" in _avail
            else "CUDA" if "CUDA" in _avail else "CPU")
_integ0 = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 1.0 / unit.picosecond, TS_FS * unit.femtoseconds)
_ctx0 = mm.Context(system, _integ0, mm.Platform.getPlatformByName(_minname))
_ctx0.setPositions(modeller.positions)
mm.LocalEnergyMinimizer.minimize(_ctx0, maxIterations=1000)
MINPOS = _ctx0.getState(getPositions=True).getPositions()
del _ctx0, _integ0
print(f"(minimized on {_minname}, 1000 iter)\n")


def bench(name):
    try:
        platform = mm.Platform.getPlatformByName(name)
    except Exception:
        return f"{name:8s}: platform not registered"
    integ = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, TS_FS * unit.femtoseconds)
    try:
        ctx = mm.Context(system, integ, platform)
    except Exception as exc:
        return f"{name:8s}: Context FAILED ({str(exc)[:70]})"
    ctx.setPositions(MINPOS)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin)
    integ.step(WARMUP)                         # absorb JIT / kernel compile
    ctx.getState(getEnergy=True)               # sync
    t0 = time.time()
    integ.step(NSTEPS)
    ctx.getState(getEnergy=True)               # force completion before timing
    dt = time.time() - t0
    ns = NSTEPS * TS_FS / 1e6
    nsday = ns / (dt / 86400.0) if dt > 0 else float("inf")
    dev = ""
    try:
        dev = platform.getPropertyValue(ctx, "DeviceName")
    except Exception:
        pass
    del ctx, integ
    return f"{name:8s}: {dt:7.2f}s  ->  {nsday:9.1f} ns/day   {dev}"


results = {}
for p in ["CUDA", "OpenCL", "CPU"]:
    line = bench(p)
    print(line)
    results[p] = line


def nsday_of(line):
    try:
        return float(line.split("->")[1].split("ns/day")[0])
    except Exception:
        return None


cu, ocl, cpu = (nsday_of(results[k]) for k in ("CUDA", "OpenCL", "CPU"))
print()
if cu and ocl:
    print(f"CUDA / OpenCL : {cu / ocl:.2f}x faster")
if ocl and cpu:
    print(f"OpenCL / CPU  : {ocl / cpu:.2f}x faster")
if cu and cpu:
    print(f"CUDA / CPU    : {cu / cpu:.2f}x faster")
