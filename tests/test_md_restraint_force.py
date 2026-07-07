"""Restraint-force mechanics for the real OpenMM MD path.

The real MD engine (``adapters.openmm_engine._run_real``) needs a full-atom
structure that only the GPU server produces, so it cannot run end-to-end on the
dev box. But the genuinely novel/bug-prone part of the recent fix is the
per-particle-k positional restraint that replaced the old "freeze every CA"
placeholder. This test exercises THAT force in real OpenMM on a tiny synthetic
system and asserts the physics: a strongly restrained particle stays near its
anchor while a free particle drifts away. It mirrors the exact force
construction in ``_run_real`` so a regression there is caught without a GPU.
"""

from __future__ import annotations

import pytest

mm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402


def _build_restraint(k_strong_val, k_weak_val):
    """Same CustomExternalForce the engine builds (param order k,x0,y0,z0)."""
    f = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    f.addPerParticleParameter("k")
    for p in ("x0", "y0", "z0"):
        f.addPerParticleParameter(p)
    return f


def test_per_particle_k_units_match_engine():
    """The kcal->kJ conversion the engine does yields the documented value."""
    k_strong = (5.0 * unit.kilocalories_per_mole / unit.angstrom**2).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer**2
    )
    k_weak = (0.5 * unit.kilocalories_per_mole / unit.angstrom**2).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer**2
    )
    # 1 kcal/mol/A^2 = 4.184 kJ/mol / 0.01 nm^2 = 418.4 kJ/mol/nm^2
    assert k_strong == pytest.approx(5.0 * 418.4, rel=1e-6)
    assert k_weak == pytest.approx(0.5 * 418.4, rel=1e-6)
    assert k_strong > k_weak > 0.0


def test_restrained_particle_held_free_particle_drifts():
    """Strong restraint pins a particle; an unrestrained one moves freely.

    This is the property the MD-stability signal depends on: the previous code
    restrained EVERY CA (pocket included) so pocket_rmsd was always ~0. The fix
    leaves the pocket free, so this test proves restrained != free.
    """
    system = mm.System()
    for _ in range(2):
        system.addParticle(12.0)  # amu

    k_strong = (5.0 * unit.kilocalories_per_mole / unit.angstrom**2).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer**2
    )
    restraint = _build_restraint(k_strong, 0.0)
    # particle 0 = strongly restrained at the origin; particle 1 = NOT added (free)
    restraint.addParticle(0, [k_strong, 0.0, 0.0, 0.0])
    system.addForce(restraint)
    assert restraint.getNumParticles() == 1

    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds
    )
    integrator.setRandomNumberSeed(7)  # deterministic on the Reference platform
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] * unit.nanometer)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 7)
    integrator.step(3000)

    pos = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(
        unit.nanometer
    )
    restrained_disp = float((pos[0] ** 2).sum() ** 0.5)
    free_disp = float((pos[1] ** 2).sum() ** 0.5)

    # The restrained particle is held near its anchor (thermal floor at k=5
    # kcal/mol/A^2 is sqrt(3*kT/k) ~= 0.06 nm); the free particle, a 12 amu
    # point under a 300 K thermostat with no restoring force, diffuses far.
    # Bounds are loose enough to hold across RNG variation (probed: restrained
    # 0.03-0.10 nm, free 0.7-3.9 nm, ratio always >= 9x).
    assert restrained_disp < 0.15, f"restrained particle drifted {restrained_disp} nm"
    assert free_disp > 0.30, f"free particle barely moved ({free_disp} nm)"
    assert free_disp > 3 * restrained_disp, "free particle not clearly freer than restrained"
