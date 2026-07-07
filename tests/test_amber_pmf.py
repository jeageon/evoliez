"""V6-3 unit tests for the Amber PMF pure helpers.

Locks the physics-critical restraint convention (Amber rk = 0.5·k so the well
matches md.umbrella.wham's 0.5·k·(r-r0)² convention), the DUMPAVE/mdin plumbing,
and the convergence -> claim classification (non-converged is diagnostic-only,
ranking-prohibited). The pmemd/cpptraj fan-out is server-only.
"""
import numpy as np
import pytest

from evoliez.amber.pmf import (
    amber_umbrella_restraint, classify_pmf, convergence_check,
    discard_equilibration, mdin_umbrella_prod, parse_dumpave,
)


def test_restraint_uses_half_k_convention():
    # k=250 (0.5k convention for wham) -> Amber rk2=rk3=125
    txt = amber_umbrella_restraint(10, 20, 3.2, 250.0)
    assert "iat=10,20," in txt
    assert "r2=3.2000, r3=3.2000," in txt          # harmonic minimum at r0
    assert "rk2=125.00000, rk3=125.00000," in txt  # rk = 0.5·k  (THE conversion)
    assert "r1=0.01" in txt and "r4=30.0" in txt   # wide -> purely harmonic in range


def test_mdin_umbrella_has_nmropt_and_dumpave():
    m = mdin_umbrella_prod(50000, 250, 200)
    assert "nmropt=1" in m
    assert "DISANG=umbrella.rst" in m
    assert "DUMPAVE=umbrella_dist.dat" in m
    assert "type='DUMPFREQ', istep1=200" in m


def test_parse_dumpave():
    txt = "#step dist\n100 3.201\n200 3.512\n300 3.108\n"
    assert parse_dumpave(txt) == [3.201, 3.512, 3.108]


def test_discard_equilibration():
    s = list(range(10))
    assert discard_equilibration(s, 0.2) == list(range(2, 10))
    assert discard_equilibration(s, 0.0) == s
    assert len(discard_equilibration([1.0], 0.5)) == 1   # never empties


def test_classify_pmf_converged_vs_diagnostic():
    good_overlap = {"sufficient": True, "min_overlap": 0.3}
    bad_overlap = {"sufficient": False, "min_overlap": 0.02}
    consistent = {"halves_consistent": True, "access_drift_kcal": 0.4}
    inconsistent = {"halves_consistent": False, "access_drift_kcal": 5.0}

    v, c, ceil = classify_pmf(good_overlap, consistent)
    assert v == "converged" and c == "screening_prioritization"
    assert "NOT activity" in ceil

    # any failing gate -> diagnostic-only, ranking prohibited
    for ov, cv in [(bad_overlap, consistent), (good_overlap, inconsistent),
                   (bad_overlap, inconsistent)]:
        v, c, ceil = classify_pmf(ov, cv)
        assert v == "diagnostic_only" and c == "diagnostic_only"
        assert "prohibited from ranking" in ceil


def test_convergence_check_runs_and_reports_halves():
    # synthetic: 4 windows centred 3.0..4.5, Gaussian samples around each centre
    rng = np.random.RandomState(0)
    centers = [3.0, 3.5, 4.0, 4.5]
    samples = [list(rng.normal(c, 0.2, 400)) for c in centers]
    conv = convergence_check(samples, centers, k_kcal=100.0, near_attack_A=3.6, T_K=300.0)
    assert "halves_consistent" in conv
    assert conv["min_window_samples"] == 400
    # both halves should yield an access cost (near-attack window 3.6 is covered)
    assert conv.get("access_first_half_kcal") is not None
