"""ROADMAP_V5 E4a — umbrella/PMF access-barrier analysis. The per-window MD is server-only (OpenMM),
but the WHAM reconstruction + access-barrier extraction are pure-python and MUST be correct: this
verifies WHAM recovers a KNOWN harmonic PMF from synthetic harmonically-biased samples."""
import numpy as np
import pytest

from evoliez.md.umbrella import access_free_energy, umbrella_windows, wham


def test_umbrella_windows_span_and_count():
    w = umbrella_windows(2.8, 5.4, 14)
    assert len(w) == 14
    assert w[0] == 2.8 and w[-1] == 5.4
    assert all(w[i] < w[i + 1] for i in range(len(w) - 1))
    with pytest.raises(ValueError):
        umbrella_windows(3.0, 4.0, 1)


def test_wham_recovers_known_harmonic_pmf():
    """True PMF = 0.5*K*(d-d_eq)^2. Biased sampling in each window is a Gaussian (product of the true
    well + the harmonic bias), sampled analytically. WHAM must reconstruct the true harmonic."""
    rng = np.random.default_rng(42)
    K_true, d_eq, k_bias, T = 2.0, 4.0, 10.0, 300.0
    kT = 0.0019872041 * T
    centers = umbrella_windows(2.8, 5.4, 14)
    samples = []
    for d0 in centers:
        Ktot = K_true + k_bias
        mean = (K_true * d_eq + k_bias * d0) / Ktot     # product-of-Gaussians mean
        std = np.sqrt(kT / Ktot)
        samples.append(rng.normal(mean, std, 4000))
    x, pmf = wham(samples, centers, k_bias, T_K=T, n_bins=50)
    true = 0.5 * K_true * (x - d_eq) ** 2
    true = true - np.nanmin(true)
    m = np.isfinite(pmf) & (true < 2.5)                 # compare where the well is sampled
    assert np.nanmax(np.abs(pmf[m] - true[m])) < 0.4    # kcal/mol
    assert abs(x[np.nanargmin(pmf)] - d_eq) < 0.3       # minimum near d_eq


def test_wham_needs_samples():
    with pytest.raises(ValueError):
        wham([[], []], [3.0, 4.0], 10.0)


def test_access_free_energy_extracts_near_attack_cost():
    # PMF minimum at 4.5 A; near-attack (<=3.6) costs ~ the PMF value there
    x = np.linspace(2.8, 5.5, 50)
    pmf = 0.5 * 2.0 * (x - 4.5) ** 2
    pmf = pmf - pmf.min()
    cost = access_free_energy(x, pmf, near_attack_max_A=3.6)
    # analytic: 0.5*2*(3.6-4.5)^2 = 0.81 kcal/mol (min in the <=3.6 window is at 3.6)
    assert cost is not None and abs(cost - 0.81) < 0.2


def test_access_free_energy_none_when_window_unsampled():
    x = np.linspace(4.0, 6.0, 30)                       # never reaches <=3.6
    pmf = 0.5 * (x - 5.0) ** 2
    assert access_free_energy(x, pmf, near_attack_max_A=3.6) is None
