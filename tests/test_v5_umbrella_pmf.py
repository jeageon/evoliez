"""ROADMAP_V5 E4a — umbrella/PMF access-barrier analysis. The per-window MD is server-only (OpenMM),
but the WHAM reconstruction + access-barrier extraction are pure-python and MUST be correct: this
verifies WHAM recovers a KNOWN harmonic PMF from synthetic harmonically-biased samples."""
import json

import numpy as np
import pytest

from evoliez.md.umbrella import (access_free_energy, near_attack_angle_occupancy,
                                 umbrella_windows, wham, window_overlap)


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


def test_window_overlap_qc():
    rng = np.random.default_rng(1)
    # tightly spaced windows (0.2 A apart, std ~0.22) -> good overlap
    good = [rng.normal(c, 0.22, 2000) for c in umbrella_windows(3.0, 4.0, 6)]
    assert window_overlap(good)["sufficient"] is True
    # far-apart windows (no overlap) -> insufficient
    bad = [rng.normal(3.0, 0.05, 2000), rng.normal(5.0, 0.05, 2000)]
    assert window_overlap(bad)["sufficient"] is False


def test_near_attack_angle_occupancy_separates_access_from_nac():
    # window A: reaches <=3.6 A but BENT (angle ~100) ; window B: reaches <=3.6 A and IN-LINE (~170)
    dist = [np.full(1000, 3.2), np.full(1000, 3.3)]
    ang = [np.full(1000, 100.0), np.full(1000, 170.0)]
    frac, n = near_attack_angle_occupancy(dist, ang, near_attack_max_A=3.6, angle_min_deg=150.0)
    assert n == 2000 and abs(frac - 0.5) < 1e-6          # half the near-attack frames are in-line
    # a purely bent short contact -> 0 in-line NAC even though the distance is reached
    frac2, _ = near_attack_angle_occupancy([np.full(500, 3.0)], [np.full(500, 90.0)])
    assert frac2 == 0.0


def test_near_attack_angle_occupancy_none_when_unreached():
    frac, n = near_attack_angle_occupancy([np.full(100, 5.0)], [np.full(100, 170.0)])
    assert frac is None and n == 0


def _load_driver():
    """Load scripts/e4a_umbrella_pmf.py (not a package) as a module."""
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "e4a_umbrella_pmf.py"
    spec = importlib.util.spec_from_file_location("e4a_driver", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_analyse_candidate_reads_umbrella_samples(tmp_path):
    """End-to-end sampling->analysis contract WITHOUT OpenMM/mdtraj: synthesise the exact
    umbrella_samples.json files s10 writes (<run>/md/<cid>/e4a/window_<d0>/), then WHAM them via the
    driver. Samples are the product-of-Gaussians the true well 0.5*K*(d-d_eq)^2 + harmonic bias give,
    same construction as test_wham_recovers_known_harmonic_pmf; angles in-line (>=150) at short d."""
    drv = _load_driver()
    rng = np.random.default_rng(7)
    K_true, d_eq, k_bias, T = 2.0, 3.4, 10.0, 300.0
    kT = 0.0019872041 * T
    e4a = tmp_path / "md" / "mut_00042" / "e4a"
    for d0 in umbrella_windows(2.8, 5.0, 12):
        Ktot = K_true + k_bias
        mean = (K_true * d_eq + k_bias * d0) / Ktot
        dists = rng.normal(mean, np.sqrt(kT / Ktot), 800)
        # in-line when close (<=3.6), progressively bent as it moves out -> nontrivial angle occupancy
        angs = np.where(dists <= 3.6, rng.normal(160, 5, 800), rng.normal(120, 15, 800))
        wd = e4a / ("window_%.2f" % d0)
        wd.mkdir(parents=True)
        (wd / "umbrella_samples.json").write_text(json.dumps({
            "window_A": d0, "k_kcal": k_bias, "distances_A": dists.tolist(),
            "angles_deg": angs.tolist(), "atoms": {"donor_heavy": 1, "acceptor": 2, "leaving": 3},
            "n_frames": len(dists)}))
    row = drv.analyse_candidate(str(e4a), k_bias, 3.6, 150.0)
    assert row["cid"] == "mut_00042"
    assert "error" not in row and row["n_windows"] == 12
    assert row["converged"] is True and row["overlap_sufficient"] is True
    # well minimum at 3.4 A is INSIDE the <=3.6 window -> access cost is small (near 0)
    assert row["access_cost_kcal"] is not None and row["access_cost_kcal"] < 1.0
    # in-line NAC occupancy is reported SEPARATELY and is a real fraction in (0,1)
    assert row["near_attack_angle_occ"] is not None and 0.0 < row["near_attack_angle_occ"] <= 1.0


def test_driver_analyse_candidate_needs_two_windows(tmp_path):
    drv = _load_driver()
    e4a = tmp_path / "md" / "mut_00001" / "e4a"
    wd = e4a / "window_3.00"
    wd.mkdir(parents=True)
    (wd / "umbrella_samples.json").write_text(json.dumps({
        "window_A": 3.0, "k_kcal": 10.0, "distances_A": [3.0] * 50, "angles_deg": [160.0] * 50}))
    row = drv.analyse_candidate(str(e4a), 10.0, 3.6, 150.0)
    assert "error" in row and "mut_00001" == row["cid"]
