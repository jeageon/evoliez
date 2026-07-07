"""ROADMAP_V5 E4a — O->Palpha umbrella / PMF access-barrier screen.

E2 showed unbiased explicit MD relaxes the reactive O_nuc->Palpha distance to ~5 A and never occupies
the productive ~3 A window. But the productive near-attack conformation (NAC) is a TRANSIENT reactive
configuration, not a stable minimum, so "does it sit at 3 A?" is the wrong observable. The right
question is the FREE-ENERGY COST to ACCESS the near-attack geometry, and that is measurable with the
existing classical fixed-charge FF via umbrella sampling along the O_nuc->Palpha distance + WHAM.

This module is the analysis + restraint core (pure-python WHAM + a thin OpenMM harmonic bias):
  - umbrella_windows()          window centres along the reaction coordinate
  - add_umbrella_bond_restraint()  harmonic bias 0.5*k*(r-r0)^2 on the O_nuc-Palpha bond (OpenMM)
  - wham()                      reconstruct the PMF from per-window distance samples (no pymbar dep)
  - access_free_energy()        the near-attack access cost = min PMF in <=window minus the global min

Claim discipline: this is SCREENING-LEVEL access-barrier evidence, NOT an activity/kcat claim. The
angle is never biased (only the distance), so the bias cannot manufacture the in-line NAC.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

# kB in kcal/mol/K (so free energies come out in kcal/mol)
_KB_KCAL = 0.0019872041


def umbrella_windows(d_min_A: float, d_max_A: float, n: int) -> List[float]:
    """Evenly spaced umbrella window centres (Angstrom) spanning the reaction coordinate, e.g.
    productive ~2.8 A -> relaxed ~5.5 A. n>=2."""
    if n < 2:
        raise ValueError("need >=2 umbrella windows")
    step = (d_max_A - d_min_A) / (n - 1)
    return [round(d_min_A + step * i, 4) for i in range(n)]


def add_umbrella_bond_restraint(system, atom_i: int, atom_j: int, d0_A: float, k_kcal: float):
    """Add a HARMONIC bias 0.5*k*(r-r0)^2 on the O_nuc(atom_i)-Palpha(atom_j) distance, centred at
    the window centre d0_A. DISTANCE ONLY -- the reactive angle is never restrained, so the bias
    cannot manufacture NAC. Returns the added force. (OpenMM; used per window on the server.)"""
    import openmm
    f = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    f.addGlobalParameter("k", float(k_kcal) * 418.4)     # kcal/mol/A^2 -> kJ/mol/nm^2
    f.addGlobalParameter("r0", float(d0_A) * 0.1)        # A -> nm
    f.addBond(int(atom_i), int(atom_j), [])
    system.addForce(f)
    return f


def wham(samples_by_window: Sequence[Sequence[float]], centers_A: Sequence[float], k_kcal: float,
         T_K: float = 300.0, n_bins: int = 60, tol: float = 1e-7, max_iter: int = 100000):
    """Iterative WHAM: reconstruct the unbiased PMF along the reaction coordinate from per-window
    biased samples. ``samples_by_window[w]`` = distances (A) sampled in window w (bias centre
    ``centers_A[w]``, harmonic force constant ``k_kcal`` in kcal/mol/A^2). Returns (bin_centers_A,
    pmf_kcal) with the PMF shifted so its global minimum is 0. Bins never sampled -> NaN in the PMF.
    No pymbar dependency."""
    import numpy as np
    kT = _KB_KCAL * T_K
    windows = [np.asarray(s, dtype=float) for s in samples_by_window]
    if not windows or all(len(s) == 0 for s in windows):
        raise ValueError("wham needs >=1 non-empty window of samples")
    alls = np.concatenate([s for s in windows if len(s)])
    lo, hi = float(alls.min()), float(alls.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    x = 0.5 * (edges[:-1] + edges[1:])                       # bin centres (A)
    n_i = np.zeros(n_bins)                                   # total counts per bin
    N_w = np.zeros(len(windows))
    for w, s in enumerate(windows):
        N_w[w] = len(s)
        if len(s):
            n_i += np.histogram(s, bins=edges)[0]
    centers = np.asarray(centers_A, dtype=float)
    # bias factor c[w,i] = exp(-beta * 0.5*k*(x_i - center_w)^2)
    c = np.exp(-(0.5 * k_kcal * (x[None, :] - centers[:, None]) ** 2) / kT)   # (W, M)
    f = np.ones(len(windows))
    P = np.where(n_i > 0, n_i, 1e-12)
    P = P / P.sum()
    for _ in range(max_iter):
        denom = ((N_w * f)[:, None] * c).sum(axis=0)         # (M,)
        P_new = np.where(denom > 0, n_i / denom, 0.0)
        s = P_new.sum()
        P_new = P_new / s if s > 0 else P_new
        f_new = 1.0 / np.clip((c * P_new[None, :]).sum(axis=1), 1e-300, None)
        f_new = f_new / f_new[0]                             # anchor window 0
        if np.max(np.abs(np.log(f_new) - np.log(f))) < tol:
            P, f = P_new, f_new
            break
        P, f = P_new, f_new
    pmf = -kT * np.log(np.where(P > 0, P, np.nan))
    pmf = pmf - np.nanmin(pmf)
    return x, pmf


def access_free_energy(bin_centers_A: Sequence[float], pmf_kcal: Sequence[float],
                       near_attack_max_A: float = 3.6) -> Optional[float]:
    """The near-attack ACCESS COST (kcal/mol): the lowest PMF value inside the productive window
    (distance <= near_attack_max_A) minus the global PMF minimum. Lower = the candidate reaches the
    near-attack geometry more cheaply. None if the near-attack window was never sampled."""
    import numpy as np
    x = np.asarray(bin_centers_A, dtype=float)
    pmf = np.asarray(pmf_kcal, dtype=float)
    mask = (x <= near_attack_max_A) & np.isfinite(pmf)
    if not mask.any():
        return None
    return float(np.min(pmf[mask]) - np.nanmin(pmf))


def window_overlap(samples_by_window: Sequence[Sequence[float]], n_bins: int = 40) -> dict:
    """QC: Bhattacharyya overlap between ADJACENT windows' distance histograms. WHAM is only
    trustworthy when adjacent windows overlap; low overlap => the PMF is numbers, not free energy.
    Returns {mean_overlap, min_overlap, sufficient} (sufficient = every adjacent pair overlaps >=0.1)."""
    import numpy as np
    ws = [np.asarray(s, dtype=float) for s in samples_by_window]
    alls = np.concatenate([s for s in ws if len(s)]) if any(len(s) for s in ws) else np.array([0.0])
    lo, hi = float(alls.min()), float(alls.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, n_bins + 1)
    hists = []
    for s in ws:
        h = np.histogram(s, bins=edges)[0].astype(float) if len(s) else np.zeros(n_bins)
        hists.append(h / h.sum() if h.sum() > 0 else h)
    bc = [float(np.sum(np.sqrt(hists[i] * hists[i + 1]))) for i in range(len(hists) - 1)]
    if not bc:
        return {"mean_overlap": None, "min_overlap": None, "sufficient": False}
    return {"mean_overlap": round(float(np.mean(bc)), 3), "min_overlap": round(float(min(bc)), 3),
            "sufficient": bool(min(bc) >= 0.10)}


def near_attack_angle_occupancy(dist_by_window: Sequence[Sequence[float]],
                                angle_by_window: Sequence[Sequence[float]],
                                near_attack_max_A: float = 3.6,
                                angle_min_deg: float = 150.0):
    """Post-hoc separation of ACCESS from NAC: of the frames whose O_nuc-Palpha DISTANCE reached the
    near-attack window (<= near_attack_max_A), the fraction that ALSO have an in-line angle
    (>= angle_min_deg). The umbrella biases DISTANCE only, so a low-distance state is not necessarily
    the in-line NAC -- this is the guard against calling a bent short-distance contact 'reactive'.
    Returns (angle_occupancy_fraction, n_near_attack_frames); (None, 0) if the window was never reached."""
    import numpy as np
    d = np.concatenate([np.asarray(x, dtype=float) for x in dist_by_window])
    a = np.concatenate([np.asarray(x, dtype=float) for x in angle_by_window])
    m = np.isfinite(d) & np.isfinite(a) & (d <= near_attack_max_A)
    n = int(m.sum())
    if n == 0:
        return None, 0
    return float((a[m] >= angle_min_deg).mean()), n
