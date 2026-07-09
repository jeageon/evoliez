"""V6-3 — Amber-native PMF / umbrella-sampling tier.

The V6-2 executor showed unbiased explicit MD never occupies the productive
≤3.6 Å O_nuc→Pα window (near-attack is transient). The right observable is the
FREE-ENERGY COST to ACCESS that geometry — an Amber `pmemd.cuda` umbrella sweep
along O_nuc→Pα + WHAM.

This module is the Amber restraint + orchestration layer; the WHAM / QC / access
core is the ENGINE-AGNOSTIC ``md.umbrella`` reused verbatim (so the OpenMM E4a and
the Amber tiers report identical quantities). Windows are fanned out as INDEPENDENT
GPU jobs (ROADMAP_V6 §2.2).

Restraint convention (critical): Amber's ``&rst`` energy is ``rk2·(r-r2)²``, but
``md.umbrella.wham`` expects the ``0.5·k·(r-r0)²`` convention — so the Amber force
constant is ``rk = 0.5·k_kcal`` and the SAME ``k_kcal`` is passed to WHAM.

Claim discipline (ROADMAP_V6 §6): the umbrella biases DISTANCE only (never the
angle), distance access cost is reported SEPARATELY from in-line NAC occupancy, and
a non-converged PMF is classified ``diagnostic_only`` and PROHIBITED from ranking.

Subprocess (pmemd/cpptraj) is server-only; the restraint/mdin/QC/classification
helpers are unit-testable off the server.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evoliez.adapters.amber_engine import (
    _pmemd_cuda, mdin_min_exp, mdin_heat_exp, mdin_npt_equil,
)
from evoliez.md.umbrella import (
    access_free_energy, near_attack_angle_occupancy, umbrella_windows, wham,
    window_overlap,
)

log = logging.getLogger(__name__)

_CLAIM_CONVERGED = ("screening-level near-attack ACCESS-COST evidence "
                    "(converged PMF; NOT activity/kcat/activation-barrier)")
_CLAIM_DIAGNOSTIC = ("DIAGNOSTIC-ONLY non-converged PMF — prohibited from ranking "
                     "candidates (NOT activity/kcat/access-cost)")


# --------------------------------------------------------------------------- #
# specs / results
# --------------------------------------------------------------------------- #
@dataclass
class PmfSpec:
    windows_A: List[float]                 # umbrella window centres (Å)
    k_kcal: float                          # force constant, 0.5·k convention (-> wham)
    dist_i: int                            # O_nuc Amber atom index (1-based)
    dist_j: int                            # Pα Amber atom index (1-based)
    dist_masks: Tuple[str, str]            # cpptraj masks for the distance (O_nuc, Pα)
    angle_masks: Optional[Tuple[str, str, str]] = None   # O_nuc, Pα, O_leaving
    production_ns: float = 0.1
    equil_ps: float = 50.0                 # shared prep equilibration
    equil_discard_frac: float = 0.2        # per-window equilibration discard
    near_attack_A: float = 3.6
    angle_min_deg: float = 150.0
    T_K: float = 300.0
    timestep_fs: float = 2.0
    solvent: str = "explicit"


@dataclass
class PmfResult:
    verdict: str                           # converged | diagnostic_only | failed
    classification: str                    # screening_prioritization | diagnostic_only
    access_cost_kcal: Optional[float]
    pmf: dict                              # {bin_centers_A, pmf_kcal}
    qc: dict
    claim_ceiling: str
    windows: List[dict] = field(default_factory=list)
    failure_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# pure helpers (unit-testable)
# --------------------------------------------------------------------------- #
def amber_umbrella_restraint(iat_i: int, iat_j: int, r0_A: float, k_kcal: float) -> str:
    """Amber ``&rst`` DISANG text for a HARMONIC O_nuc→Pα distance restraint centred
    at ``r0_A``. Amber energy = rk·(r-r0)² with rk = 0.5·k_kcal (so the well matches
    the 0.5·k·(r-r0)² convention md.umbrella.wham uses). r1/r4 are wide so the
    sampled distance is always in the harmonic region (never the linear tails)."""
    rk = 0.5 * float(k_kcal)
    return (f" &rst\n  iat={int(iat_i)},{int(iat_j)},\n"
            f"  r1=0.01, r2={r0_A:.4f}, r3={r0_A:.4f}, r4=30.0,\n"
            f"  rk2={rk:.5f}, rk3={rk:.5f},\n /\n")


def mdin_umbrella_prod(nsteps: int, nframes: int, dumpfreq: int,
                       disang: str = "umbrella.rst",
                       dumpave: str = "umbrella_dist.dat") -> str:
    """NPT production with the distance restraint active (nmropt=1), the reaction
    coordinate dumped every ``dumpfreq`` steps."""
    interval = max(1, nsteps // max(1, nframes))
    return ("umbrella production, O_nuc->Pa restrained (explicit PME)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt=0.002, irest=1, ntx=5,\n"
            " ntb=2, ntp=1, barostat=2, pres0=1.0, taup=2.0, cut=10.0, iwrap=1,\n"
            " ntc=2, ntf=2, ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,\n"
            f" ntpr={interval}, ntwx={interval}, ntwr={nsteps},\n nmropt=1,\n/\n"
            f"&wt type='DUMPFREQ', istep1={dumpfreq} /\n&wt type='END' /\n"
            f"DISANG={disang}\nDUMPAVE={dumpave}\n")


def parse_dumpave(text: str) -> List[float]:
    """Reaction-coordinate values from an Amber DUMPAVE file (col 2 = restrained
    distance; col 1 = step)."""
    out: List[float] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(("#", "@")):
            continue
        f = ln.split()
        if len(f) >= 2:
            try:
                out.append(float(f[1]))
            except ValueError:
                pass
    return out


def discard_equilibration(samples: Sequence[float], frac: float) -> List[float]:
    n = len(samples)
    k = int(n * max(0.0, min(0.9, frac)))
    return list(samples[k:]) if n - k >= 1 else list(samples)


def convergence_check(samples_by_window: Sequence[Sequence[float]],
                      centers_A: Sequence[float], k_kcal: float,
                      near_attack_A: float, T_K: float,
                      tol_kcal: float = 1.0) -> dict:
    """Split each window's samples first-half / second-half, WHAM each, and compare
    the near-attack access cost. Small change => the PMF is settling (converged);
    large change => it is still drifting (diagnostic only). Also returns whether every
    window had enough samples."""
    def halves(s):
        m = len(s) // 2
        return s[:m], s[m:]

    firsts, seconds = [], []
    min_n = min((len(s) for s in samples_by_window), default=0)
    for s in samples_by_window:
        a, b = halves(list(s))
        firsts.append(a)
        seconds.append(b)
    out = {"min_window_samples": min_n, "tol_kcal": tol_kcal}
    try:
        xf, pf = wham(firsts, centers_A, k_kcal, T_K=T_K)
        xs, ps = wham(seconds, centers_A, k_kcal, T_K=T_K)
        af = access_free_energy(xf, pf, near_attack_A)
        as_ = access_free_energy(xs, ps, near_attack_A)
        out["access_first_half_kcal"] = round(af, 3) if af is not None else None
        out["access_second_half_kcal"] = round(as_, 3) if as_ is not None else None
        if af is not None and as_ is not None:
            out["access_drift_kcal"] = round(abs(af - as_), 3)
            out["halves_consistent"] = bool(abs(af - as_) <= tol_kcal)
        else:
            out["access_drift_kcal"] = None
            out["halves_consistent"] = False
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:160]
        out["halves_consistent"] = False
    return out


def classify_pmf(overlap_qc: dict, conv_qc: dict) -> Tuple[str, str, str]:
    """Decide the PMF verdict. Converged = adjacent-window overlap sufficient AND the
    first/second-half access cost agree. Otherwise diagnostic-only (ranking
    prohibited). Returns (verdict, classification, claim_ceiling)."""
    overlap_ok = bool(overlap_qc.get("sufficient"))
    halves_ok = bool(conv_qc.get("halves_consistent"))
    if overlap_ok and halves_ok:
        return "converged", "screening_prioritization", _CLAIM_CONVERGED
    return "diagnostic_only", "diagnostic_only", _CLAIM_DIAGNOSTIC


# --------------------------------------------------------------------------- #
# orchestration (server-only)
# --------------------------------------------------------------------------- #
def _sh(cmd: List[str], cwd: Path, timeout: int, gpu: Optional[int] = None):
    from evoliez.utils.subprocess_utils import _LIBC, _set_pdeathsig
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
                          timeout=timeout,
                          preexec_fn=_set_pdeathsig if _LIBC is not None else None)


def _prep_equilibrated_start(workdir: Path, spec: PmfSpec, pmemd: str,
                             gpu: int) -> Path:
    """min → heat → equil (unbiased) → shared start restart for all windows. Skipped
    if ``shared_equil.rst`` already exists (restart-safe)."""
    out = workdir / "shared_equil.rst"
    if out.exists() and out.stat().st_size > 0:
        return out
    P = "complex.prmtop"
    equil_steps = max(1000, int(spec.equil_ps * 1000 / max(0.1, spec.timestep_fs)))
    (workdir / "pmf_min.in").write_text(mdin_min_exp())
    _run(pmemd, ["-O", "-i", "pmf_min.in", "-o", "pmf_min.out", "-p", P,
                 "-c", "complex.inpcrd", "-r", "pmf_min.rst", "-ref", "complex.inpcrd"],
         workdir, 3600, gpu, "pmf_min")
    (workdir / "pmf_heat.in").write_text(mdin_heat_exp(10000))
    _run(pmemd, ["-O", "-i", "pmf_heat.in", "-o", "pmf_heat.out", "-p", P,
                 "-c", "pmf_min.rst", "-r", "pmf_heat.rst", "-ref", "pmf_min.rst"],
         workdir, 3600, gpu, "pmf_heat")
    (workdir / "pmf_equil.in").write_text(mdin_npt_equil(equil_steps))
    _run(pmemd, ["-O", "-i", "pmf_equil.in", "-o", "pmf_equil.out", "-p", P,
                 "-c", "pmf_heat.rst", "-r", str(out.name), "-ref", "pmf_heat.rst"],
         workdir, 21600, gpu, "pmf_equil")
    return out


def _run(pmemd, args, workdir, timeout, gpu, tag):
    r = _sh([pmemd] + args, workdir, timeout, gpu)
    out = workdir / f"{tag}.out"
    ok = out.exists() and any(m in out.read_text(errors="replace")
                              for m in ("Final Performance", "TIMINGS",
                                        "Master Total wall time"))
    if r.returncode != 0 or not ok:
        raise RuntimeError(f"{tag} rc={r.returncode}: "
                           f"{((out.read_text(errors='replace') if out.exists() else '') + r.stderr)[-700:]}")


def _run_window(workdir: Path, spec: PmfSpec, pmemd: str, gpu: int,
                start_rst: Path, w_idx: int, r0: float) -> Tuple[List[float], List[float]]:
    """One biased window: restrained production from the shared start, then extract
    the O_nuc→Pα distance + in-line angle per frame. Returns (distances, angles)."""
    wd = workdir / f"win_{r0:.3f}"
    wd.mkdir(exist_ok=True)
    P = "../complex.prmtop"
    nsteps = max(1, int(spec.production_ns * 1e6 / max(0.1, spec.timestep_fs)))
    nframes = min(500, max(20, nsteps // 20))
    dumpfreq = max(1, nsteps // nframes)
    (wd / "umbrella.rst").write_text(
        amber_umbrella_restraint(spec.dist_i, spec.dist_j, r0, spec.k_kcal))
    nc = f"prod.nc"
    if not ((wd / nc).exists() and (wd / nc).stat().st_size > 0):
        (wd / "prod.in").write_text(mdin_umbrella_prod(nsteps, nframes, dumpfreq))
        _run(pmemd, ["-O", "-i", "prod.in", "-o", "prod.out", "-p", P,
                     "-c", str(start_rst), "-r", "prod.rst", "-x", nc,
                     "-ref", str(start_rst)], wd, 172800, gpu, "prod")
    # cpptraj: distance + angle per frame (angle read-only)
    om, pm = spec.dist_masks
    lines = [f"parm ../complex.prmtop", f"trajin {nc}", "autoimage",
             f"distance d {om} {pm} out dist.dat"]
    if spec.angle_masks:
        a, b, c = spec.angle_masks
        lines.append(f"angle ang {a} {b} {c} out angle.dat")
    lines += ["run", "quit"]
    (wd / "extract.in").write_text("\n".join(lines) + "\n")
    _sh(["cpptraj", "-i", "extract.in"], wd, 1800)
    dist = _col2(wd / "dist.dat")
    ang = _col2(wd / "angle.dat") if spec.angle_masks else []
    dist = discard_equilibration(dist, spec.equil_discard_frac)
    if ang:
        k = len(ang) - len(dist)
        ang = ang[k:] if k > 0 else ang
    return dist, ang


def _col2(p: Path) -> List[float]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith(("#", "@")):
            f = ln.split()
            if len(f) >= 2:
                try:
                    out.append(float(f[1]))
                except ValueError:
                    pass
    return out


def run_pmf(system_workdir: Path, spec: PmfSpec, gpu_pool: Sequence[int],
            *, pmemd: Optional[str] = None,
            jsonl_path: Optional[Path] = None) -> PmfResult:
    """Full Amber PMF: shared equilibration, biased window fan-out (one pmemd per
    GPU), WHAM + QC + convergence classification."""
    system_workdir = Path(system_workdir)
    pmemd = pmemd or _pmemd_cuda()
    if not pmemd:
        return PmfResult("failed", "diagnostic_only", None, {}, {}, _CLAIM_DIAGNOSTIC,
                         failure_reason="pmemd.cuda not found")
    pool = list(gpu_pool) or [0]
    # shared equilibrated start (on the first GPU)
    try:
        start = _prep_equilibrated_start(system_workdir, spec, pmemd, pool[0])
    except Exception as exc:  # noqa: BLE001
        return PmfResult("failed", "diagnostic_only", None, {}, {}, _CLAIM_DIAGNOSTIC,
                         failure_reason=f"equilibration prep: {exc}")

    gpu_q: "queue.Queue[int]" = queue.Queue()
    for g in pool:
        gpu_q.put(g)
    lock = threading.Lock()
    per_window: Dict[int, Tuple[List[float], List[float]]] = {}

    def worker(item):
        w_idx, r0 = item
        gpu = gpu_q.get()
        try:
            d, a = _run_window(system_workdir, spec, pmemd, gpu, start, w_idx, r0)
            with lock:
                per_window[w_idx] = (d, a)
        except Exception as exc:  # noqa: BLE001
            log.warning("PMF window %.3f failed: %s", r0, exc)
            with lock:
                per_window[w_idx] = ([], [])
        finally:
            gpu_q.put(gpu)

    with ThreadPoolExecutor(max_workers=len(pool)) as ex:
        list(ex.map(worker, list(enumerate(spec.windows_A))))

    dist_by_window = [per_window.get(i, ([], []))[0] for i in range(len(spec.windows_A))]
    angle_by_window = [per_window.get(i, ([], []))[1] for i in range(len(spec.windows_A))]
    non_empty = sum(1 for d in dist_by_window if d)
    if non_empty < 2:
        return PmfResult("failed", "diagnostic_only", None, {},
                         {"windows_with_samples": non_empty}, _CLAIM_DIAGNOSTIC,
                         windows=_win_records(spec, dist_by_window),
                         failure_reason="fewer than 2 windows produced samples")

    overlap = window_overlap(dist_by_window)
    conv = convergence_check(dist_by_window, spec.windows_A, spec.k_kcal,
                             spec.near_attack_A, spec.T_K)
    angle_occ, n_na = near_attack_angle_occupancy(
        dist_by_window, angle_by_window, spec.near_attack_A, spec.angle_min_deg)
    try:
        x, pmf = wham(dist_by_window, spec.windows_A, spec.k_kcal, T_K=spec.T_K)
        access = access_free_energy(x, pmf, spec.near_attack_A)
        pmf_curve = {"bin_centers_A": [round(float(v), 4) for v in x],
                     "pmf_kcal": [None if v != v else round(float(v), 4) for v in pmf]}
    except Exception as exc:  # noqa: BLE001
        return PmfResult("failed", "diagnostic_only", None, {},
                         {"overlap": overlap, "convergence": conv},
                         _CLAIM_DIAGNOSTIC,
                         windows=_win_records(spec, dist_by_window),
                         failure_reason=f"WHAM: {exc}")

    verdict, classification, ceiling = classify_pmf(overlap, conv)
    qc = {
        "overlap": overlap,
        "convergence": conv,
        "windows_with_samples": non_empty,
        "n_windows": len(spec.windows_A),
        # access distance vs in-line NAC kept SEPARATE (ROADMAP_V6 §6)
        "near_attack_angle_occupancy": (round(angle_occ, 3) if angle_occ is not None
                                        else None),
        "near_attack_frames": n_na,
        "equilibration_discard_frac": spec.equil_discard_frac,
    }
    # a converged PMF still cannot claim access cost if near-attack was never reached
    access_out = access if (access is not None and verdict == "converged") else access
    result = PmfResult(
        verdict=verdict, classification=classification,
        access_cost_kcal=(round(access, 3) if access is not None else None),
        pmf=pmf_curve, qc=qc, claim_ceiling=ceiling,
        windows=_win_records(spec, dist_by_window))
    if jsonl_path:
        Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(jsonl_path).open("a") as fh:
            fh.write(json.dumps({
                "schema": "amber_pmf/v1", "verdict": verdict,
                "classification": classification,
                "access_cost_kcal": result.access_cost_kcal, "qc": qc,
                "claim_ceiling": ceiling}) + "\n")
    return result


def _win_records(spec: PmfSpec, dist_by_window) -> List[dict]:
    return [{"center_A": round(spec.windows_A[i], 4), "n_samples": len(d),
             "mean_A": round(sum(d) / len(d), 3) if d else None}
            for i, d in enumerate(dist_by_window)]
