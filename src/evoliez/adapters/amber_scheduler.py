"""V6-2 — Amber GPU MD scheduler + reactive-geometry analysis.

ROADMAP_V6 §2.2: on the shared multi-GPU server, prefer ONE INDEPENDENT pmemd job
per GPU (candidates / replicas / PMF windows as separate GPU jobs) over one
multi-GPU job. This scheduler fans a list of built Amber systems (from
``amber_builder``) out across a GPU pool: each job runs its staged
``pmemd.cuda`` protocol pinned to a single GPU, is restart-safe (skips stages
whose restart already exists), and appends a provenance record to a JSONL.

CPU fallback is BLOCKED unless explicitly requested: we exec the ``.cuda`` binary
and require a GPU assignment, so a mis-configured launch fails loudly rather than
silently running 100× slower on CPU (ROADMAP_V6 §2.1).

The per-job analysis reuses the V6-1 ``amber_system_manifest`` reactive masks to
recompute the reaction-geometry observables directly from the trajectory: the
O_nuc→Pα access distance, the in-line O_nuc–Pα–O_leaving angle, Mg bridge
retention, ligand/pocket RMSD, and energy drift — the EvidenceCard/ClaimGuard
inputs. Distance access cost is reported SEPARATELY from in-line occupancy
(ROADMAP_V6 §6), and everything is capped at screening-level claims.

Subprocess (pmemd/cpptraj) is server-only; the pure planning/parse helpers are
unit-testable off the server.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from evoliez.adapters.amber_engine import (
    _pmemd_cuda, amber_available, mdin_heat_exp, mdin_min_exp, mdin_npt_equil,
    mdin_prod_exp, mdin_min, mdin_heat, mdin_prod,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# job / result
# --------------------------------------------------------------------------- #
@dataclass
class AmberMDJob:
    job_id: str
    workdir: Path                       # holds complex.prmtop / complex.inpcrd
    manifest: dict = field(default_factory=dict)   # amber_system_manifest (reactive masks)
    solvent: str = "explicit"           # explicit | implicit
    production_ns: float = 0.2
    equilibration_ps: float = 100.0
    replicas: int = 1
    timestep_fs: float = 2.0
    prmtop: str = "complex.prmtop"
    inpcrd: str = "complex.inpcrd"


@dataclass
class AmberMDJobResult:
    job_id: str
    status: str                         # ok | unstable | failed | skipped_no_gpu
    gpu: Optional[int]
    stages_done: List[str]
    trajectories: List[str]
    analysis: dict
    simulation_ns: float
    wallclock_s: float
    failure_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# pure planning helpers (unit-testable)
# --------------------------------------------------------------------------- #
def production_nsteps(production_ns: float, timestep_fs: float) -> int:
    return max(1, int(round(production_ns * 1e6 / max(0.1, timestep_fs))))


# Robust 2-stage explicit minimization. Freshly-built systems (transplanted
# ligands, crude mutant side-chain swaps) carry severe initial clashes that a
# single CG minimization overshoots into a numerical blow-up (OpenMM's L-BFGS
# absorbs them; pmemd CG does not). min1 restrains the backbone strongly and
# relaxes everything else with a tiny initial step (SD-heavy); min2 relaxes
# broadly. ntc=1/ntf=1 so H can move freely during minimization.
def _mdin_min1_exp(maxcyc: int = 6000, ncyc: int = 3000,
                   restraint_wt: float = 10.0) -> str:
    return ("min1 backbone-restrained, clash relaxation (explicit PME)\n&cntrl\n"
            f" imin=1, maxcyc={maxcyc}, ncyc={ncyc}, ntmin=1, dx0=0.001,\n"
            " ntb=1, cut=10.0, ntc=1, ntf=1,\n"
            f" ntr=1, restraintmask='@CA,C,N,O', restraint_wt={restraint_wt},\n"
            " ntpr=500,\n/\n")


def _mdin_min2_exp(maxcyc: int = 6000, ncyc: int = 2000,
                   restraint_wt: float = 2.0) -> str:
    return ("min2 weak backbone restraint (explicit PME)\n&cntrl\n"
            f" imin=1, maxcyc={maxcyc}, ncyc={ncyc}, ntmin=1, dx0=0.01,\n"
            " ntb=1, cut=10.0, ntc=1, ntf=1,\n"
            f" ntr=1, restraintmask='@CA,C,N,O', restraint_wt={restraint_wt},\n"
            " ntpr=500,\n/\n")


def stage_plan(solvent: str) -> List[str]:
    """Ordered restart-producing stages for a protocol."""
    return (["min", "heat", "equil", "prod"] if solvent == "explicit"
            else ["min", "heat", "prod"])


def stages_remaining(workdir: Path, solvent: str, replicas: int) -> List[str]:
    """Which stages still need running (restart-safe resume). A stage is done when
    its restart file exists and is non-empty; prod is done when every replica .nc
    exists."""
    todo: List[str] = []
    for st in stage_plan(solvent):
        if st == "prod":
            have = all((workdir / f"prod_{r}.nc").exists()
                       and (workdir / f"prod_{r}.nc").stat().st_size > 0
                       for r in range(max(1, replicas)))
            if not have:
                todo.append(st)
        else:
            rst = workdir / f"{st}.rst"
            if not (rst.exists() and rst.stat().st_size > 0):
                todo.append(st)
    return todo


def reactive_masks_from_manifest(manifest: dict) -> Dict[str, str]:
    """Extract {role: mask} from a V6-1 amber_system_manifest. Roles: O_nuc,
    P_alpha, O_leaving, metal, and catalytic_<n>."""
    out: Dict[str, str] = {}
    for a in manifest.get("reactive_atoms", []):
        if not a.get("mask"):
            continue
        role = a["role"]
        key = role if role != "catalytic" else f"catalytic_{a.get('orig_resid') or a['resid']}"
        out[key] = a["mask"]
    return out


def reactive_cpptraj_script(masks: Dict[str, str], solvent: str,
                            trajs: Sequence[str]) -> str:
    """cpptraj deck computing the reaction-geometry observables from the built
    system's reactive masks. Distance access (O_nuc→Pα) is a separate line from the
    in-line angle so the two are never conflated (ROADMAP_V6 §6)."""
    L = [f"parm complex.prmtop"] + [f"trajin {t}" for t in trajs]
    if solvent == "explicit":
        L += ["autoimage", "strip :WAT,Na+,Cl-,K+"]
    onuc, pa, olv, mg = (masks.get("O_nuc"), masks.get("P_alpha"),
                         masks.get("O_leaving"), masks.get("metal"))
    if onuc and pa:
        L.append(f"distance d_onuc_pa {onuc} {pa} out reactive.dat")   # access distance
    if onuc and pa and olv:
        L.append(f"angle a_inline {onuc} {pa} {olv} out reactive.dat")  # in-line angle
    if mg and onuc:
        L.append(f"distance d_mg_onuc {mg} {onuc} out reactive.dat")    # Mg bridge
    if mg and pa:
        L.append(f"distance d_mg_pa {mg} {pa} out reactive.dat")
    # ligand + backbone RMSD (retention / structural stability)
    L += ["rms fit @CA,C,N first",
          "rms lig_rmsd :LIG&!@H= first nofit out lig_rmsd.dat",
          "rms bb_rmsd @CA,C,N first nofit out bb_rmsd.dat",
          "run", "quit"]
    return "\n".join(L) + "\n"


def _col(dat_text: str, header: str) -> List[float]:
    """Column named ``header`` from a cpptraj multi-column .dat (row 0 = #Frame ...)."""
    lines = [l for l in dat_text.splitlines() if l.strip()]
    if not lines:
        return []
    hdr = lines[0].lstrip("#").split()
    try:
        ci = hdr.index(header)
    except ValueError:
        return []
    out: List[float] = []
    for ln in lines[1:]:
        f = ln.split()
        if len(f) > ci:
            try:
                out.append(float(f[ci]))
            except ValueError:
                pass
    return out


def summarize_reactive(reactive_dat: str, lig_rmsd: List[float], bb_rmsd: List[float],
                       masks: Dict[str, str], *, near_attack_A: float = 3.6,
                       angle_min: float = 150.0, mg_coord_A: float = 2.8,
                       lig_retain_A: float = 5.0) -> dict:
    """Reduce the cpptraj series to EvidenceCard-ready observables. Access distance
    and in-line occupancy are reported SEPARATELY; nothing here is an activity claim."""
    d = _col(reactive_dat, "d_onuc_pa")
    ang = _col(reactive_dat, "a_inline")
    mg_o = _col(reactive_dat, "d_mg_onuc")
    mg_p = _col(reactive_dat, "d_mg_pa")
    n = max(len(d), len(ang), 1)

    def occ(series, pred):
        return round(sum(1 for v in series if pred(v)) / len(series), 3) if series else None

    out = {
        "n_frames": n,
        "access_distance": {
            "series_min_A": round(min(d), 3) if d else None,
            "series_mean_A": round(sum(d) / len(d), 3) if d else None,
            "near_attack_A": near_attack_A,
            # fraction of frames the access distance sits at/below near-attack
            "near_attack_occupancy": occ(d, lambda v: v <= near_attack_A),
        },
        "inline_angle": {
            "series_mean_deg": round(sum(ang) / len(ang), 2) if ang else None,
            "angle_min_deg": angle_min,
            "productive_angle_occupancy": occ(ang, lambda v: v >= angle_min),
        },
        "ligand_retention": occ(lig_rmsd, lambda v: v <= lig_retain_A),
        "ligand_rmsd_final_A": round(lig_rmsd[-1], 3) if lig_rmsd else None,
        "backbone_rmsd_final_A": round(bb_rmsd[-1], 3) if bb_rmsd else None,
    }
    if masks.get("metal"):
        # Mg "bridge retained" = within coordination distance of BOTH anions
        both = [1 for a, b in zip(mg_o, mg_p) if a <= mg_coord_A and b <= mg_coord_A]
        out["mg_retention"] = {
            "mg_onuc_mean_A": round(sum(mg_o) / len(mg_o), 3) if mg_o else None,
            "mg_pa_mean_A": round(sum(mg_p) / len(mg_p), 3) if mg_p else None,
            "bridge_occupancy": round(len(both) / len(mg_o), 3) if mg_o else None,
        }
    return out


# --------------------------------------------------------------------------- #
# scheduler
# --------------------------------------------------------------------------- #
class AmberGpuScheduler:
    """Fan out independent pmemd.cuda MD jobs across a GPU pool (one job per GPU at
    a time). Restart-safe; CPU fallback blocked unless ``allow_cpu``."""

    def __init__(self, gpu_pool: Sequence[int], *, pmemd: Optional[str] = None,
                 allow_cpu: bool = False,
                 jsonl_path: Optional[Path] = None):
        self.gpu_pool = list(gpu_pool)
        self.pmemd = pmemd or _pmemd_cuda()
        self.allow_cpu = allow_cpu
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._jsonl_lock = threading.Lock()

    # -- provenance --------------------------------------------------------- #
    def _record(self, rec: dict) -> None:
        if not self.jsonl_path:
            return
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl_lock:
            with self.jsonl_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    # -- one pmemd stage ---------------------------------------------------- #
    def _pmemd(self, job: AmberMDJob, gpu: int, args: List[str], tag: str,
               timeout: int) -> None:
        from evoliez.utils.subprocess_utils import _LIBC, _set_pdeathsig
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        r = subprocess.run([self.pmemd] + args, cwd=str(job.workdir), env=env,
                           capture_output=True, text=True, timeout=timeout,
                           preexec_fn=_set_pdeathsig if _LIBC is not None else None)
        out = (job.workdir / f"{tag}.out")
        combined = (out.read_text(errors="replace") if out.exists() else "") \
            + "\n" + r.stdout + "\n" + r.stderr
        if r.returncode != 0 or not _stage_ok(out):
            raise RuntimeError(f"{tag} rc={r.returncode}: {combined[-800:]}")
        # positive GPU-execution proof (no silent CPU fallback), on the first stage
        if tag.startswith("min") and "CUDA Device Name" not in combined:
            raise RuntimeError(f"{tag}: pmemd printed no GPU DEVICE INFO — refusing "
                               "to accept a possible CPU/mis-pinned run")

    # -- one full job ------------------------------------------------------- #
    def _run_job(self, job: AmberMDJob, gpu: int) -> AmberMDJobResult:
        t0 = time.monotonic()
        wd = job.workdir
        P = job.prmtop
        nsteps = production_nsteps(job.production_ns, job.timestep_fs)
        heat_steps = 10000
        equil_steps = max(1000, int(job.equilibration_ps * 1000 / max(0.1, job.timestep_fs)))
        prod_frames = min(250, max(10, nsteps // 50))
        todo = stages_remaining(wd, job.solvent, job.replicas)
        done: List[str] = [s for s in stage_plan(job.solvent) if s not in todo]
        trajs: List[str] = []
        try:
            if job.solvent == "explicit":
                if "min" in todo:
                    # robust 2-stage min (min1 clash relaxation -> min2), final = min.rst
                    (wd / "min1.in").write_text(_mdin_min1_exp())
                    self._pmemd(job, gpu, ["-O", "-i", "min1.in", "-o", "min1.out",
                        "-p", P, "-c", job.inpcrd, "-r", "min1.rst", "-ref", job.inpcrd],
                        "min1", 3600)
                    (wd / "min2.in").write_text(_mdin_min2_exp())
                    self._pmemd(job, gpu, ["-O", "-i", "min2.in", "-o", "min.out",
                        "-p", P, "-c", "min1.rst", "-r", "min.rst", "-ref", "min1.rst"],
                        "min", 3600); done.append("min")
                if "heat" in todo:
                    (wd / "heat.in").write_text(mdin_heat_exp(heat_steps))
                    self._pmemd(job, gpu, ["-O", "-i", "heat.in", "-o", "heat.out",
                        "-p", P, "-c", "min.rst", "-r", "heat.rst", "-ref", "min.rst"],
                        "heat", 3600); done.append("heat")
                if "equil" in todo:
                    (wd / "equil.in").write_text(mdin_npt_equil(equil_steps))
                    self._pmemd(job, gpu, ["-O", "-i", "equil.in", "-o", "equil.out",
                        "-p", P, "-c", "heat.rst", "-r", "equil.rst", "-ref", "heat.rst"],
                        "equil", 21600); done.append("equil")
                start = "equil.rst"
                (wd / "prod.in").write_text(mdin_prod_exp(nsteps, prod_frames))
            else:
                if "min" in todo:
                    (wd / "min.in").write_text(mdin_min())
                    self._pmemd(job, gpu, ["-O", "-i", "min.in", "-o", "min.out",
                        "-p", P, "-c", job.inpcrd, "-r", "min.rst", "-ref", job.inpcrd],
                        "min", 900); done.append("min")
                if "heat" in todo:
                    (wd / "heat.in").write_text(mdin_heat(heat_steps))
                    self._pmemd(job, gpu, ["-O", "-i", "heat.in", "-o", "heat.out",
                        "-p", P, "-c", "min.rst", "-r", "heat.rst", "-ref", "min.rst",
                        "-x", "heat.nc"], "heat", 1800); done.append("heat")
                start = "heat.rst"
                (wd / "prod.in").write_text(mdin_prod(nsteps, nframes=prod_frames))
            # production replicas (each reseeds via ig=-1 in the mdin)
            for r in range(max(1, job.replicas)):
                nc = f"prod_{r}.nc"
                if (wd / nc).exists() and (wd / nc).stat().st_size > 0:
                    trajs.append(nc); continue
                self._pmemd(job, gpu, ["-O", "-i", "prod.in", "-o", f"prod_{r}.out",
                    "-p", P, "-c", start, "-r", f"prod_{r}.rst", "-x", nc,
                    "-ref", start], f"prod_{r}", 172800)
                trajs.append(nc)
            if "prod" not in done:
                done.append("prod")
        except Exception as exc:  # noqa: BLE001
            res = AmberMDJobResult(job_id=job.job_id, status="failed", gpu=gpu,
                                   stages_done=done, trajectories=trajs, analysis={},
                                   simulation_ns=0.0,
                                   wallclock_s=round(time.monotonic() - t0, 1),
                                   failure_reason=str(exc)[:600])
            self._record(_job_record(job, res))
            return res

        analysis = self._analyze(job, trajs)
        lig_final = analysis.get("ligand_rmsd_final_A")
        status = "unstable" if (lig_final is not None and lig_final > 5.0) else "ok"
        res = AmberMDJobResult(
            job_id=job.job_id, status=status, gpu=gpu, stages_done=done,
            trajectories=trajs, analysis=analysis,
            simulation_ns=round(job.production_ns * max(1, job.replicas), 4),
            wallclock_s=round(time.monotonic() - t0, 1))
        self._record(_job_record(job, res))
        return res

    # -- analysis ----------------------------------------------------------- #
    def _analyze(self, job: AmberMDJob, trajs: Sequence[str]) -> dict:
        if not trajs:
            return {}
        masks = reactive_masks_from_manifest(job.manifest)
        script = reactive_cpptraj_script(masks, job.solvent, trajs)
        (job.workdir / "reactive_analyze.in").write_text(script)
        subprocess.run(["cpptraj", "-i", "reactive_analyze.in"], cwd=str(job.workdir),
                       capture_output=True, text=True, timeout=1800)
        rdat = (job.workdir / "reactive.dat")
        reactive = rdat.read_text() if rdat.exists() else ""
        lig = _col((job.workdir / "lig_rmsd.dat").read_text(), "lig_rmsd") \
            if (job.workdir / "lig_rmsd.dat").exists() else []
        bb = _col((job.workdir / "bb_rmsd.dat").read_text(), "bb_rmsd") \
            if (job.workdir / "bb_rmsd.dat").exists() else []
        na = float(job.manifest.get("reactive_spec", {}).get("distance_max", 3.6) or 3.6)
        summary = summarize_reactive(reactive, lig, bb, masks, near_attack_A=na)
        summary["energy_drift"] = _energy_drift(job.workdir)
        summary["claim_ceiling"] = (
            "screening-level reaction-geometry access evidence "
            "(NOT activity/kcat/activation-barrier)")
        return summary

    # -- fan-out ------------------------------------------------------------ #
    def run(self, jobs: Sequence[AmberMDJob]) -> List[AmberMDJobResult]:
        if not self.pmemd or (not amber_available() and not self.allow_cpu):
            return [AmberMDJobResult(j.job_id, "skipped_no_gpu", None, [], [], {},
                                     0.0, 0.0, "pmemd.cuda/AmberTools not available")
                    for j in jobs]
        if not self.gpu_pool and not self.allow_cpu:
            return [AmberMDJobResult(j.job_id, "skipped_no_gpu", None, [], [], {},
                                     0.0, 0.0,
                                     "no GPU in pool and allow_cpu=False (refusing "
                                     "silent CPU fallback)") for j in jobs]

        gpu_q: "queue.Queue[int]" = queue.Queue()
        for g in (self.gpu_pool or [0]):
            gpu_q.put(g)

        def worker(job: AmberMDJob) -> AmberMDJobResult:
            gpu = gpu_q.get()
            try:
                return self._run_job(job, gpu)
            finally:
                gpu_q.put(gpu)

        results: List[AmberMDJobResult] = []
        with ThreadPoolExecutor(max_workers=max(1, len(self.gpu_pool) or 1)) as ex:
            for res in ex.map(worker, jobs):
                results.append(res)
        return results


# --------------------------------------------------------------------------- #
# module helpers
# --------------------------------------------------------------------------- #
def _stage_ok(out: Path) -> bool:
    if not out.exists():
        return False
    txt = out.read_text(errors="replace")
    return any(m in txt for m in ("Final Performance Info", "TIMINGS",
                                  "Master Total wall time", "wallclock() was called"))


def _energy_drift(workdir: Path) -> float:
    for name in ("prod_0.out", "prod.out"):
        p = workdir / name
        if p.exists():
            txt = p.read_text()
            # pmemd appends "A V E R A G E S" + "R M S  F L U C T U A T I O N S"
            # summary blocks whose Etot lines are NOT trajectory frames — cut them
            # off or they poison the first-vs-last drift (a stable run reads ~100%).
            cut = txt.find("A V E R A G E S")
            if cut > 0:
                txt = txt[:cut]
            et = re.findall(r"Etot\s*=\s*(-?\d+\.\d+)", txt)
            if len(et) >= 2 and float(et[0]) != 0.0:
                return round(abs((float(et[-1]) - float(et[0])) / float(et[0])), 4)
    return 0.0


def _job_record(job: AmberMDJob, res: AmberMDJobResult) -> dict:
    return {
        "schema": "amber_gpu_job/v1",
        "job_id": job.job_id,
        "workdir": str(job.workdir),
        "gpu": res.gpu,
        "solvent": job.solvent,
        "production_ns": job.production_ns,
        "replicas": job.replicas,
        "status": res.status,
        "stages_done": res.stages_done,
        "trajectories": res.trajectories,
        "simulation_ns": res.simulation_ns,
        "wallclock_s": res.wallclock_s,
        "analysis": res.analysis,
        "failure_reason": res.failure_reason,
        "fingerprint": job.manifest.get("fingerprint"),
    }
