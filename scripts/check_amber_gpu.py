#!/usr/bin/env python
"""V6-0 Amber Capability Audit.

Discovers and records the Amber / AmberTools / CUDA / GPU runtime, then runs a
real ``tleap -> prmtop/inpcrd -> pmemd.cuda -> cpptraj`` round-trip on a tiny
built-from-scratch system to prove the production backend actually works on the
GPU (no silent CPU fallback: we exec the ``.cuda`` binary and parse its GPU
DEVICE INFO block).

Deliberately STANDALONE (no ``evoliez`` imports) so it runs as a pure capability
probe even in a broken env, and can be scp'd to the server on its own. Discovery
mirrors ``adapters/amber_engine`` env-var conventions
(``EVOLIEZ_AMBERTOOLS_BIN`` / ``EVOLIEZ_PMEMD_CUDA`` / ``AMBERHOME``).

Failure taxonomy (recorded, not swallowed): ``environment`` (tool missing / libs
not sourced), ``license``, ``parameterization`` (tleap/build), ``gpu`` (no device
/ CUDA error), ``input-preparation`` (bad decks).

Writes a reproducible profile JSON (``--out``); exit 0 iff the required implicit
round-trip completed on a confirmed GPU. Explicit-solvent (PME) is run too and
reported, but a PME-only failure downgrades the verdict rather than failing the
gate.

Usage (server, via scripts/run_amber_audit.sh which sets the env)::

    python check_amber_gpu.py --out reports/provenance/amber_runtime_profile.json \\
        --work /mnt/data/jglee/v6_audit_work
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA = "amber_runtime_profile/v1"

# Tools the roadmap (V6-0) wants verified. pmemd.cuda handled separately.
_AMBERTOOLS = [
    "tleap", "parmed", "antechamber", "parmchk2", "pdb4amber",
    "MCPB.py", "cpptraj", "sander", "sqm", "MMPBSA.py", "quick",
]


class RoundTripError(Exception):
    """Carries a failure-class label for the taxonomy."""

    def __init__(self, klass: str, stage: str, message: str):
        super().__init__(message)
        self.klass = klass
        self.stage = stage
        self.message = message


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def _which(tool: str) -> Optional[str]:
    return shutil.which(tool)


def _run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 120,
         env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True,
                          text=True, timeout=timeout, env=env)


def _tool_version(path: str, tool: str) -> str:
    """Best-effort version string; Amber tools are inconsistent about --version."""
    for flag in ("--version", "-V", "--help"):
        try:
            p = _run([path, flag], timeout=20)
            out = (p.stdout + p.stderr).strip()
            if not out:
                continue
            for line in out.splitlines():
                if re.search(r"(version|Amber|AmberTools|V\d|\d+\.\d+)", line, re.I):
                    return line.strip()[:200]
            return out.splitlines()[0][:200]
        except Exception:
            continue
    return "unknown"


def discover_pmemd() -> Tuple[Optional[str], Optional[str]]:
    """(path, variant). Honors EVOLIEZ_PMEMD_CUDA, then PATH."""
    cand = os.environ.get("EVOLIEZ_PMEMD_CUDA")
    if cand and Path(cand).exists():
        return cand, Path(cand).name
    for name in ("pmemd.cuda_SPFP", "pmemd.cuda", "pmemd.cuda_DPFP"):
        p = _which(name)
        if p:
            return p, name
    return None, None


def discover_tools() -> Dict[str, dict]:
    tools: Dict[str, dict] = {}
    # optional PATH augmentation, mirroring amber_engine.ensure_amber_on_path
    extra = os.environ.get("EVOLIEZ_AMBERTOOLS_BIN") or (
        str(Path(os.environ["AMBERHOME"]) / "bin") if os.environ.get("AMBERHOME") else None)
    if extra and Path(extra).is_dir() and not _which("tleap"):
        os.environ["PATH"] = f"{extra}:{os.environ.get('PATH', '')}"
    for tool in _AMBERTOOLS:
        path = _which(tool)
        tools[tool] = {
            "found": bool(path),
            "path": path,
            "version": _tool_version(path, tool) if path else None,
        }
    ppath, pvariant = discover_pmemd()
    tools["pmemd.cuda"] = {
        "found": bool(ppath),
        "path": ppath,
        "variant": pvariant,
        "version": _tool_version(ppath, "pmemd") if ppath else None,
    }
    return tools


def discover_cuda() -> dict:
    info: dict = {"driver_version": None, "toolkit_dirs": [], "nvcc_version": None}
    drv = Path("/proc/driver/nvidia/version")
    if drv.exists():
        try:
            first = drv.read_text().splitlines()[0]
            m = re.search(r"Kernel Module\s+([\d.]+)", first)
            info["driver_version"] = m.group(1) if m else first.strip()
        except Exception:
            pass
    for d in ("/usr/local/cuda", "/usr/local/cuda-12.4", "/usr/local/cuda-12"):
        if Path(d).exists():
            info["toolkit_dirs"].append(d)
    nvcc = _which("nvcc")
    if nvcc:
        try:
            out = _run([nvcc, "--version"], timeout=20).stdout
            m = re.search(r"release ([\d.]+)", out)
            info["nvcc_version"] = m.group(1) if m else None
        except Exception:
            pass
    return info


def discover_gpus() -> List[dict]:
    gpus: List[dict] = []
    smi = _which("nvidia-smi")
    if not smi:
        return gpus
    try:
        out = _run([smi, "--query-gpu=index,name,memory.total,memory.used,"
                    "utilization.gpu,driver_version",
                    "--format=csv,noheader,nounits"], timeout=30).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_total_mib": _to_int(parts[2]),
                    "mem_used_mib": _to_int(parts[3]),
                    "util_pct": _to_int(parts[4]),
                    "driver_version": parts[5] if len(parts) > 5 else None,
                })
    except Exception:
        pass
    return gpus


def _to_int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# round-trip
# --------------------------------------------------------------------------- #
_TLEAP_IMPLICIT = """\
source leaprc.protein.ff14SB
sys = sequence {{ ACE ALA ALA ALA NME }}
set default PBRadii mbondi3
saveamberparm sys {prm} {crd}
savepdb sys sys.pdb
quit
"""

_TLEAP_EXPLICIT = """\
source leaprc.protein.ff14SB
source leaprc.water.tip3p
sys = sequence {{ ACE ALA ALA ALA NME }}
solvateBox sys TIP3PBOX 18.0
addIons sys Na+ 0
addIons sys Cl- 0
saveamberparm sys {prm} {crd}
quit
"""
# NOTE: an 18 A buffer (not the 12 A production default) is used ONLY here so the
# tiny audit peptide clears pmemd.cuda's gpu_neighbor_list_setup "small box"
# guard (>2 hash cells/dim). Real production systems (a ~500-residue A-domain)
# are far above the threshold; this keeps the audit on the real GPU PME path
# rather than the -AllowSmallBox workaround.

_MIN_IMPLICIT = ("min implicit\n&cntrl\n imin=1, maxcyc=200, ncyc=100,\n"
                 " igb=8, ntb=0, cut=999.0, ntpr=50,\n/\n")
_MD_IMPLICIT = ("md implicit\n&cntrl\n imin=0, nstlim=500, dt=0.002,\n"
                " igb=8, ntb=0, cut=999.0, ntc=2, ntf=2,\n"
                " ntt=3, gamma_ln=2.0, tempi=50.0, temp0=300.0, ig=12345,\n"
                " ntpr=100, ntwx=100, ntwr=500,\n/\n")
_MIN_EXPLICIT = ("min explicit PME\n&cntrl\n imin=1, maxcyc=200, ncyc=100,\n"
                 " ntb=1, cut=10.0, ntpr=50,\n/\n")
_MD_EXPLICIT = ("md explicit PME NVT\n&cntrl\n imin=0, nstlim=500, dt=0.002,\n"
                " ntb=1, cut=10.0, iwrap=1, ntc=2, ntf=2,\n"
                " ntt=3, gamma_ln=2.0, tempi=50.0, temp0=300.0, ig=12345,\n"
                " ntpr=100, ntwx=100, ntwr=500,\n/\n")

_CPPTRAJ = """\
parm {prm}
trajin md.nc
rms toFirst @CA,C,N first out rms.dat
run
quit
"""

# strings that reveal the failure class if pmemd bombs
_GPU_ERR = re.compile(
    r"no CUDA-capable device|cudaGetDeviceCount|CUDA_ERROR|GPU allocation|"
    r"unspecified launch failure|invalid device|CUDA Device|out of memory|"
    r"UNSUPPORTED_PTX|forward compatibility", re.I)


def _pmemd_gpu_block(out_text: str) -> Optional[dict]:
    """Parse the GPU DEVICE INFO block pmemd.cuda prints. None => not present."""
    name = re.search(r"CUDA Device Name\s*:\s*(.+)", out_text)
    did = re.search(r"CUDA Device ID in use\s*:\s*(\d+)", out_text)
    mem = re.search(r"CUDA Device Global Mem Size\s*:\s*(\d+)", out_text)
    if not name:
        return None
    return {
        "device_id": int(did.group(1)) if did else None,
        "device_name": name.group(1).strip(),
        "global_mem_mib": int(mem.group(1)) if mem else None,
    }


def _pmemd_ok(out_path: Path) -> bool:
    """pmemd wrote a normal-termination footer."""
    if not out_path.exists():
        return False
    txt = out_path.read_text(errors="replace")
    return ("Final Performance Info" in txt or "wallclock() was called"
            in txt or "TIMINGS" in txt or "Master Total wall time" in txt)


def _stage(tag: str, cmd: List[str], cwd: Path, timeout: int,
           fail_class: str) -> subprocess.CompletedProcess:
    try:
        p = _run(cmd, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RoundTripError(fail_class, tag, f"{tag}: timed out after {timeout}s")
    except FileNotFoundError as exc:
        raise RoundTripError("environment", tag, f"{tag}: {exc}")
    return p


def _tleap_build(work: Path, script: str, prm: str, crd: str,
                 tag: str) -> None:
    (work / f"leap_{tag}.in").write_text(script.format(prm=prm, crd=crd))
    p = _stage(tag, ["tleap", "-f", f"leap_{tag}.in"], work, 300, "parameterization")
    log = (p.stdout + p.stderr)
    (work / f"leap_{tag}.log").write_text(log)
    if not (work / prm).exists() or not (work / crd).exists():
        klass = "parameterization"
        if re.search(r"could not|not found|leaprc|FATAL|Unknown", log, re.I):
            klass = "parameterization"
        raise RoundTripError(klass, tag, f"{tag}: tleap produced no prmtop/inpcrd\n"
                             + log[-1500:])


def _pmemd_run(work: Path, pmemd: str, mdin: str, prm: str, crd: str,
               out: str, rst: str, tag: str, timeout: int,
               nc: Optional[str] = None) -> dict:
    (work / f"{tag}.in").write_text(mdin)
    args = [pmemd, "-O", "-i", f"{tag}.in", "-o", out, "-p", prm, "-c", crd,
            "-r", rst, "-ref", crd]
    if nc:
        args += ["-x", nc]
    p = _stage(tag, args, work, timeout, "gpu")
    out_text = (work / out).read_text(errors="replace") if (work / out).exists() else ""
    combined = out_text + "\n" + p.stdout + "\n" + p.stderr
    gpu = _pmemd_gpu_block(out_text)
    if p.returncode != 0 or not _pmemd_ok(work / out):
        # "Small box" is a system-size prep issue, NOT a GPU-capability fault —
        # classify it as such so the audit does not falsely condemn the GPU.
        if re.search(r"Small box detected|AllowSmallBox", combined):
            klass = "input-preparation"
        elif re.search(r"license|LICENSE", combined):
            klass = "license"
        elif _GPU_ERR.search(combined):
            klass = "gpu"
        else:
            klass = "input-preparation"
        raise RoundTripError(klass, tag,
                             f"{tag}: pmemd rc={p.returncode}\n" + combined[-1800:])
    if gpu is None:
        raise RoundTripError("gpu", tag,
                             f"{tag}: pmemd finished but printed NO GPU DEVICE INFO "
                             "block — cannot confirm GPU execution")
    return gpu


def _cpptraj_analyze(work: Path, prm: str, tag: str) -> dict:
    (work / f"cpptraj_{tag}.in").write_text(_CPPTRAJ.format(prm=prm))
    p = _stage(tag, ["cpptraj", "-i", f"cpptraj_{tag}.in"], work, 120, "environment")
    (work / f"cpptraj_{tag}.log").write_text(p.stdout + p.stderr)
    rms = work / "rms.dat"
    if not rms.exists():
        raise RoundTripError("input-preparation", tag,
                             f"{tag}: cpptraj produced no rms.dat\n"
                             + (p.stdout + p.stderr)[-1200:])
    vals = [float(l.split()[1]) for l in rms.read_text().splitlines()
            if l.strip() and not l.startswith("#") and len(l.split()) >= 2]
    return {"n_frames": len(vals), "final_rmsd_A": vals[-1] if vals else None,
            "max_rmsd_A": max(vals) if vals else None}


def round_trip(work: Path, pmemd: str, mode: str) -> dict:
    """One full tleap->pmemd(min)->pmemd(md)->cpptraj cycle. mode in {implicit, explicit}."""
    sub = work / mode
    sub.mkdir(parents=True, exist_ok=True)
    t0 = _now_monotonic()
    prm, crd = "sys.prmtop", "sys.inpcrd"
    if mode == "implicit":
        _tleap_build(sub, _TLEAP_IMPLICIT, prm, crd, "implicit")
        _pmemd_run(sub, pmemd, _MIN_IMPLICIT, prm, crd, "min.out", "min.rst",
                   "min", 300)
        gpu = _pmemd_run(sub, pmemd, _MD_IMPLICIT, prm, "min.rst", "md.out",
                         "md.rst", "md", 600, nc="md.nc")
    else:
        _tleap_build(sub, _TLEAP_EXPLICIT, prm, crd, "explicit")
        _pmemd_run(sub, pmemd, _MIN_EXPLICIT, prm, crd, "min.out", "min.rst",
                   "min", 600)
        gpu = _pmemd_run(sub, pmemd, _MD_EXPLICIT, prm, "min.rst", "md.out",
                         "md.rst", "md", 900, nc="md.nc")
    analysis = _cpptraj_analyze(sub, prm, mode)
    n_atoms = _prmtop_natoms(sub / prm)
    return {
        "status": "ok",
        "mode": mode,
        "system": ("ACE-(ALA)3-NME, implicit GB (igb=8)" if mode == "implicit"
                   else "ACE-(ALA)3-NME + TIP3P box (PME, NVT)"),
        "n_atoms": n_atoms,
        "gpu_confirmed": True,
        "gpu_device": gpu,
        "analysis": analysis,
        "wallclock_s": round(_now_monotonic() - t0, 2),
        "input_sha1": _sha1_decks(mode),
    }


def _prmtop_natoms(prm: Path) -> Optional[int]:
    if not prm.exists():
        return None
    txt = prm.read_text(errors="replace")
    m = re.search(r"%FLAG POINTERS.*?%FORMAT[^\n]*\n\s*(\d+)", txt, re.S)
    return int(m.group(1)) if m else None


def _sha1_decks(mode: str) -> str:
    decks = ((_TLEAP_IMPLICIT, _MIN_IMPLICIT, _MD_IMPLICIT, _CPPTRAJ)
             if mode == "implicit"
             else (_TLEAP_EXPLICIT, _MIN_EXPLICIT, _MD_EXPLICIT, _CPPTRAJ))
    return hashlib.sha1("".join(decks).encode()).hexdigest()[:12]


# monotonic clock that also works when Date.now-style calls are unavailable
def _now_monotonic() -> float:
    import time
    return time.monotonic()


def _iso_now() -> str:
    # wall time for the record; if unavailable, fall back to "unknown"
    try:
        return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception:
        return "unknown"


def _git_commit() -> Optional[str]:
    try:
        p = _run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                 timeout=15)
        return p.stdout.strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_profile(args) -> dict:
    tools = discover_tools()
    profile: dict = {
        "schema": SCHEMA,
        "generated_at": _iso_now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": _git_commit(),
        "env": {
            "AMBERHOME": os.environ.get("AMBERHOME"),
            "EVOLIEZ_AMBERTOOLS_BIN": os.environ.get("EVOLIEZ_AMBERTOOLS_BIN"),
            "EVOLIEZ_PMEMD_CUDA": os.environ.get("EVOLIEZ_PMEMD_CUDA"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ld_library_has_amber": bool(
                re.search(r"amber|pmemd|evoliez",
                          os.environ.get("LD_LIBRARY_PATH", ""), re.I)),
        },
        "cuda": discover_cuda(),
        "gpus": discover_gpus(),
        "tools": tools,
        "roundtrip": {"implicit": None, "explicit": None},
        "failure": None,
        "verdict": None,
    }

    missing = [t for t in ("tleap", "cpptraj", "antechamber", "parmchk2")
               if not tools[t]["found"]]
    if missing or not tools["pmemd.cuda"]["found"]:
        profile["failure"] = {
            "class": "environment", "stage": "discovery",
            "message": f"missing tools: {missing}; "
                       f"pmemd.cuda found={tools['pmemd.cuda']['found']}",
        }
        profile["verdict"] = "FAIL: environment (tools not on PATH)"
        return profile

    if args.skip_roundtrip:
        profile["verdict"] = "DISCOVERY-ONLY (round-trip skipped)"
        return profile

    pmemd = tools["pmemd.cuda"]["path"]
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="amber_audit_"))
    work.mkdir(parents=True, exist_ok=True)
    profile["work_dir"] = str(work)

    # required gate: implicit round-trip on a confirmed GPU
    try:
        profile["roundtrip"]["implicit"] = round_trip(work, pmemd, "implicit")
    except RoundTripError as exc:
        profile["failure"] = {"class": exc.klass, "stage": exc.stage,
                              "message": exc.message}
        profile["verdict"] = f"FAIL: {exc.klass} at {exc.stage}"
        return profile

    # production path: explicit PME. Failure downgrades, does not fail the gate.
    if not args.no_explicit:
        try:
            profile["roundtrip"]["explicit"] = round_trip(work, pmemd, "explicit")
        except RoundTripError as exc:
            profile["roundtrip"]["explicit"] = {
                "status": "failed", "mode": "explicit",
                "failure": {"class": exc.klass, "stage": exc.stage,
                            "message": exc.message},
            }

    imp = profile["roundtrip"]["implicit"]
    exp = profile["roundtrip"]["explicit"]
    exp_ok = bool(exp and exp.get("status") == "ok")
    dev = imp["gpu_device"]["device_name"]
    if exp_ok or args.no_explicit:
        profile["verdict"] = (
            f"PASS: GPU round-trip OK on {dev} "
            f"(implicit{'+explicit-PME' if exp_ok else ''})")
    else:
        profile["verdict"] = (
            f"PARTIAL: implicit GPU round-trip OK on {dev}, "
            f"but explicit-PME failed ({exp['failure']['class']})")
    return profile


def main() -> int:
    ap = argparse.ArgumentParser(description="V6-0 Amber GPU capability audit")
    ap.add_argument("--out", default="reports/provenance/amber_runtime_profile.json")
    ap.add_argument("--work", default=None, help="scratch dir for the round-trip")
    ap.add_argument("--skip-roundtrip", action="store_true",
                    help="discovery only (local/dry)")
    ap.add_argument("--no-explicit", action="store_true",
                    help="skip the explicit-solvent PME round-trip")
    args = ap.parse_args()

    profile = build_profile(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))

    # human summary
    print(f"\n{'='*72}\nAMBER GPU CAPABILITY AUDIT\n{'='*72}")
    print(f"host           : {profile['host']}")
    print(f"cuda driver    : {profile['cuda']['driver_version']}")
    print(f"gpus           : {len(profile['gpus'])} "
          + ", ".join(f"[{g['index']}]{g['name']}" for g in profile['gpus'][:4]))
    tl = profile["tools"]
    print(f"pmemd.cuda     : {tl['pmemd.cuda'].get('variant')} "
          f"@ {tl['pmemd.cuda'].get('path')}")
    print("tools          : " + " ".join(
        f"{t}{'✓' if tl[t]['found'] else '✗'}" for t in _AMBERTOOLS))
    for mode in ("implicit", "explicit"):
        rt = profile["roundtrip"][mode]
        if rt is None:
            continue
        if rt.get("status") == "ok":
            d = rt["gpu_device"]
            print(f"roundtrip {mode:8} : OK  {rt['n_atoms']} atoms, "
                  f"{rt['wallclock_s']}s, GPU[{d['device_id']}]={d['device_name']}, "
                  f"cpptraj final_rmsd={rt['analysis']['final_rmsd_A']} Å")
        else:
            f = rt["failure"]
            print(f"roundtrip {mode:8} : FAIL [{f['class']}@{f['stage']}]")
    if profile["failure"]:
        f = profile["failure"]
        print(f"failure        : [{f['class']}@{f['stage']}] {f['message'][:160]}")
    print(f"\nVERDICT: {profile['verdict']}")
    print(f"profile written: {out}\n")

    ok = profile["verdict"] and profile["verdict"].startswith(("PASS", "PARTIAL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
