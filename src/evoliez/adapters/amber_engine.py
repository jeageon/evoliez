"""Amber ``pmemd.cuda`` MD backend - the s10 tier-3 confirmatory engine.

Mirrors ``openmm_engine.run_md``'s contract (-> :class:`MDResult`) but runs the
rigorous Amber path: ``tleap`` (ff14SB + GAFF2/AM1-BCC) -> ``pmemd.cuda`` (a
staged restrained protocol: minimize -> heat 0->300K -> production) -> ``cpptraj``
(ligand/pocket RMSD, catalytic min-distances, h-bonds). Server-only (AmberTools
+ a CUDA ``pmemd``); honest skips mirror the OpenMM path so ``md/analysis.py``
treats both engines identically.

Mechanics validated by ``scripts/smoke_amber_{param,pmemd,cpptraj}.py``.

Tool discovery is PATH-based; the launcher (or ``EVOLIEZ_AMBERTOOLS_BIN`` /
``EVOLIEZ_PMEMD_CUDA``) must put AmberTools + a CUDA ``pmemd`` on PATH. v1 is
implicit-GB (igb8); explicit-solvent is a planned config branch.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from evoliez.adapters.base import is_full_atom_pdb
from evoliez.adapters.openmm_engine import (
    MDResult,
    _ligand_rdkit_at_pose,
    _pdb_one_letter_seq,
    _production_nsteps,
    _run_mock,
)
from evoliez.config import MDConfig
from evoliez.types import Complex

log = logging.getLogger(__name__)

_AMBER_TOOLS = ("pdb4amber", "antechamber", "parmchk2", "tleap", "cpptraj")


# --------------------------------------------------------------------------- #
# Environment / tool discovery
# --------------------------------------------------------------------------- #
def ensure_amber_on_path() -> None:
    """Prepend ``EVOLIEZ_AMBERTOOLS_BIN`` to PATH if AmberTools isn't already
    found. Shared safety net: BOTH the OpenMM tier (OpenFF needs antechamber/sqm
    for AM1-BCC charges) and this Amber tier silently skip every candidate when
    the charge backend is missing, so a launcher that forgets to set PATH must
    not nuke the whole MD layer."""
    if shutil.which("antechamber"):
        return
    extra = os.environ.get("EVOLIEZ_AMBERTOOLS_BIN")
    if extra and Path(extra).is_dir():
        os.environ["PATH"] = f"{extra}:{os.environ.get('PATH', '')}"
        lib = str(Path(extra).parent / "lib")
        if Path(lib).is_dir():
            os.environ["LD_LIBRARY_PATH"] = (
                f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}")
        log.info("prepended EVOLIEZ_AMBERTOOLS_BIN=%s to PATH", extra)


def _pmemd_cuda() -> Optional[str]:
    cand = os.environ.get("EVOLIEZ_PMEMD_CUDA")
    if cand and Path(cand).exists():
        return cand
    return shutil.which("pmemd.cuda_SPFP") or shutil.which("pmemd.cuda")


def amber_available() -> bool:
    ensure_amber_on_path()
    return all(shutil.which(t) for t in _AMBER_TOOLS) and bool(_pmemd_cuda())


# --------------------------------------------------------------------------- #
# Input-deck generators (pure -> unit-testable)
# --------------------------------------------------------------------------- #
_BB = "@CA,C,N,O"   # backbone restraint mask (kept off the pocket refinement)


def mdin_min(restraint_wt: float = 5.0) -> str:
    return ("restrained minimize (implicit GB)\n&cntrl\n"
            " imin=1, maxcyc=2000, ncyc=1000,\n igb=8, cut=999.0, ntb=0,\n"
            f" ntr=1, restraintmask='{_BB}', restraint_wt={restraint_wt},\n"
            " ntpr=200,\n/\n")


def mdin_heat(nsteps: int, restraint_wt: float = 5.0) -> str:
    return ("heat 0->300K, restrained backbone\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt=0.002, irest=0, ntx=1,\n"
            " igb=8, cut=999.0, ntb=0,\n ntc=2, ntf=2,\n"
            " ntt=3, gamma_ln=2.0, tempi=0.0, temp0=300.0, ig=-1,\n"
            f" ntr=1, restraintmask='{_BB}', restraint_wt={restraint_wt},\n"
            " ntpr=500, ntwx=500,\n/\n")


def mdin_prod(nsteps: int, restraint_wt: float = 0.5, nframes: int = 50) -> str:
    interval = max(1, nsteps // max(1, nframes))
    return ("production 300K (weak backbone restraint)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt=0.002, irest=1, ntx=5,\n"
            " igb=8, cut=999.0, ntb=0,\n ntc=2, ntf=2,\n"
            " ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,\n"
            f" ntr=1, restraintmask='{_BB}', restraint_wt={restraint_wt},\n"
            f" ntpr={interval}, ntwx={interval}, ntwr={nsteps},\n/\n")


def mdin_min_exp(restraint_wt: float = 5.0) -> str:
    return ("restrained minimize (explicit, PME)\n&cntrl\n"
            " imin=1, maxcyc=2000, ncyc=1000,\n ntb=1, cut=10.0,\n"
            f" ntr=1, restraintmask='{_BB}', restraint_wt={restraint_wt},\n ntpr=200,\n/\n")


def mdin_heat_exp(nsteps: int, restraint_wt: float = 5.0) -> str:
    return ("heat 0->300K NVT, restrained backbone (explicit, PME)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt=0.002, irest=0, ntx=1,\n"
            " ntb=1, cut=10.0, iwrap=1,\n ntc=2, ntf=2,\n"
            " ntt=3, gamma_ln=2.0, tempi=0.0, temp0=300.0, ig=-1,\n"
            f" ntr=1, restraintmask='{_BB}', restraint_wt={restraint_wt},\n"
            " ntpr=500, ntwx=0,\n/\n")


def mdin_npt_equil(nsteps: int, restraint_wt: float = 1.0) -> str:
    return ("NPT density equilibration, weak restraint (explicit, PME)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt=0.002, irest=1, ntx=5,\n"
            " ntb=2, ntp=1, barostat=2, pres0=1.0, taup=2.0,\n cut=10.0, iwrap=1,\n"
            " ntc=2, ntf=2,\n ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,\n"
            f" ntr=1, restraintmask='{_BB}', restraint_wt={restraint_wt},\n"
            " ntpr=500, ntwx=0,\n/\n")


def mdin_prod_exp(nsteps: int, nframes: int = 250) -> str:
    interval = max(1, nsteps // max(1, nframes))
    return ("NPT production, free (explicit, PME)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt=0.002, irest=1, ntx=5,\n"
            " ntb=2, ntp=1, barostat=2, pres0=1.0, taup=2.0,\n cut=10.0, iwrap=1,\n"
            " ntc=2, ntf=2,\n ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,\n"
            f" ntpr={interval}, ntwx={interval}, ntwr={nsteps},\n/\n")


def tleap_script(net_charge: int = 0, solvent: str = "implicit") -> str:
    """tleap build. implicit = ff14SB + GB radii (tier-2-equivalent rigour).
    explicit = ff19SB + OPC water (the FF19SB-matched model) in a truncated
    octahedron + addIons neutralize (addIons, NOT addIonsRand: with #=0 the
    latter rejects a second ion type). 0.15 M physiological salt is a TODO
    refinement (needs the post-solvate water count)."""
    if solvent == "explicit":
        return ("source leaprc.protein.ff19SB\n"
                "source leaprc.gaff2\n"
                "source leaprc.water.opc\n"
                "loadamberparams ligand.frcmod\n"
                "LIG = loadmol2 ligand.mol2\n"
                "prot = loadpdb protein_clean.pdb\n"
                "comp = combine {prot LIG}\n"
                "solvateOct comp OPCBOX 12.0\n"
                "addIons comp Na+ 0\n"
                "addIons comp Cl- 0\n"
                "saveamberparm comp complex.prmtop complex.inpcrd\n"
                "quit\n")
    return ("source leaprc.protein.ff14SB\n"
            "source leaprc.gaff2\n"
            "loadamberparams ligand.frcmod\n"
            "LIG = loadmol2 ligand.mol2\n"
            "prot = loadpdb protein_clean.pdb\n"
            "comp = combine {prot LIG}\n"
            "set default PBRadii mbondi3\n"
            "saveamberparm comp complex.prmtop complex.inpcrd\n"
            "quit\n")


def cpptraj_script(catalytic_positions: Sequence[int],
                   solvent: str = "implicit") -> str:
    """rms (ligand + backbone), per-catalytic-residue min-distance to the
    ligand (nativecontacts mindist -> matches the OpenMM heavy-atom metric),
    and ligand h-bonds. Explicit trajectories are autoimaged (PBC) and the
    solvent/ions stripped so the masks below act on the solute only."""
    lines = ["parm complex.prmtop", "trajin prod.nc"]
    if solvent == "explicit":
        lines += ["autoimage", "strip :WAT,Na+,Cl-,K+"]
    lines += [
        "rms fit @CA,C,N first",                       # superpose on backbone
        "rms ligand :LIG&!@H= first nofit out ligand_rmsd.dat",
        "rms backbone @CA,C,N first nofit out pocket_rmsd.dat",
    ]
    for p in catalytic_positions:
        lines.append(
            f"nativecontacts :LIG :{int(p)} mindist out cat_{int(p)}.dat "
            f"distance 12.0 first")
    lines += [
        "hbond HB :LIG out hbond.dat",
        "run",
        "quit",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #
def _sh(cmd: List[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)


def _build_system(cx: Complex, pdb_path: Path, workdir: Path,
                  solvent: str = "implicit") -> None:
    """pdb4amber + antechamber(AM1-BCC) + parmchk2 + tleap -> complex.prmtop.
    Raises _AmberParamUnsupported (-> neutral skip) when antechamber/sqm cannot
    charge the ligand; raises RuntimeError on a real protein/tleap failure."""
    # protein-only PDB
    prot = [l for l in pdb_path.read_text().splitlines() if l.startswith("ATOM")]
    (workdir / "protein_in.pdb").write_text("\n".join(prot) + "\nEND\n")
    r = _sh(["pdb4amber", "-i", "protein_in.pdb", "-o", "protein_clean.pdb",
             "--nohyd", "--dry"], workdir, 120)
    if r.returncode != 0 or not (workdir / "protein_clean.pdb").exists():
        raise RuntimeError(f"pdb4amber failed: {r.stderr[-400:]}")

    # ligand at the docked pose -> SDF -> antechamber (AM1-BCC, gaff2)
    rd = _ligand_rdkit_at_pose(pdb_path, cx.ligand.smiles)
    from rdkit import Chem
    with Chem.SDWriter(str(workdir / "ligand.sdf")) as w:
        w.write(rd)
    nc = int(getattr(cx.ligand, "formal_charge", 0) or 0)
    r = _sh(["antechamber", "-i", "ligand.sdf", "-fi", "sdf", "-o", "ligand.mol2",
             "-fo", "mol2", "-c", "bcc", "-nc", str(nc), "-at", "gaff2",
             "-rn", "LIG"], workdir, 900)
    if r.returncode != 0 or not (workdir / "ligand.mol2").exists():
        raise _AmberParamUnsupported(
            f"antechamber/AM1-BCC failed (large/charged cofactor?): "
            f"{(r.stdout + r.stderr)[-400:]}")
    r = _sh(["parmchk2", "-i", "ligand.mol2", "-f", "mol2", "-o", "ligand.frcmod"],
            workdir, 120)
    if r.returncode != 0:
        raise _AmberParamUnsupported(f"parmchk2 failed: {r.stderr[-300:]}")

    (workdir / "tleap.in").write_text(tleap_script(nc, solvent))
    r = _sh(["tleap", "-s", "-f", "tleap.in"], workdir, 600)
    if not (workdir / "complex.prmtop").exists():
        raise RuntimeError(f"tleap failed: {r.stdout[-400:]}")


def _run_stage(tag: str, args: List[str], workdir: Path, pmemd: str,
               timeout: int) -> None:
    r = _sh([pmemd] + args, workdir, timeout)
    if r.returncode != 0:
        raise RuntimeError(f"pmemd {tag} failed (rc={r.returncode}): "
                           f"{r.stderr[-400:]}")


def _series(workdir: Path, name: str, col: int = 1) -> List[float]:
    """Value column ``col`` (1-based after the frame index) of a cpptraj .dat.
    nativecontacts ``out`` writes Frame|native|nonnative|mindist, so catalytic
    min-distance is col 3 (not the native-contact COUNT in col 1)."""
    p = workdir / name
    if not p.exists():
        return []
    out: List[float] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) > col:
            try:
                out.append(float(parts[col]))
            except ValueError:
                pass
    return out


def _analyze(workdir: Path, cfg: MDConfig,
             catalytic_positions: Sequence[int]) -> Dict[str, object]:
    solvent = "explicit" if getattr(cfg, "solvent", "implicit") == "explicit" else "implicit"
    (workdir / "analyze.in").write_text(cpptraj_script(catalytic_positions, solvent))
    _sh(["cpptraj", "-i", "analyze.in"], workdir, 600)

    lig = _series(workdir, "ligand_rmsd.dat")
    pkt = _series(workdir, "pocket_rmsd.dat")
    key: Dict[str, List[float]] = {}
    for p in catalytic_positions:
        s = _series(workdir, f"cat_{int(p)}.dat", col=3)   # mindist, not count
        if s:
            key[f"cat_{int(p)}"] = [round(v, 3) for v in s]
    # hbond.dat col 1 = number of ligand h-bonds per frame -> occupancy
    hb = _series(workdir, "hbond.dat")
    hbond_occ = round(sum(1 for v in hb if v >= 1) / len(hb), 3) if hb else 0.0
    # energy drift from the production mdout
    drift = 0.0
    mdout = workdir / "prod.out"
    if mdout.exists():
        et = re.findall(r"Etot\s*=\s*(-?\d+\.\d+)", mdout.read_text())
        if len(et) >= 2 and float(et[0]) != 0.0:
            drift = abs((float(et[-1]) - float(et[0])) / float(et[0]))
    return {
        "ligand_rmsd_series": [round(v, 3) for v in lig] or [0.0],
        "pocket_rmsd_series": [round(v, 3) for v in pkt] or [0.0],
        "key_distances": key,
        "hbond_occupancy": hbond_occ,
        "energy_drift": round(drift, 4),
    }


class _AmberParamUnsupported(Exception):
    """Ligand the AmberTools small-molecule path cannot charge -> neutral skip."""


def _skip(candidate_id: str, cfg: MDConfig, status: str, reason: str) -> MDResult:
    return MDResult(candidate_id=candidate_id, status=status,
                    protocol_level=cfg.protocol_level, solvent_mode="implicit",
                    simulation_time_ns=0.0, failure_reason=reason)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_md_amber(cx: Complex, candidate_id: str, cfg: MDConfig, workdir: Path,
                 *, catalytic_positions: Sequence[int], dry_run: bool,
                 ligand_cache_dir: "Path | None" = None) -> MDResult:
    workdir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log.info("[dry-run] would run Amber pmemd.cuda L%d for %s",
                 cfg.protocol_level, candidate_id)
        return _run_mock(cx, candidate_id, cfg, workdir, instability=0.2,
                         catalytic_positions=catalytic_positions)

    pmemd = _pmemd_cuda()
    if not amber_available():
        return _skip(candidate_id, cfg, "skipped_parameterization",
                     "AmberTools/pmemd.cuda not on PATH (set EVOLIEZ_AMBERTOOLS_BIN"
                     " / EVOLIEZ_PMEMD_CUDA)")

    # full-atom structure required (same honest skip as the OpenMM path)
    src = getattr(cx.structure, "pdb_path", None)
    if not (src and Path(src).exists() and is_full_atom_pdb(Path(src))):
        return _skip(candidate_id, cfg, "skipped_no_full_atom_structure",
                     "Amber MD requires a full-atom protein (got CA-only/mock)")
    pdb_path = Path(src)
    want = (cx.structure.sequence or "").upper()
    have = _pdb_one_letter_seq(pdb_path)
    if want and have and want != have:
        n = sum(1 for a, b in zip(want, have) if a != b)
        return _skip(candidate_id, cfg, "skipped_no_mutant_structure",
                     f"{n} residue(s) differ: PDB is WT, not this mutant")

    solvent = "explicit" if getattr(cfg, "solvent", "implicit") == "explicit" else "implicit"
    try:
        _build_system(cx, pdb_path, workdir, solvent)
    except _AmberParamUnsupported as exc:
        log.warning("Amber ligand param skipped for %s: %s", candidate_id, exc)
        return _skip(candidate_id, cfg, "skipped_parameterization", str(exc))
    except Exception as exc:  # protein / tleap assembly = real failure
        log.warning("Amber system build failed for %s: %s", candidate_id, exc)
        return MDResult(candidate_id=candidate_id, status="failed",
                        protocol_level=cfg.protocol_level, solvent_mode="implicit",
                        simulation_time_ns=0.0, integration_failed=True,
                        failure_reason=f"amber build: {exc}")

    # staged GPU protocol. implicit: min -> heat -> production. explicit (PME):
    # min -> NVT heat -> NPT density equilibration -> NPT (free) production.
    nsteps, actual_ns = _production_nsteps(cfg)
    heat_steps = 10000
    equil_steps = max(1000, int(getattr(cfg, "equilibration_ps", 100.0) * 1000
                                / max(0.1, cfg.timestep_fs)))
    P = "complex.prmtop"
    try:
        if solvent == "explicit":
            (workdir / "min.in").write_text(mdin_min_exp())
            (workdir / "heat.in").write_text(mdin_heat_exp(heat_steps))
            (workdir / "equil.in").write_text(mdin_npt_equil(equil_steps))
            (workdir / "prod.in").write_text(mdin_prod_exp(nsteps))
            _run_stage("min", ["-O", "-i", "min.in", "-o", "min.out", "-p", P,
                               "-c", "complex.inpcrd", "-r", "min.rst",
                               "-ref", "complex.inpcrd"], workdir, pmemd, 1800)
            _run_stage("heat", ["-O", "-i", "heat.in", "-o", "heat.out", "-p", P,
                                "-c", "min.rst", "-r", "heat.rst",
                                "-ref", "min.rst"], workdir, pmemd, 3600)
            _run_stage("equil", ["-O", "-i", "equil.in", "-o", "equil.out", "-p", P,
                                 "-c", "heat.rst", "-r", "equil.rst",
                                 "-ref", "heat.rst"], workdir, pmemd, 21600)
            _run_stage("prod", ["-O", "-i", "prod.in", "-o", "prod.out", "-p", P,
                                "-c", "equil.rst", "-r", "prod.rst",
                                "-x", "prod.nc"], workdir, pmemd, 172800)
        else:
            (workdir / "min.in").write_text(mdin_min())
            (workdir / "heat.in").write_text(mdin_heat(heat_steps))
            (workdir / "prod.in").write_text(mdin_prod(nsteps))
            _run_stage("min", ["-O", "-i", "min.in", "-o", "min.out", "-p", P,
                               "-c", "complex.inpcrd", "-r", "min.rst",
                               "-ref", "complex.inpcrd"], workdir, pmemd, 600)
            _run_stage("heat", ["-O", "-i", "heat.in", "-o", "heat.out", "-p", P,
                                "-c", "min.rst", "-r", "heat.rst",
                                "-ref", "min.rst", "-x", "heat.nc"], workdir, pmemd, 1200)
            _run_stage("prod", ["-O", "-i", "prod.in", "-o", "prod.out", "-p", P,
                                "-c", "heat.rst", "-r", "prod.rst",
                                "-ref", "heat.rst", "-x", "prod.nc"], workdir, pmemd, 3600)
    except Exception as exc:
        log.warning("Amber pmemd run failed for %s: %s", candidate_id, exc)
        return MDResult(candidate_id=candidate_id, status="failed",
                        protocol_level=cfg.protocol_level, solvent_mode="implicit",
                        simulation_time_ns=0.0, integration_failed=True,
                        failure_reason=f"amber pmemd: {exc}")

    m = _analyze(workdir, cfg, catalytic_positions)
    lig = m["ligand_rmsd_series"]
    status = "unstable" if (lig and lig[-1] > 5.0) else "ok"
    return MDResult(
        candidate_id=candidate_id, status=status,
        protocol_level=cfg.protocol_level, solvent_mode=solvent,
        simulation_time_ns=round(actual_ns, 6),
        minimized_pdb=str(workdir / "min.rst"),
        trajectory_path=str(workdir / "prod.nc"),
        ligand_rmsd_series=m["ligand_rmsd_series"],
        pocket_rmsd_series=m["pocket_rmsd_series"],
        key_distances=m["key_distances"],
        contact_occupancy={},
        hbond_occupancy=m["hbond_occupancy"],
        energy_drift=m["energy_drift"],
    )
