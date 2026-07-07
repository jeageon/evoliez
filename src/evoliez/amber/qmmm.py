"""V6-5 — Amber QM/MM-lite tier.

The classical V6-3 PMF localized the CAR bottleneck to the in-line ATTACK ANGLE
(never productive even when the umbrella forces the near-attack distance). This
tier escalates the SMALLEST necessary subset — the reaction core — to an
electronic-structure-aware treatment to ask whether that geometric finding
survives QM inspection, and whether any PMF-derived candidate difference is real.

Design (ROADMAP_V6 §8): the QM region is chosen by the MECHANISM (the reactive
substrate + cofactor + metal) — here the 3-HP nucleophile, the ATP phosphates and
the bridging Mg (net charge −3), treated with a semiempirical Hamiltonian (`sqm`
PM6 inside `sander`), the rest of the solvated protein staying MM. A short QM/MM
minimization of the reaction core (protein backbone restrained) relaxes the core
under QM and reports the QM/MM energy + the QM-relaxed reactive-core geometry
(O_nuc→Pα distance, in-line angle, Mg coordination).

Claim discipline: this produces reaction-core PLAUSIBILITY evidence — NOT an
activity/kcat/barrier claim. It is single-frame (or few-frame) and semiempirical.

Server-only (sander + sqm); the mask/mdin/parse helpers are unit-testable.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

_CLAIM = ("reaction-core PLAUSIBILITY evidence (semiempirical QM/MM-lite, "
          "single-frame; NOT activity/kcat/activation-barrier)")


@dataclass
class QmmmLiteSpec:
    qm_residues: List[str]          # residue names in the QM region, e.g. ['LIG','ATP','MG']
    qm_charge: int                  # net charge of the QM region (3-HP −1 + ATP −4 + Mg +2 = −3)
    qm_theory: str = "PM6"          # sqm semiempirical
    min_steps: int = 60
    backbone_restraint_wt: float = 5.0
    cut: float = 12.0
    # QM/MM-lite carves a SUBSYSTEM (full protein + QM core + a solvent shell) and
    # runs NON-periodic — a single-frame reaction-core plausibility check does not
    # need the full 72k-atom PME box, and running it does (the full box OOM'd sander
    # at 258 GB). Bulk solvent/ions beyond this radius of the core are stripped.
    strip_solvent_beyond_A: float = 6.0


@dataclass
class QmmmLiteResult:
    status: str                     # ok | failed | skipped
    qmmm_energy_kcal: Optional[float]
    reactive_geometry: Dict[str, float] = field(default_factory=dict)
    claim_ceiling: str = _CLAIM
    failure_reason: Optional[str] = None
    qm_atoms: Optional[int] = None


def qmmask_from_residues(residues: Sequence[str]) -> str:
    """Amber qmmask selecting whole residues by name (e.g. ':LIG,ATP,MG'). Whole
    ligand/metal residues -> the only QM/MM boundary is the non-covalent ligand↔
    protein/solvent interface, so NO link atoms are needed."""
    return ":" + ",".join(residues)


def mdin_qmmm_min(spec: QmmmLiteSpec, qmmask: str, periodic: bool = False) -> str:
    """sander QM/MM minimization deck. ifqnt=1 turns on QM/MM; the &qmmm namelist
    selects the QM region + Hamiltonian. On the carved (non-periodic) subsystem we
    use ntb=0 + a cutoff (qm_ewald=0) — the 'lite' choice for a single-frame
    plausibility check."""
    box = f"ntb=1, cut={spec.cut}" if periodic else f"ntb=0, cut={spec.cut}"
    return (
        "QM/MM-lite reaction-core minimization (backbone restrained)\n&cntrl\n"
        f" imin=1, maxcyc={spec.min_steps}, ncyc={spec.min_steps}, ntmin=1, drms=0.1,\n"
        f" {box}, ntc=1, ntf=1,\n"
        f" ntr=1, restraintmask='@CA,C,N,O', restraint_wt={spec.backbone_restraint_wt},\n"
        " ntpr=25, ifqnt=1,\n/\n"
        "&qmmm\n"
        f" qmmask='{qmmask}',\n qmcharge={spec.qm_charge},\n"
        f" qm_theory='{spec.qm_theory}',\n qmshake=0, qm_ewald=0, qmmm_int=1,\n"
        " verbosity=0,\n/\n")


def parse_qmmm_energy(mdout_text: str) -> Optional[float]:
    """Final QM/MM total energy (kcal/mol) from a sander QM/MM min. The FINAL RESULTS
    block reports the total (QM + MM + QM/MM) potential energy in the ENERGY column."""
    # last 'ENERGY' value in FINAL RESULTS
    m = re.findall(r"NSTEP\s+ENERGY.*?\n\s*\d+\s+(-?\d+\.\d+E?[+-]?\d*)", mdout_text, re.S)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            return None
    # fallback: any 'EAMBER (non-restraint)' / 'ESCF' aggregate
    m2 = re.findall(r"minimization\s+.*?ENERGY\s*=\s*(-?\d+\.\d+)", mdout_text)
    return float(m2[-1]) if m2 else None


# --------------------------------------------------------------------------- #
# orchestration (server-only)
# --------------------------------------------------------------------------- #
def _sh(cmd: List[str], cwd: Path, timeout: int):
    from evoliez.utils.subprocess_utils import _LIBC, _set_pdeathsig
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout,
                          preexec_fn=_set_pdeathsig if _LIBC is not None else None)


def _strip_subsystem(wd: Path, start_rst: Path, spec: QmmmLiteSpec,
                     tag: str) -> "tuple[str, str]":
    """Carve (full protein + QM core + a solvent shell) from the full solvated system
    at ``start_rst`` -> a small NON-periodic prmtop/rst. Returns (prmtop, rst)
    filenames relative to ``wd``. Bulk solvent/ions beyond the shell are stripped so
    sander QM/MM does not OOM on the 72k-atom box."""
    core = "|".join(f":{r}" for r in spec.qm_residues)
    keep_shell = f"{core}<:{spec.strip_solvent_beyond_A:.1f}"
    strip_mask = f"(:WAT|:Na+|:Cl-|:K+)&!({keep_shell})"
    prm, rst = f"{tag}_sub.complex.prmtop", f"{tag}_sub.rst"
    # a distance-based strip needs a REFERENCE frame to measure against
    script = (f"parm complex.prmtop\nreference {start_rst}\ntrajin {start_rst}\n"
              f"strip {strip_mask} outprefix {tag}_sub\n"
              f"trajout {rst} restart nobox\nrun\nquit\n")
    (wd / f"{tag}_strip.in").write_text(script)
    _sh(["cpptraj", "-i", f"{tag}_strip.in"], wd, 600)
    return prm, rst


def run_qmmm_lite(system_workdir: Path, start_rst: Path, spec: QmmmLiteSpec,
                  reactive_masks: Dict[str, str], *, sander: str = "sander",
                  tag: str = "qmmm") -> QmmmLiteResult:
    """One QM/MM-lite reaction-core minimization from ``start_rst`` (a near-attack
    frame): carve a non-periodic subsystem, QM/MM-minimize it, then read the
    QM-relaxed reactive-core geometry via cpptraj."""
    wd = Path(system_workdir)
    if not (wd / "complex.prmtop").exists():
        return QmmmLiteResult("skipped", None, failure_reason="no complex.prmtop")

    if spec.strip_solvent_beyond_A > 0:
        P, cfrom = _strip_subsystem(wd, start_rst, spec, tag)
        if not (wd / P).exists() or not (wd / cfrom).exists():
            return QmmmLiteResult("failed", None,
                                  failure_reason="cpptraj subsystem strip produced no prmtop/rst")
        periodic = False
    else:
        P, cfrom, periodic = "complex.prmtop", str(start_rst), True

    qmmask = qmmask_from_residues(spec.qm_residues)
    (wd / f"{tag}.in").write_text(mdin_qmmm_min(spec, qmmask, periodic=periodic))
    r = _sh([sander, "-O", "-i", f"{tag}.in", "-o", f"{tag}.out", "-p", P,
             "-c", cfrom, "-r", f"{tag}.rst", "-ref", cfrom], wd, 7200)
    out = wd / f"{tag}.out"
    txt = out.read_text(errors="replace") if out.exists() else ""
    if r.returncode != 0 or "FINAL RESULTS" not in txt:
        return QmmmLiteResult(
            "failed", None,
            failure_reason=f"sander QM/MM rc={r.returncode}: {(txt + r.stderr)[-700:]}")
    energy = parse_qmmm_energy(txt)
    n_qm = None
    mqm = re.search(r"Number of QM atoms\s*=?\s*(\d+)", txt) or \
        re.search(r"QM ATOM.*?=\s*(\d+)", txt)
    if mqm:
        n_qm = int(mqm.group(1))

    geom = _core_geometry(wd, P, f"{tag}.rst", reactive_masks)
    return QmmmLiteResult("ok", round(energy, 3) if energy is not None else None,
                          reactive_geometry=geom, qm_atoms=n_qm)


def _core_geometry(wd: Path, prmtop: str, rst: str, masks: Dict[str, str]) -> Dict[str, float]:
    onuc, pa, olv, mg = (masks.get("O_nuc"), masks.get("P_alpha"),
                         masks.get("O_leaving"), masks.get("metal"))
    lines = [f"parm {prmtop}", f"trajin {rst}"]
    if onuc and pa:
        lines.append(f"distance d_onuc_pa {onuc} {pa} out qmgeom.dat")
    if onuc and pa and olv:
        lines.append(f"angle a_inline {onuc} {pa} {olv} out qmgeom.dat")
    if mg and onuc:
        lines.append(f"distance d_mg_onuc {mg} {onuc} out qmgeom.dat")
    if mg and pa:
        lines.append(f"distance d_mg_pa {mg} {pa} out qmgeom.dat")
    lines += ["run", "quit"]
    (wd / "qmgeom.in").write_text("\n".join(lines) + "\n")
    _sh(["cpptraj", "-i", "qmgeom.in"], wd, 300)
    dat = wd / "qmgeom.dat"
    out: Dict[str, float] = {}
    if dat.exists():
        rows = [l for l in dat.read_text().splitlines() if l.strip()]
        if len(rows) >= 2:
            hdr = rows[0].lstrip("#").split()
            vals = rows[1].split()
            for k, v in zip(hdr[1:], vals[1:]):
                try:
                    out[k] = round(float(v), 3)
                except ValueError:
                    pass
    return out
