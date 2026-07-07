"""Amber alchemical RBFE (Phase B) -- relative ΔΔG for residue mutations via
modern unified softcore TI on the GPU.

Thermodynamic cycle (binding):  ΔΔG_bind = ΔG_complex(X→Y) − ΔG_apo(X→Y)
Each leg is one alchemical X→Y mutation run as a single softcore TI
(icfe=1, ifsc=1; timask1/scmask1 = the WT-unique sidechain atoms, timask2/scmask2
= the mutant-unique atoms) integrated over a λ-schedule. The hybrid
(dual-sidechain) topology + the unique-atom masks come from the vendored Py3
softcore_setup (scripts/softcore_setup_py3.py). The two end states share ONE
solvent box -- the mutant is the solvated WT with the target residue swapped and
all waters kept -- so the common region (protein backbone + every other residue
+ every water/ion) matches atom-for-atom, which is what softcore_setup requires.

Engine constraints (validated, see scripts/smoke_amber_ti.py): pmemd.cuda TI
needs EXPLICIT solvent and softcore needs ntf=1 + noshakemask on the perturbed
region. dV/dλ is parsed from the mdout and integrated (TI) or fed to
ndfes/edgembar (MBAR) for ΔG.

Build order is incremental; this file currently covers: hybrid-topology builder
(apo + complex legs), the TI mdin generators, and the λ-schedule. The λ-window
orchestration + ndfes/edgembar ΔΔG live in run_rbfe (below).
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evoliez.adapters.amber_engine import _sh, ensure_amber_on_path, _pmemd_cuda

# vendored Py3-patched softcore_setup (repo scripts/) -- amber_rbfe.py is at
# src/evoliez/adapters/, so parents[3] == repo root.
_SOFTCORE = Path(__file__).resolve().parents[3] / "scripts" / "softcore_setup_py3.py"

_STD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "HID",
    "HIE", "HIP", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR",
    "TRP", "TYR", "VAL", "CYX",
}
_BACKBONE = {"N", "CA", "C", "O"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Mutation:
    resid: int
    wt_resname: str
    mut_resname: str
    chain: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.wt_resname}{self.resid}{self.mut_resname}"


@dataclass
class TIMasks:
    """Unique-atom Amber masks for unified softcore TI (V0=WT, V1=mutant)."""
    timask1: str        # WT-unique atoms,  e.g. ':84@HA2,HA3'
    timask2: str        # mutant-unique atoms, e.g. ':84@CB,HA,HB1,HB2,HB3'

    @property
    def scmask1(self) -> str:
        return self.timask1

    @property
    def scmask2(self) -> str:
        return self.timask2

    @property
    def noshakemask(self) -> str:
        """The whole mutated residue -- a compact superset of both perturbed
        regions (softcore bonds can't be SHAKEn) that keeps the mdin line under
        the 80-column Fortran namelist limit, unlike the explicit atom union."""
        mask = self.timask1 or self.timask2
        return mask.split("@", 1)[0] if mask else ""


def _union_mask(m1: str, m2: str) -> str:
    """':R@A,B' ∪ ':R@C,D' -> ':R@A,B,C,D' (single-residue mutation)."""
    r1, a1 = m1.split("@", 1)
    r2, a2 = m2.split("@", 1)
    atoms = a1.split(",") + [a for a in a2.split(",") if a not in a1.split(",")]
    res = r1 if r1 == r2 else f"{r1}|{r2}"   # same residue in practice
    return f"{r1}@{','.join(atoms)}" if r1 == r2 else f"({m1})|({m2})"


def _normalise_amber_mask(mask: str) -> str:
    mask = (mask or "").strip()
    if not mask:
        return ""
    return mask if mask.startswith((":", "@", "(")) else f":{mask}"


def _parse_softcore_masks(text: str) -> TIMasks:
    """Pull the two 'Alchemy step' scmasks softcore_setup prints: the first is
    the WT-unique region (for wt.prmtop), the second the mutant-unique region
    (for mut.SC.prmtop)."""
    sc = re.findall(r"\bscmask\s*=\s*['\"]([^'\"]*)['\"]", text)
    if len(sc) < 2:
        raise RuntimeError(
            f"softcore_setup did not emit two scmasks (got {sc}). Output:\n{text[-800:]}")
    return TIMasks(
        timask1=_normalise_amber_mask(sc[0]),
        timask2=_normalise_amber_mask(sc[1]),
    )


# --------------------------------------------------------------------------- #
# Structure mutation (operates on the SOLVATED WT pdb -> matched-solvent mutant)
# --------------------------------------------------------------------------- #
def mutate_structure(pdb_text: str, resid: int, mut_resname: str,
                     chain: Optional[str] = None) -> str:
    """Swap one protein residue X->Y in a (solvated) PDB: keep N/CA/C/O (+CB if
    Y has one), rename to Y, drop the rest of X's sidechain; tleap rebuilds Y's
    sidechain. Every other residue, water and ion is kept verbatim so the two
    end states share an identical solvent box. PRO is unsupported (ring)."""
    if mut_resname == "PRO":
        raise ValueError("mutation to PRO is unsupported (proline ring)")
    keep = set(_BACKBONE)
    if mut_resname != "GLY":
        keep.add("CB")
    out: List[str] = []
    for ln in pdb_text.splitlines():
        if ln.startswith(("ATOM", "HETATM")):
            rn = ln[17:20].strip()
            try:
                ri = int(ln[22:26])
            except ValueError:
                out.append(ln)
                continue
            same = ri == resid and rn in _STD_AA and (
                chain is None or ln[21] == chain)
            if same:
                atom = ln[12:16].strip()
                if atom in keep:
                    out.append(ln[:17] + f"{mut_resname:>3}" + ln[20:])
                # else: drop X-unique sidechain atom
                continue
        out.append(ln)
    return "\n".join(out) + "\n"


def _cryst1_box(pdb_text: str) -> Optional[Tuple[float, float, float]]:
    for ln in pdb_text.splitlines():
        if ln.startswith("CRYST1"):
            return (float(ln[6:15]), float(ln[15:24]), float(ln[24:33]))
    return None


# --------------------------------------------------------------------------- #
# tleap scripts (explicit; rectangular box for a clean box transfer to the
# mutant -- solvateBox not solvateOct, so `set mut box {x y z}` is exact)
# --------------------------------------------------------------------------- #
def _tleap_solvate(ligand: bool, buffer_A: float = 12.0) -> str:
    """Solvate the WT and save ONLY the solvated structure. Both end states are
    then built from this one structure via loadpdb (_tleap_from_pdb), so their
    common atoms share byte-identical coordinates -- the prerequisite for
    softcore_setup's coordinate matching (building WT via the solvate path and
    the mutant via loadpdb gave 522 'mismatched' atoms)."""
    lig = ("source leaprc.gaff2\nloadamberparams ligand.frcmod\n"
           "LIG = loadmol2 ligand.mol2\n") if ligand else ""
    comb = "sys = combine {prot LIG}\n" if ligand else "sys = prot\n"
    return (
        "source leaprc.protein.ff19SB\nsource leaprc.water.opc\n"
        f"{lig}prot = loadpdb wt_clean.pdb\n{comb}"
        f"solvateBox sys OPCBOX {buffer_A}\n"
        "addIons sys Na+ 0\naddIons sys Cl- 0\n"
        "savepdb sys wt_solv.pdb\nquit\n")


def _tleap_from_pdb(pdb_name: str, tag: str, box: Tuple[float, float, float],
                    ligand: bool) -> str:
    """Build a prmtop from an already-solvated structure via loadpdb, restoring
    the rectangular box. Used for BOTH end states (identical path)."""
    # The LIG residue TEMPLATE (loadmol2) must be loaded here too, not just in
    # _tleap_solvate: loadpdb provides only coordinates, so without the unit the
    # complex's LIG atoms have no gaff2 types and tleap FATALs ("Atom ... does
    # not have a type"). loadmol2 defines the types; loadpdb supplies the (solvated)
    # coordinates -- consistent because wt_solv.pdb's LIG came from this same mol2.
    lig = ("source leaprc.gaff2\nloadamberparams ligand.frcmod\n"
           "LIG = loadmol2 ligand.mol2\n") if ligand else ""
    a, b, c = box
    return (
        "source leaprc.protein.ff19SB\nsource leaprc.water.opc\n"
        f"{lig}sys = loadpdb {pdb_name}\n"
        f"set sys box {{ {a:.3f} {b:.3f} {c:.3f} }}\n"
        f"saveamberparm sys {tag}.prmtop {tag}.rst\nquit\n")


# --------------------------------------------------------------------------- #
# TI mdin generators (unified softcore; explicit NVT). All stages carry the TI
# block because the hybrid has BOTH sidechains overlapping -- without softcore a
# plain minimiser would explode on the clashing dual atoms.
# --------------------------------------------------------------------------- #
def _ti_block(m: TIMasks, clambda: float) -> str:
    # one mask per line: pmemd's mdin reader truncates at 80 columns, so packing
    # two long masks on a line silently corrupts the namelist (-> I/O crash).
    #
    # GTI smoothed softcore (gti_*) + scalpha/scbeta: the bare ifsc=1 softcore
    # diverges at endpoint windows for a large perturbation -- e.g. Ser->Gly loses
    # the WHOLE side chain, so the disappearing-atom dV/dlambda overflows the mdout
    # field (SC_VDW_DER=********), the window crashes, and (pre-fix) one such window
    # killed the whole RBFE stage. The GTI softcore (gti_add_sc + a softcore-specific
    # cutoff) and the standard scalpha=0.5/scbeta=12 keep dV/dlambda finite at lambda
    # ~0/1. cut_sc_off must be <= the nonbonded cut (10.0).
    return (
        f" icfe=1, ifsc=1, clambda={clambda:.5f},\n"
        f" timask1='{m.timask1}',\n"
        f" timask2='{m.timask2}',\n"
        f" scmask1='{m.scmask1}',\n"
        f" scmask2='{m.scmask2}',\n"
        " scalpha=0.5, scbeta=12.0,\n"
        " gti_cut=1, gti_output=1, gti_add_sc=25, gti_scale_beta=1,\n"
        " gti_cut_sc_on=8.0, gti_cut_sc_off=10.0,\n"
        " gti_lam_sch=1, gti_ele_sc=1, gti_vdw_sc=1, gti_cut_sc=2,\n"
        f" ntf=1, noshakemask='{m.noshakemask}',\n")


def mdin_ti_min(m: TIMasks, clambda: float, maxcyc: int = 5000,
                restraint_wt: float = 5.0) -> str:
    # ntmin=2 (pure steepest descent) is REQUIRED with ifsc=1 -- the conjugate
    # gradient switch (ncyc) is rejected for softcore minimisation. Backbone
    # restraints keep the system from distorting while the dual sidechains relax.
    return ("TI min (softcore SD, restrained backbone)\n&cntrl\n"
            f" imin=1, maxcyc={maxcyc}, ntmin=2,\n"
            " ntb=1, cut=10.0, ntc=2,\n"
            f" ntr=1, restraintmask='@CA,C,N,O', restraint_wt={restraint_wt},\n"
            f"{_ti_block(m, clambda)}"
            " ntpr=500,\n/\n")


def mdin_ti_heat(m: TIMasks, clambda: float, nsteps: int,
                 restraint_wt: float = 5.0, dt: float = 0.001,
                 gamma_ln: float = 5.0) -> str:
    # gentle restrained heat 5->300K: without backbone restraints the freshly
    # built softcore region distorts during heating and the window blows up.
    # dt/gamma_ln are tunable so a window that blew up can be retried with a
    # smaller timestep + stronger thermostat (run_leg retry path).
    return ("TI heat 5->300K (softcore, restrained, NVT)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt={dt}, irest=0, ntx=1,\n"
            " ntb=1, cut=10.0, iwrap=1, ntc=2,\n"
            f" ntt=3, gamma_ln={gamma_ln}, tempi=5.0, temp0=300.0, ig=-1,\n"
            f" ntr=1, restraintmask='@CA,C,N,O', restraint_wt={restraint_wt},\n"
            f"{_ti_block(m, clambda)}"
            " nmropt=1, ntpr=500,\n/\n"
            "&wt type='TEMP0', istep1=0, istep2=%d, value1=5.0, value2=300.0 /\n"
            "&wt type='END' /\n" % int(nsteps * 0.8))


def mdin_ti_prod(m: TIMasks, clambda: float, nsteps: int,
                 ntpr: int = 500, dt: float = 0.001,
                 gamma_ln: float = 2.0) -> str:
    """TI production: dV/dλ is written to the mdout every ntpr steps. dt/gamma_ln
    tunable for the retry path (a charge-clash runaway at an intermediate λ is
    tamed by a smaller dt + a stronger Langevin thermostat that drains the heat)."""
    return ("TI production (softcore, NVT)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt={dt}, irest=1, ntx=5,\n"
            " ntb=1, cut=10.0, iwrap=1, ntc=2,\n"
            f" ntt=3, gamma_ln={gamma_ln}, temp0=300.0, ig=-1,\n"
            f"{_ti_block(m, clambda)}"
            f" ntpr={ntpr}, ntwx=0, ntwr={nsteps},\n/\n")


def gauss_legendre_01(n: int = 9) -> List[Tuple[float, float]]:
    """n-point Gauss-Legendre (λ, weight) quadrature on [0,1] -- the standard
    Amber TI schedule. Nodes are strictly INTERIOR (never the 0/1 endpoints,
    which are softcore-unstable and crash mid-run) and the weights sum to 1, so
    ΔG = Σ wᵢ ⟨dV/dλ⟩ᵢ exactly for a smooth integrand."""
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(int(n))
    return [(round(float((xi + 1) / 2), 5), float(wi / 2))
            for xi, wi in zip(x, w)]


def lambda_schedule(n: int = 9) -> List[float]:
    """The λ nodes of the n-point Gauss-Legendre TI schedule (interior)."""
    return [lam for lam, _ in gauss_legendre_01(n)]


# --------------------------------------------------------------------------- #
# Hybrid topology builder
# --------------------------------------------------------------------------- #
def build_hybrid(clean_pdb: Path, mutation: Mutation, workdir: Path,
                 ligand: bool = False, buffer_A: float = 12.0,
                 timeout: int = 1800) -> Tuple[Path, Path, TIMasks]:
    """Solvate the (pre-cleaned) WT, swap the residue keeping the waters, build
    both end-state prmtops, then softcore_setup -> hybrid prmtop + TI masks.
    `ligand` True expects ligand.mol2/ligand.frcmod already in workdir (complex
    leg). Returns (hybrid_prmtop, hybrid_rst, masks)."""
    workdir.mkdir(parents=True, exist_ok=True)
    # protein-only WT: drop any ligand/HETATM/water (the complex leg re-adds the
    # ligand from ligand.mol2; a bare LIG with no type FATALs tleap otherwise).
    prot_only = "\n".join(
        l for l in Path(clean_pdb).read_text().splitlines()
        if not l.startswith(("ATOM", "HETATM")) or l[17:20].strip() in _STD_AA)
    (workdir / "wt_clean.pdb").write_text(prot_only + "\n")

    # 1. solvate WT -> SOLVATED STRUCTURE only (both end states derive from it)
    (workdir / "tleap_solv.in").write_text(_tleap_solvate(ligand, buffer_A))
    _sh(["tleap", "-s", "-f", "tleap_solv.in"], workdir, timeout)
    wt_solv = workdir / "wt_solv.pdb"
    if not wt_solv.exists():
        raise RuntimeError(f"WT solvation failed (see {workdir}/leap.log)")
    box = _cryst1_box(wt_solv.read_text())
    if box is None:
        raise RuntimeError("no CRYST1 box in wt_solv.pdb")

    # 2. WT prmtop via loadpdb (the SAME path the mutant uses -> the common
    #    region gets byte-identical coordinates, so softcore_setup matches)
    (workdir / "tleap_wt.in").write_text(
        _tleap_from_pdb("wt_solv.pdb", "wt", box, ligand))
    _sh(["tleap", "-s", "-f", "tleap_wt.in"], workdir, timeout)
    if not (workdir / "wt.prmtop").exists():
        raise RuntimeError(f"WT prmtop build failed (see {workdir}/leap.log)")

    # 3. mutate the solvated WT (every water kept) -> mutant solvated structure
    (workdir / "mut_solv.pdb").write_text(
        mutate_structure(wt_solv.read_text(), mutation.resid,
                         mutation.mut_resname, mutation.chain))

    # 4. mutant prmtop via the SAME loadpdb path + identical box
    (workdir / "tleap_mut.in").write_text(
        _tleap_from_pdb("mut_solv.pdb", "mut", box, ligand))
    _sh(["tleap", "-s", "-f", "tleap_mut.in"], workdir, timeout)
    if not (workdir / "mut.prmtop").exists():
        raise RuntimeError(f"mutant build failed (see {workdir}/leap.log)")

    # 5. hybrid topology + masks
    r = _sh([sys.executable, str(_SOFTCORE), "wt.prmtop", "wt.rst",
             "mut.prmtop", "mut.rst"], workdir, 600)
    if r.returncode != 0:
        raise RuntimeError(
            f"softcore_setup failed (rc={r.returncode}). Output:\n"
            f"{((r.stdout or '') + (r.stderr or ''))[-800:]}")
    masks = _parse_softcore_masks((r.stdout or "") + (r.stderr or ""))
    if not (workdir / "mut.SC.prmtop").exists():
        raise RuntimeError(
            f"softcore_setup produced no hybrid prmtop. masks={masks} "
            f"out:\n{(r.stdout or '')[-600:]}")

    # 6. launder through parmed: softcore_setup's Fortran writer is slightly off
    #    pmemd's strict format (parmed parses the result fine, but pmemd
    #    I/O-crashes mid-prmtop), so re-emit both files with parmed's compliant
    #    writer. Atom order is preserved, so the masks stay valid.
    (workdir / "launder.parmed").write_text(
        "outparm mut.SC.fixed.prmtop mut.SC.fixed.rst\n")
    _sh(["parmed", "-O", "-p", "mut.SC.prmtop", "-c", "mut.SC.rst",
         "-i", "launder.parmed"], workdir, 300)
    hyb_prm = workdir / "mut.SC.fixed.prmtop"
    hyb_rst = workdir / "mut.SC.fixed.rst"
    if not hyb_prm.exists() or not hyb_rst.exists():
        raise RuntimeError(f"parmed launder of the hybrid failed (see {workdir})")
    return hyb_prm, hyb_rst, masks


# --------------------------------------------------------------------------- #
# Lambda-window orchestration + TI integration (one alchemical leg)
# --------------------------------------------------------------------------- #
def _parse_dvdl(out_path: Path) -> List[float]:
    vals: List[float] = []
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            m = re.search(r"DV/DL\s*=\s*(-?\d+\.\d+)", line)
            if m:
                vals.append(float(m.group(1)))
    return vals


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _trapz(ys: Sequence[float], xs: Sequence[float]) -> float:
    return sum(0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
               for i in range(len(xs) - 1))


def _dvdl_finite(out_path: Path) -> bool:
    """True iff the prod mdout has dV/dλ samples and NONE overflowed. A charge-clash
    runaway prints 'SC_VDW_DER=********' (field overflow) -- pmemd may still write a
    (garbage) restart, so 'prod.rst exists' is NOT sufficient; the mdout must be clean."""
    if not out_path.exists():
        return False
    txt = out_path.read_text()
    if "********" in txt:          # any overflowed energy/derivative field
        return False
    return bool(re.search(r"DV/DL\s*=\s*-?\d+\.\d+", txt))


def _run_window(wd: Path, prm: str, start_rst: str, masks: TIMasks, lam: float,
                pmemd: str, min_cyc: int, heat_steps: int, prod_steps: int,
                ntpr: int, timeout: int, dt: float = 0.001,
                prod_gamma_ln: float = 2.0) -> Optional[List[float]]:
    """One λ window: min -> heat -> production (all at λ) -> dV/dλ samples. dt +
    prod_gamma_ln are tunable so run_leg can RETRY a blown-up window with a smaller
    timestep + stronger thermostat. Returns None if any stage fails OR the prod
    overflowed (so the retry path triggers)."""
    wd.mkdir(parents=True, exist_ok=True)

    def stage(tag, mdin, cin, rout, oout=None, ref=None):
        (wd / f"{tag}.in").write_text(mdin)
        cmd = [pmemd, "-O", "-i", f"{tag}.in", "-o", oout or f"{tag}.out",
               "-p", prm, "-c", cin, "-r", rout]
        if ref:                       # ntr=1 needs a restraint reference
            cmd += ["-ref", ref]
        _sh(cmd, wd, timeout)
        return wd / rout if (wd / rout).exists() else None

    mn = stage("min", mdin_ti_min(masks, lam, min_cyc), start_rst, "min.rst",
               ref=start_rst)
    if not mn:
        return None
    ht = stage("heat", mdin_ti_heat(masks, lam, heat_steps, dt=dt), str(mn),
               "heat.rst", ref=start_rst)
    if not ht:
        return None
    stage("prod", mdin_ti_prod(masks, lam, prod_steps, ntpr, dt=dt,
                               gamma_ln=prod_gamma_ln), str(ht), "prod.rst", "prod.out")
    if not _dvdl_finite(wd / "prod.out"):    # crashed OR overflowed -> let run_leg retry
        return None
    return _parse_dvdl(wd / "prod.out")


def run_leg(hyb_prm: Path, hyb_rst: Path, masks: TIMasks, workdir: Path,
            n_lambda: int = 9, min_cyc: int = 2000, heat_steps: int = 10000,
            prod_steps: int = 50000, ntpr: int = 500, equil_frac: float = 0.2,
            pmemd: Optional[str] = None, timeout: int = 3600) -> Dict:
    """Run every Gauss-Legendre λ window of one alchemical leg and integrate
    ⟨dV/dλ⟩ -> ΔG (Σ wᵢ ⟨dV/dλ⟩ᵢ; trapezoidal fallback over the survivors if a
    window dies). Each window does min->heat->prod independently from the shared
    hybrid restart (each λ equilibrates at its own state). Returns
    {dg, method, lambdas, dvdl_means, n_fail}. MBAR via ndfes/edgembar is a
    later refinement; Gauss-Legendre TI is the v1 estimator."""
    pmemd = pmemd or _pmemd_cuda()
    if not pmemd:
        raise RuntimeError("no pmemd.cuda (set EVOLIEZ_PMEMD_CUDA)")
    prm = str(Path(hyb_prm).resolve())
    start = str(Path(hyb_rst).resolve())
    results: List[Tuple[float, float, float]] = []     # (λ, weight, ⟨dV/dλ⟩)
    n_fail = 0
    n_retry = 0
    for lam, w in gauss_legendre_01(n_lambda):
        wdir = workdir / f"lam_{lam:.4f}"
        dvdl = _run_window(wdir, prm, start, masks, lam, pmemd,
                           min_cyc, heat_steps, prod_steps, ntpr, timeout)
        if not dvdl:
            # Salvage retry: a window that crashed/overflowed at dt=0.001 (a charge-
            # clash runaway mid-production) is re-run with a smaller timestep + a
            # stronger thermostat + 2x heat/prod steps (same simulated time at half dt).
            # Only the FEW hard windows pay this; the easy ones keep the fast path.
            n_retry += 1
            dvdl = _run_window(wdir, prm, start, masks, lam, pmemd,
                               min_cyc, heat_steps * 2, prod_steps * 2, ntpr,
                               timeout, dt=0.0005, prod_gamma_ln=5.0)
        if not dvdl:
            n_fail += 1
            continue
        eq = dvdl[int(len(dvdl) * equil_frac):] or dvdl   # drop equilibration
        results.append((lam, w, _mean(eq)))
    # Quality gate: a ΔG integrated over <60% of the λ schedule is unreliable (the
    # original failure produced a spurious ddg=-12 kcal/mol from a handful of noisy
    # survivors). Below the gate, report failure (dg=None) -- graceful, so run_rbfe
    # records this candidate's ddg as unavailable and CONTINUES with the rest, rather
    # than (a) raising and aborting the stage or (b) emitting a garbage number.
    min_ok = max(2, int(round(n_lambda * 0.6)))
    if len(results) < min_ok:
        return {"dg": None, "method": "failed",
                "lambdas": [r[0] for r in results],
                "dvdl_means": [r[2] for r in results], "n_fail": n_fail,
                "n_retry": n_retry,
                "failed": f"only {len(results)}/{n_lambda} λ windows converged "
                          f"({n_fail} failed, {n_retry} retried)"}
    lams = [r[0] for r in results]
    means = [r[2] for r in results]
    if n_fail == 0:
        dg, method = sum(w * m for _, w, m in results), "gauss"
    else:                                  # quadrature weights invalid -> trapz
        dg, method = _trapz(means, lams), "trapz-fallback"
    return {"dg": dg, "method": method, "lambdas": lams,
            "dvdl_means": means, "n_fail": n_fail, "n_retry": n_retry}


# --------------------------------------------------------------------------- #
# Full RBFE orchestration: ΔΔG_bind(WT->mutant) via the double-leg cycle
# --------------------------------------------------------------------------- #
_AA1to3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
           "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
           "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
           "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}


def run_rbfe(wt_pdb_text: str, ligand_mol2, ligand_frcmod, mutations,
             workdir: Path, *, multipoint: str = "additive", n_lambda: int = 9,
             min_cyc: int = 2000, heat_steps: int = 10000, prod_steps: int = 50000,
             ntpr: int = 500, pmemd: Optional[str] = None,
             timeout: int = 3600) -> Dict:
    """ΔΔG_bind(WT->mutant) for the design ligand via the thermodynamic cycle
    ΔΔG = ΔG_complex(X->Y) - ΔG_apo(X->Y), run per mutated residue with softcore TI.

    ``mutations``: the candidate's ``types.Mutation`` list (1-letter wt/mut +
    position). Multi-point candidates: ``multipoint='additive'`` sums the
    single-residue cycles (coupling-free APPROXIMATION, flagged in the result) or
    ``'skip'`` returns ddg_bind=None. Mutations to PRO are skipped (proline ring is
    unsupported by the hybrid builder). Returns
    {ddg_bind, per_mutation:[{mutation,ddg_bind,dg_complex,dg_apo,...}], mode, n_mut}.
    EXPENSIVE (2 legs x n_lambda windows x min+heat+prod per residue) -- gate behind
    md.rbfe.enabled + top_n upstream."""
    workdir = Path(workdir)
    muts = [m for m in mutations if getattr(m, "mut", "") in _AA1to3]
    unsupported = [f"{m.wt}{m.position}{m.mut}" for m in mutations
                   if getattr(m, "mut", "") == "P" or getattr(m, "mut", "") not in _AA1to3]
    if not muts:
        return {"ddg_bind": None, "skipped": f"no alchemy-supported mutation "
                f"({unsupported})", "n_mut": 0}
    muts = [m for m in muts if m.mut != "P"]
    if not muts:
        return {"ddg_bind": None, "skipped": "only PRO mutations (unsupported)",
                "n_mut": 0}
    if len(muts) > 1 and multipoint == "skip":
        return {"ddg_bind": None, "skipped": f"multi-point ({len(muts)})",
                "n_mut": len(muts)}
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "wt_input.pdb").write_text(wt_pdb_text)
    per: List[Dict] = []
    total = 0.0
    for m in muts:
        amut = Mutation(resid=int(m.position), wt_resname=_AA1to3.get(m.wt, m.wt),
                        mut_resname=_AA1to3[m.mut])
        tag = str(amut)
        legs = {}
        for leg, ligand in (("complex", True), ("apo", False)):
            ld = workdir / tag / leg
            ld.mkdir(parents=True, exist_ok=True)
            if ligand:                         # complex leg needs the ligand params
                (ld / "ligand.mol2").write_text(Path(ligand_mol2).read_text())
                (ld / "ligand.frcmod").write_text(Path(ligand_frcmod).read_text())
            hyb_prm, hyb_rst, masks = build_hybrid(
                workdir / "wt_input.pdb", amut, ld, ligand=ligand, timeout=timeout)
            legs[leg] = run_leg(
                hyb_prm, hyb_rst, masks, ld / "ti", n_lambda=n_lambda,
                min_cyc=min_cyc, heat_steps=heat_steps, prod_steps=prod_steps,
                ntpr=ntpr, pmemd=pmemd, timeout=timeout)
        dgc, dga = legs["complex"]["dg"], legs["apo"]["dg"]
        nfail = legs["complex"].get("n_fail", 0) + legs["apo"].get("n_fail", 0)
        if dgc is None or dga is None:     # a leg lost too many windows -> no ΔΔG
            per.append({
                "mutation": tag, "ddg_bind": None, "dg_complex": dgc,
                "dg_apo": dga, "method": "failed", "n_fail": nfail,
                "failed": legs["complex"].get("failed") or legs["apo"].get("failed")})
            total = None                    # additive sum is undefined if any leg died
            continue
        ddg = dgc - dga
        per.append({
            "mutation": tag, "ddg_bind": round(ddg, 3),
            "dg_complex": round(dgc, 3), "dg_apo": round(dga, 3),
            "method": legs["complex"]["method"], "n_fail": nfail})
        if total is not None:
            total += ddg
    n_ok = sum(1 for x in per if x.get("ddg_bind") is not None)
    return {"ddg_bind": (round(total, 3) if total is not None else None),
            "per_mutation": per,
            "mode": "single" if len(muts) == 1 else f"additive_x{len(muts)}",
            "n_mut": len(muts),
            "failed": (None if total is not None else
                       f"{len(muts) - n_ok}/{len(muts)} residue cycle(s) failed"),
            "note": ("additive multi-residue approximation (no coupling)"
                     if len(muts) > 1 else "")}
