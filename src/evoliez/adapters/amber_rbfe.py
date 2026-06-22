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
        return self.timask1.split("@", 1)[0]   # ':188'


def _union_mask(m1: str, m2: str) -> str:
    """':R@A,B' ∪ ':R@C,D' -> ':R@A,B,C,D' (single-residue mutation)."""
    r1, a1 = m1.split("@", 1)
    r2, a2 = m2.split("@", 1)
    atoms = a1.split(",") + [a for a in a2.split(",") if a not in a1.split(",")]
    res = r1 if r1 == r2 else f"{r1}|{r2}"   # same residue in practice
    return f"{r1}@{','.join(atoms)}" if r1 == r2 else f"({m1})|({m2})"


def _parse_softcore_masks(text: str) -> TIMasks:
    """Pull the two 'Alchemy step' scmasks softcore_setup prints: the first is
    the WT-unique region (for wt.prmtop), the second the mutant-unique region
    (for mut.SC.prmtop)."""
    sc = re.findall(r"scmask\s*=\s*':([^']*)'", text)
    if len(sc) < 2:
        raise RuntimeError(
            f"softcore_setup did not emit two scmasks (got {sc}). Output:\n{text[-800:]}")
    return TIMasks(timask1=f":{sc[0]}", timask2=f":{sc[1]}")


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
    lig = ("source leaprc.gaff2\nloadamberparams ligand.frcmod\n") if ligand else ""
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
    return (
        f" icfe=1, ifsc=1, clambda={clambda:.5f},\n"
        f" timask1='{m.timask1}',\n"
        f" timask2='{m.timask2}',\n"
        f" scmask1='{m.scmask1}',\n"
        f" scmask2='{m.scmask2}',\n"
        f" ntf=1, noshakemask='{m.noshakemask}',\n")


def mdin_ti_min(m: TIMasks, clambda: float, maxcyc: int = 5000) -> str:
    # ntmin=2 (pure steepest descent) is REQUIRED with ifsc=1 -- the conjugate
    # gradient switch (ncyc) is rejected for softcore minimisation.
    return ("TI min (softcore, steepest descent)\n&cntrl\n"
            f" imin=1, maxcyc={maxcyc}, ntmin=2,\n"
            " ntb=1, cut=10.0, ntc=2,\n"
            f"{_ti_block(m, clambda)}"
            " ntpr=500,\n/\n")


def mdin_ti_heat(m: TIMasks, clambda: float, nsteps: int) -> str:
    return ("TI heat 0->300K (softcore, NVT)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt=0.001, irest=0, ntx=1,\n"
            " ntb=1, cut=10.0, iwrap=1, ntc=2,\n"
            " ntt=3, gamma_ln=2.0, tempi=5.0, temp0=300.0, ig=-1,\n"
            f"{_ti_block(m, clambda)}"
            " ntpr=500,\n/\n")


def mdin_ti_prod(m: TIMasks, clambda: float, nsteps: int,
                 ntpr: int = 500) -> str:
    """TI production: dV/dλ is written to the mdout every ntpr steps."""
    return ("TI production (softcore, NVT)\n&cntrl\n"
            f" imin=0, nstlim={nsteps}, dt=0.001, irest=1, ntx=5,\n"
            " ntb=1, cut=10.0, iwrap=1, ntc=2,\n"
            " ntt=3, gamma_ln=2.0, temp0=300.0, ig=-1,\n"
            f"{_ti_block(m, clambda)}"
            f" ntpr={ntpr}, ntwx=0, ntwr={nsteps},\n/\n")


def lambda_schedule(n: int = 11) -> List[float]:
    """Evenly-spaced λ windows in (0,1). Avoids the exact 0/1 endpoints where
    even softcore can be stiff; uses Gauss-friendly interior spacing for n>=9."""
    if n < 3:
        raise ValueError("need >=3 lambda windows")
    return [round((i + 0.5) / n, 5) for i in range(n)] if n < 9 else \
        [round(i / (n - 1), 5) for i in range(1, n - 1)]  # interior 0<λ<1


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
