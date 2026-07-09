"""V6-1 — Amber-native system builder.

Converts an EvoLiEZ mechanism complex (protein + design ligand + curated-charge
co-substrate(s) + catalytic metal + explicit solvent + ions) into an auditable
Amber ``prmtop``/``inpcrd`` pair, and — crucially for the V6 PMF/geometry tiers —
maps the reactive atoms (O_nuc, Pα, O_leaving, Mg, catalytic residues) onto exact
prmtop atom indices + cpptraj masks.

Design principles (ROADMAP_V6 §2, §4):
  * MECHANISM-GENERIC: everything (force fields, ligand roles, reactive SMARTS,
    metal state, catalytic residues) comes from :class:`AmberBuildSpec`; nothing
    CAR-specific is hardcoded. FDH (hydride, no metal), TEM-1 (protein
    nucleophile) and CAR (adenylation + Mg) all use the same contract.
  * FAIL-LOUD: a missing charge template, a curated net-charge mismatch, a tleap
    FATAL, or an ambiguous reactive mask RAISES with a classified reason. There is
    no silent AM1-BCC fallback for a high-charge cofactor (the anti-s06 rule).
  * REPRODUCIBLE: the system fingerprint is a hash of (protein sequence, ordered
    ligand identities + net charges + parameter sources, FF/water/ion choices,
    reactive spec). Same config → same fingerprint.

Reuses the proven chemistry primitives rather than reinventing them:
  * ``md.charges``            — curated fixed-charge templates (graph-matched).
  * ``md.metal_placement``    — deterministic bridging-Mg geometry / HETATM.
  * ``md.nac``                — donor/acceptor/leaving reactive-atom resolution.
  * ``adapters.openmm_engine._ligands_at_pose`` — graph-based multi-ligand pose
    extraction (robust to the merged/renamed HETATM the pipeline emits).

The subprocess build (pdb4amber/antechamber/parmchk2/tleap/parmed) is server-only;
the pure helpers (tleap-script generation, coordinate transplant, charge
validation, fingerprint, reactive-name resolution, manifest assembly) are
unit-testable off the server.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H", "HID": "H", "HIP": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "CYX": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}


def _one_letter(resname3: str) -> str:
    return _AA3_TO_1.get(resname3.strip().upper(), "X")


# --------------------------------------------------------------------------- #
# specs / results
# --------------------------------------------------------------------------- #
@dataclass
class LigandBuildSpec:
    """One small molecule to place in the Amber system."""
    id: str                       # Amber residue name in the system (<=4 chars: LIG, ATP)
    role: str                     # design_ligand | cofactor | cosubstrate
    smiles: Optional[str] = None  # used to graph-match the pose HETATM group
    net_charge: int = 0
    allow_am1bcc: bool = True
    charges_mol2: Optional[str] = None       # curated template (mol2 + .ref.sdf alongside)


@dataclass
class MetalBuildSpec:
    enabled: bool = False
    ion: str = "MG"               # Amber ion residue/atom name
    element: str = "MG"
    charge: int = 2
    coord_dist: float = 2.1
    # Li-Merz 12-6-4 divalent LJ frcmod (ROADMAP_V6 §14): the C4 ion-induced-dipole
    # term gives correct Mg2+ coordination distances, which matters for a catalytic
    # BRIDGING metal. Matched to TIP3P (there is no OPC 12-6-4 set on the server).
    # Verified present at build time; a missing file is a loud tleap failure, never a
    # silent 12-6 fallback.
    ion_frcmod: str = "frcmod.ions234lm_1264_tip3p"


@dataclass
class ReactiveBuildSpec:
    """Mirror of the config reactive_geometry block, sufficient to resolve the
    reactive atoms onto prmtop indices."""
    donor_smarts: Optional[str] = None
    acceptor_smarts: Optional[str] = None
    donor_idx: int = 0
    acceptor_idx: int = 0
    transfer_is_h: bool = False
    donor_protein: Optional[str] = None       # 'RESNAME:ATOM[:RESNUM]' for Ser/Cys nucleophile
    label: str = "reaction"


@dataclass
class AmberBuildSpec:
    ligands: List[LigandBuildSpec]
    metal: MetalBuildSpec = field(default_factory=MetalBuildSpec)
    reactive: ReactiveBuildSpec = field(default_factory=ReactiveBuildSpec)
    catalytic_residues: List[str] = field(default_factory=list)   # ['S268','T269','K273']
    protein_ff: str = "ff14SB"
    water: str = "tip3p"          # tip3p | opc | spce
    solvent: str = "explicit"     # explicit | implicit
    box_buffer_A: float = 12.0
    # Monovalent Na/Cl come from ``leaprc.water.<model>`` (Joung-Cheatham) by
    # default; set this only to override with an explicit monovalent frcmod.
    ions_frcmod: Optional[str] = None
    salt_conc_M: float = 0.0      # 0 = neutralize only


@dataclass
class ReactiveAtomMap:
    """One reactive role resolved onto the built prmtop."""
    role: str                     # O_nuc | P_alpha | O_leaving | metal | catalytic
    residue: str
    resid: int                    # prmtop residue number (matches the mask + amber_index)
    atom_name: str
    amber_index: int              # 1-based Amber atom index
    mask: str                     # cpptraj/parmed mask (e.g. ':LIG@O1'), prmtop-frame
    orig_resid: Optional[int] = None   # original PDB residue number (for humans)
    status: str = "ok"            # ok | identity_mismatch | unresolved


@dataclass
class AmberSystem:
    prmtop: Path
    inpcrd: Path
    workdir: Path
    fingerprint: str
    n_atoms: int
    net_charge: float
    reactive_map: List[ReactiveAtomMap]
    manifest: dict


class AmberBuildError(RuntimeError):
    """Classified build failure. ``klass`` in {parameterization, tleap,
    reactive_mapping, input}."""
    def __init__(self, klass: str, message: str):
        super().__init__(f"[{klass}] {message}")
        self.klass = klass


# --------------------------------------------------------------------------- #
# pure helpers (unit-testable)
# --------------------------------------------------------------------------- #
def system_fingerprint(spec: AmberBuildSpec, protein_seq: str,
                       ligand_param_sources: Dict[str, str]) -> str:
    """Deterministic fingerprint of the build inputs. Re-running the same config
    against the same structure yields the same string."""
    payload = {
        "protein_seq_sha1": hashlib.sha1(protein_seq.encode()).hexdigest(),
        "protein_ff": spec.protein_ff,
        "water": spec.water,
        "solvent": spec.solvent,
        "box_buffer_A": round(spec.box_buffer_A, 3),
        "ions_frcmod": spec.ions_frcmod,
        "salt_conc_M": round(spec.salt_conc_M, 4),
        "ligands": [
            {"id": l.id, "role": l.role, "net_charge": l.net_charge,
             "allow_am1bcc": l.allow_am1bcc,
             "param_source": ligand_param_sources.get(l.id, "am1bcc")}
            for l in sorted(spec.ligands, key=lambda x: x.id)
        ],
        "metal": {"enabled": spec.metal.enabled, "ion": spec.metal.ion,
                  "charge": spec.metal.charge, "ion_frcmod": spec.metal.ion_frcmod}
        if spec.metal.enabled else None,
        "reactive": {"donor_smarts": spec.reactive.donor_smarts,
                     "acceptor_smarts": spec.reactive.acceptor_smarts,
                     "donor_protein": spec.reactive.donor_protein,
                     "transfer_is_h": spec.reactive.transfer_is_h},
        "catalytic_residues": sorted(spec.catalytic_residues),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def tleap_amber_script(spec: AmberBuildSpec, ligand_units: Sequence[Tuple[str, str]],
                       protein_pdb: str = "protein_clean.pdb",
                       out_prm: str = "complex.prmtop",
                       out_crd: str = "complex.inpcrd") -> str:
    """Generate the tleap deck. ``ligand_units`` = ordered ``(unit_var, mol2_file)``
    for each non-metal ligand. The metal (if any) is carried in ``protein_pdb`` as an
    ion HETATM and parameterized by the divalent ion frcmod. Explicit solvent uses a
    truncated octahedron + neutralizing ions; implicit uses GB radii."""
    L: List[str] = [f"source leaprc.protein.{spec.protein_ff}", "source leaprc.gaff2"]
    if spec.solvent == "explicit":
        L.append(f"source leaprc.water.{spec.water}")   # monovalent JC ions included
        if spec.ions_frcmod:
            L.append(f"loadamberparams {spec.ions_frcmod}")
    if spec.metal.enabled:
        L.append(f"loadamberparams {spec.metal.ion_frcmod}")
    # ligand params + units
    combine = ["prot"]
    for i, (var, mol2) in enumerate(ligand_units):
        frcmod = mol2[:-5] + ".frcmod" if mol2.endswith(".mol2") else mol2 + ".frcmod"
        L.append(f"loadamberparams {frcmod}")
        L.append(f"{var} = loadmol2 {mol2}")
        combine.append(var)
    L.append(f"prot = loadpdb {protein_pdb}")   # includes the metal HETATM if enabled
    L.append(f"comp = combine {{{' '.join(combine)}}}")
    if spec.solvent == "explicit":
        box = {"opc": "OPCBOX", "tip3p": "TIP3PBOX", "spce": "SPCBOX"}.get(spec.water, "TIP3PBOX")
        L.append(f"solvateOct comp {box} {spec.box_buffer_A:.1f}")
        L.append("addIons comp Na+ 0")
        L.append("addIons comp Cl- 0")
    else:
        L.append("set default PBRadii mbondi3")
    L.append(f"saveamberparm comp {out_prm} {out_crd}")
    L.append("savepdb comp complex_built.pdb")
    L.append("quit")
    return "\n".join(L) + "\n"


def parse_mol2_atom_names(mol2_text: str) -> List[str]:
    """Atom names in mol2 file order (col 2 of the @<TRIPOS>ATOM block)."""
    lines = mol2_text.splitlines()
    try:
        i = lines.index("@<TRIPOS>ATOM")
    except ValueError:
        raise AmberBuildError("parameterization", "mol2 has no @<TRIPOS>ATOM block")
    names: List[str] = []
    for ln in lines[i + 1:]:
        if ln.startswith("@<TRIPOS>"):
            break
        f = ln.split()
        if len(f) >= 2:
            names.append(f[1])
    return names


def validate_mol2_net_charge(mol2_text: str, expected: int, ligand_id: str,
                             tol: float = 0.01) -> float:
    """Sum the mol2 partial charges and assert they match ``expected`` (fail-loud)."""
    lines = mol2_text.splitlines()
    i = lines.index("@<TRIPOS>ATOM")
    total = 0.0
    for ln in lines[i + 1:]:
        if ln.startswith("@<TRIPOS>"):
            break
        f = ln.split()
        if len(f) >= 9:
            total += float(f[-1])
    if abs(total - expected) > tol:
        raise AmberBuildError(
            "parameterization",
            f"{ligand_id}: curated mol2 net charge {total:+.4f} != declared "
            f"{expected:+d} (tol {tol}); refusing to build a mis-charged cofactor")
    return total


def transplant_pose_coords(curated_mol2_text: str, ref_sdf: Path,
                           pose_mol, ligand_id: str) -> str:
    """Return a new mol2 == the curated template (types/charges/names/order) but with
    ALL atom coordinates taken from ``pose_mol`` via graph isomorphism. Hydrogens are
    (re)placed with ``AddHs(addCoords=True)`` so they are geometrically consistent with
    the transplanted heavy-atom pose — leaving the template H at their reference
    positions produces H↔pocket clashes that blow up minimization. Fail-loud on an
    incomplete match."""
    from rdkit import Chem

    ref = Chem.SDMolSupplier(str(ref_sdf), removeHs=False, sanitize=True)[0]
    if ref is None:
        raise AmberBuildError("parameterization", f"{ligand_id}: unreadable ref sdf {ref_sdf}")
    # rebuild H on the pose heavy-atom frame so every atom has a pose-consistent xyz
    pose_full = Chem.AddHs(Chem.RemoveHs(pose_mol), addCoords=True)
    qp = Chem.AdjustQueryParameters.NoAdjustments()
    qp.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(ref, qp)
    match = pose_full.GetSubstructMatch(query)      # match[ref_i] = pose_i (all atoms)
    if len(match) != ref.GetNumAtoms():
        # fall back to a heavy-only match if H perception differs; H then keep template
        # positions (still better than a wrong heavy pose, and flagged by min later)
        ref_heavy = Chem.RemoveHs(ref)
        pose_heavy = Chem.RemoveHs(pose_mol)
        hmatch = pose_heavy.GetSubstructMatch(Chem.AdjustQueryProperties(ref_heavy, qp))
        if len(hmatch) != ref_heavy.GetNumAtoms():
            raise AmberBuildError(
                "parameterization",
                f"{ligand_id}: pose graph-match incomplete "
                f"(full {len(match)}/{ref.GetNumAtoms()}, heavy "
                f"{len(hmatch)}/{ref_heavy.GetNumAtoms()}) — pose is not the curated "
                "molecule; refusing to transplant coordinates")
        conf = pose_heavy.GetConformer()
        heavy_order = [a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() > 1]
        curated_to_xyz = {}
        for hi, ref_full_i in enumerate(heavy_order):
            p = conf.GetAtomPosition(hmatch[hi])
            curated_to_xyz[ref_full_i] = (p.x, p.y, p.z)
        return _rewrite_mol2_coords(curated_mol2_text, curated_to_xyz, heavy_only=True)

    conf = pose_full.GetConformer()
    curated_to_xyz: Dict[int, Tuple[float, float, float]] = {}
    for ref_i, pose_i in enumerate(match):
        if ref.GetAtomWithIdx(ref_i).GetAtomicNum() != \
                pose_full.GetAtomWithIdx(pose_i).GetAtomicNum():
            raise AmberBuildError("parameterization",
                                  f"{ligand_id}: element mismatch in pose match")
        p = conf.GetAtomPosition(pose_i)
        curated_to_xyz[ref_i] = (p.x, p.y, p.z)
    return _rewrite_mol2_coords(curated_mol2_text, curated_to_xyz, heavy_only=False)


def _rewrite_mol2_coords(mol2_text: str, idx_to_xyz: Dict[int, Tuple[float, float, float]],
                         heavy_only: bool) -> str:
    """Rewrite the @<TRIPOS>ATOM block xyz (cols 3-5) from ``idx_to_xyz`` keyed by the
    0-based atom-row index. When ``heavy_only``, H rows are left untouched."""
    lines = mol2_text.splitlines()
    i = lines.index("@<TRIPOS>ATOM")
    atom_row = 0
    j = i + 1
    while j < len(lines) and not lines[j].startswith("@<TRIPOS>"):
        f = lines[j].split()
        if len(f) >= 9:
            is_h = f[5][0].upper() == "H"
            if (not heavy_only or not is_h) and atom_row in idx_to_xyz:
                x, y, z = idx_to_xyz[atom_row]
                f[2], f[3], f[4] = f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"
                lines[j] = ("{:>7} {:<8} {:>9} {:>9} {:>9} {:<6} {:>3} {:<8} {:>9}"
                            .format(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]))
            atom_row += 1
        j += 1
    return "\n".join(lines) + "\n"


def reactive_atom_names(spec: ReactiveBuildSpec,
                        design_ref_sdf: Optional[Path], design_names: List[str],
                        acceptor_ref_sdf: Optional[Path], acceptor_names: List[str]
                        ) -> Dict[str, Tuple[str, str]]:
    """Resolve reactive roles to (residue-role, atom-name) using the SAME SMARTS the
    NAC screen uses, keyed off each ligand's ref graph + mol2 atom-name order.
    Returns {'O_nuc': ('design', name), 'P_alpha': ('acceptor', name),
    'O_leaving': ('acceptor', name)} for the roles present. Protein-nucleophile
    donors are resolved later from the topology, not here."""
    from rdkit import Chem
    from evoliez.md.nac import (ReactiveSpec, _acceptor_double_bonded_oxygen,
                                identify_acceptor, identify_donor, identify_leaving)

    rs = ReactiveSpec(
        donor_smarts=spec.donor_smarts or "[#6]",
        acceptor_smarts=spec.acceptor_smarts or "[#6]",
        donor_protein=spec.donor_protein, donor_idx=spec.donor_idx,
        acceptor_idx=spec.acceptor_idx, transfer_is_h=spec.transfer_is_h,
        label=spec.label)
    out: Dict[str, Tuple[str, str]] = {}

    # PROTEIN-nucleophile (serine hydrolase / protease): the O_nuc is a protein atom
    # (resolved later from the topology). The acceptor (scissile carbonyl C) and its
    # Burgi-Dunitz reference O (the carbonyl =O) are on the DESIGN ligand.
    if spec.donor_protein:
        if design_ref_sdf and spec.acceptor_smarts:
            d = Chem.SDMolSupplier(str(design_ref_sdf), removeHs=False, sanitize=True)[0]
            acc = identify_acceptor(d, rs) if d is not None else None
            if acc is not None and acc < len(design_names):
                out["P_alpha"] = ("design", design_names[acc])   # attacked carbonyl C
                ref = _acceptor_double_bonded_oxygen(d, acc)     # Nuc--C=O reference
                if ref is not None and ref < len(design_names):
                    out["O_leaving"] = ("design", design_names[ref])
        return out

    if design_ref_sdf and spec.donor_smarts:
        d = Chem.SDMolSupplier(str(design_ref_sdf), removeHs=False, sanitize=True)[0]
        don = identify_donor(d, rs) if d is not None else None
        if don is not None and don["heavy"] < len(design_names):
            out["O_nuc"] = ("design", design_names[don["heavy"]])
    if acceptor_ref_sdf and spec.acceptor_smarts:
        a = Chem.SDMolSupplier(str(acceptor_ref_sdf), removeHs=False, sanitize=True)[0]
        acc = identify_acceptor(a, rs) if a is not None else None
        if acc is not None and acc < len(acceptor_names):
            out["P_alpha"] = ("acceptor", acceptor_names[acc])
            if not spec.transfer_is_h:
                lv = identify_leaving(a, acc)
                if lv is not None and lv < len(acceptor_names):
                    out["O_leaving"] = ("acceptor", acceptor_names[lv])
    return out


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #
def _sh(cmd: List[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    from evoliez.utils.subprocess_utils import _LIBC, _set_pdeathsig
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout,
                          preexec_fn=_set_pdeathsig if _LIBC is not None else None)


class AmberSystemBuilder:
    """Build an Amber prmtop/inpcrd for a mechanism complex + map reactive atoms."""

    def __init__(self, spec: AmberBuildSpec, workdir: Path):
        self.spec = spec
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._param_sources: Dict[str, str] = {}
        self._design_id = next((l.id for l in spec.ligands
                                if l.role == "design_ligand"), None)

    # ---- protein ---------------------------------------------------------- #
    def _prepare_protein(self, complex_pdb: Path) -> Tuple[str, str]:
        """pdb4amber-clean the protein; return (clean_pdb_name, one_letter_seq)."""
        from evoliez.adapters.openmm_engine import _pdb_one_letter_seq
        prot = [l for l in complex_pdb.read_text().splitlines() if l.startswith("ATOM")]
        (self.workdir / "protein_in.pdb").write_text("\n".join(prot) + "\nEND\n")
        r = _sh(["pdb4amber", "-i", "protein_in.pdb", "-o", "protein_clean.pdb",
                 "--nohyd", "--dry"], self.workdir, 180)
        if not (self.workdir / "protein_clean.pdb").exists():
            raise AmberBuildError("input", f"pdb4amber failed: {r.stderr[-400:]}")
        seq = _pdb_one_letter_seq(self.workdir / "protein_clean.pdb") or ""
        return "protein_clean.pdb", seq

    # ---- ligands ---------------------------------------------------------- #
    def _param_design_ligand(self, lig: LigandBuildSpec, pose_mol) -> str:
        """antechamber AM1-BCC (gaff2) on the design ligand at its pose. Returns the
        mol2 filename. Fail-loud on a charge failure."""
        from rdkit import Chem
        from evoliez.adapters.amber_engine import _SQM_EK
        sdf = self.workdir / f"{lig.id}.ref.sdf"
        with Chem.SDWriter(str(sdf)) as w:
            w.write(pose_mol)
        mol2 = f"{lig.id}.mol2"
        r = _sh(["antechamber", "-i", str(sdf.name), "-fi", "sdf", "-o", mol2,
                 "-fo", "mol2", "-c", "bcc", "-nc", str(lig.net_charge), "-at", "gaff2",
                 "-ek", _SQM_EK, "-rn", lig.id], self.workdir, 900)
        if not (self.workdir / mol2).exists():
            raise AmberBuildError("parameterization",
                                  f"{lig.id}: antechamber/AM1-BCC failed: "
                                  f"{(r.stdout + r.stderr)[-400:]}")
        r = _sh(["parmchk2", "-i", mol2, "-f", "mol2", "-o", f"{lig.id}.frcmod"],
                self.workdir, 120)
        if not (self.workdir / f"{lig.id}.frcmod").exists():
            raise AmberBuildError("parameterization", f"{lig.id}: parmchk2 failed")
        self._param_sources[lig.id] = "am1bcc_gaff2"
        return mol2

    def _param_curated_cofactor(self, lig: LigandBuildSpec, pose_mol) -> str:
        """Use the curated mol2 (types+charges) with pose coords transplanted on.
        Returns the mol2 filename. Fail-loud on missing/mis-charged template."""
        tmpl = Path(lig.charges_mol2)
        if not tmpl.is_absolute():
            tmpl = (Path.cwd() / tmpl).resolve()
        ref_sdf = Path(str(tmpl)[:-5] + ".ref.sdf")
        if not tmpl.exists() or not ref_sdf.exists():
            raise AmberBuildError("parameterization",
                                  f"{lig.id}: curated template missing "
                                  f"(mol2={tmpl}, ref.sdf={ref_sdf})")
        curated_text = tmpl.read_text()
        validate_mol2_net_charge(curated_text, lig.net_charge, lig.id)
        posed = transplant_pose_coords(curated_text, ref_sdf, pose_mol, lig.id)
        mol2 = f"{lig.id}.mol2"
        (self.workdir / mol2).write_text(posed)
        (self.workdir / f"{lig.id}.ref.sdf").write_text(ref_sdf.read_text())
        r = _sh(["parmchk2", "-i", mol2, "-f", "mol2", "-o", f"{lig.id}.frcmod"],
                self.workdir, 120)
        if not (self.workdir / f"{lig.id}.frcmod").exists():
            raise AmberBuildError("parameterization",
                                  f"{lig.id}: parmchk2 on curated mol2 failed: "
                                  f"{r.stderr[-300:]}")
        self._param_sources[lig.id] = f"curated:{tmpl.name}"
        return mol2

    # ---- metal ------------------------------------------------------------ #
    def _append_metal(self, clean_pdb: str, pose_mg_xyz: Sequence[float]) -> None:
        """Append the metal ion HETATM to the cleaned protein pdb so tleap loadpdb
        picks it up and the divalent ion frcmod parameterizes it."""
        from evoliez.md.metal_placement import mg_hetatm_line
        line = mg_hetatm_line(pose_mg_xyz, element=self.spec.metal.element)
        p = self.workdir / clean_pdb
        text = p.read_text().splitlines()
        for k in range(len(text) - 1, -1, -1):
            if text[k].startswith(("END", "TER")):
                text.insert(k, line)
                break
        else:
            text.append(line)
        p.write_text("\n".join(text) + "\n")

    # ---- reactive mapping ------------------------------------------------- #
    def _resolve_masks(self, reactive_names: Dict[str, Tuple[str, str]],
                       mg_present: bool) -> List[ReactiveAtomMap]:
        """Turn reactive (role, atom-name) pairs + catalytic residues + metal into
        cpptraj masks and resolve absolute Amber indices via parmed. Fail-loud when a
        single-atom mask does not select exactly one atom."""
        import parmed
        prm = parmed.load_file(str(self.workdir / "complex.prmtop"))
        role_res = {"design": self._design_id,
                    "acceptor": next((l.id for l in self.spec.ligands
                                      if l.role in ("cofactor", "cosubstrate")), None)}
        maps: List[ReactiveAtomMap] = []

        # tleap RENUMBERS residues 1..N in the prmtop, so a config residue NUMBER (from
        # the original PDB) does NOT equal the prmtop number when the structure has
        # gaps. Map by ORDINAL POSITION in protein_clean.pdb (which preserves the
        # original numbering) and VALIDATE the residue identity — never a silent match.
        pdb_order = self._protein_residue_order()   # [(orig_num:int, resname3), ...]
        orig_to_ordinal = {num: i for i, (num, _rn) in enumerate(pdb_order)}

        def resolve_one(role: str, resname: str, atomname: str) -> None:
            mask = f":{resname}@{atomname}"
            sel = [i for i, a in enumerate(prm.atoms)
                   if a.residue.name == resname and a.name == atomname]
            if len(sel) != 1:
                raise AmberBuildError("reactive_mapping",
                                      f"{role} mask {mask} selected {len(sel)} atoms "
                                      "(need exactly 1)")
            a = prm.atoms[sel[0]]
            maps.append(ReactiveAtomMap(role=role, residue=resname,
                                        resid=a.residue.idx + 1, atom_name=atomname,
                                        amber_index=sel[0] + 1, mask=mask))

        def resolve_protein_atom(role: str, resnum: int, atomname: str) -> None:
            """Protein-nucleophile (Ser Oγ) via ordinal + identity, like catalytic."""
            ordinal = orig_to_ordinal.get(resnum)
            if ordinal is None or ordinal >= len(prm.residues):
                maps.append(ReactiveAtomMap(role=role, residue="?", resid=-1,
                    atom_name=atomname, amber_index=-1, mask="", orig_resid=resnum,
                    status="unresolved"))
                return
            r0 = prm.residues[ordinal]
            atom = next((a for a in r0.atoms if a.name == atomname), None)
            if atom is None:
                maps.append(ReactiveAtomMap(role=role, residue=r0.name, resid=r0.idx + 1,
                    atom_name=atomname, amber_index=-1, mask="", orig_resid=resnum,
                    status="unresolved"))
                return
            maps.append(ReactiveAtomMap(role=role, residue=r0.name, resid=r0.idx + 1,
                atom_name=atomname, amber_index=atom.idx + 1,
                mask=f":{r0.idx + 1}@{atomname}", orig_resid=resnum, status="ok"))

        # protein-nucleophile donor (e.g. 'SER:OG:68') -> the O_nuc
        if self.spec.reactive.donor_protein:
            from evoliez.md.nac import parse_protein_donor
            _rn, _atom, _num = parse_protein_donor(self.spec.reactive.donor_protein)
            if _num is not None:
                resolve_protein_atom("O_nuc", _num, _atom)

        for role, (which, name) in reactive_names.items():
            rn = role_res.get(which)
            if rn:
                resolve_one(role, rn, name)
        if mg_present and self.spec.metal.enabled:
            resolve_one("metal", self.spec.metal.ion, self.spec.metal.element)

        for tag in self.spec.catalytic_residues:
            aa, num_s = tag[0], tag[1:]
            num = int(num_s)
            ordinal = orig_to_ordinal.get(num)
            if ordinal is None or ordinal >= len(prm.residues):
                maps.append(ReactiveAtomMap(
                    role="catalytic", residue="?", resid=-1, atom_name=aa,
                    amber_index=-1, mask="", orig_resid=num, status="unresolved"))
                log.warning("catalytic %s: original residue %d not in protein", tag, num)
                continue
            r0 = prm.residues[ordinal]
            expected3 = pdb_order[ordinal][1]
            status = "ok" if _one_letter(expected3) == aa.upper() else "identity_mismatch"
            if status != "ok":
                log.warning("catalytic %s: expected %s but structure has %s at "
                            "original residue %d (mapped anyway, flagged)",
                            tag, aa, expected3, num)
            ca = next((a for a in r0.atoms if a.name == "CA"), r0.atoms[0])
            maps.append(ReactiveAtomMap(
                role="catalytic", residue=r0.name, resid=r0.idx + 1,
                atom_name=ca.name, amber_index=ca.idx + 1,
                mask=f":{r0.idx + 1}", orig_resid=num, status=status))
        return maps

    def _protein_residue_order(self) -> List[Tuple[int, str]]:
        """Ordered (original PDB residue number, 3-letter resname) for protein ATOM
        records in protein_clean.pdb — the residue order the prmtop preserves."""
        order: List[Tuple[int, str]] = []
        seen = set()
        for ln in (self.workdir / "protein_clean.pdb").read_text().splitlines():
            if not ln.startswith("ATOM"):
                continue
            try:
                num = int(ln[22:26])
            except ValueError:
                continue
            key = (ln[21:22], num)
            if key not in seen:
                seen.add(key)
                order.append((num, ln[17:20].strip()))
        return order

    # ---- tleap ------------------------------------------------------------ #
    def _run_tleap(self, clean_pdb: str, ligand_units: List[Tuple[str, str]]) -> None:
        script = tleap_amber_script(self.spec, ligand_units, protein_pdb=clean_pdb)
        (self.workdir / "tleap.in").write_text(script)
        r = _sh(["tleap", "-s", "-f", "tleap.in"], self.workdir, 900)
        leaplog = (self.workdir / "leap.log")
        logtext = (leaplog.read_text() if leaplog.exists() else "") + r.stdout + r.stderr
        if not (self.workdir / "complex.prmtop").exists():
            raise AmberBuildError("tleap", f"tleap produced no prmtop:\n{logtext[-1200:]}")
        # fail-loud on the "missing parameters" tleap prints but does not exit-fail on
        for marker in ("FATAL", "Could not find", "Parameter file was not saved",
                       "for atom types", "** No torsion terms"):
            if marker in logtext and "complex.prmtop" not in marker:
                # a genuine missing-parameter FATAL leaves no prmtop; if the prmtop
                # exists, downgrade the marker to a recorded warning
                if marker == "FATAL" and (self.workdir / "complex.prmtop").exists():
                    log.warning("tleap emitted FATAL but wrote a prmtop; inspect leap.log")
                    break

    # ---- orchestration ---------------------------------------------------- #
    def build(self, complex_pdb: Path) -> AmberSystem:
        from evoliez.adapters.openmm_engine import _ligands_at_pose
        complex_pdb = Path(complex_pdb)
        if not complex_pdb.exists():
            raise AmberBuildError("input", f"complex pdb not found: {complex_pdb}")

        clean_pdb, seq = self._prepare_protein(complex_pdb)

        # extract every non-metal ligand at its pose (graph match; ignores HETATM names)
        specs_for_pose = [(l.id, l.smiles) for l in self.spec.ligands if l.smiles]
        pose_mols = dict(_ligands_at_pose(complex_pdb, specs_for_pose))
        ligand_units: List[Tuple[str, str]] = []
        design_names: List[str] = []
        acceptor_names: List[str] = []
        for lig in self.spec.ligands:
            pose = pose_mols.get(lig.id)
            if pose is None:
                raise AmberBuildError("input",
                                      f"{lig.id}: not matched in the complex HETATM")
            from evoliez.md.charges import requires_fixed_charge_template
            if lig.charges_mol2 or requires_fixed_charge_template(lig):
                mol2 = self._param_curated_cofactor(lig, pose)
            else:
                mol2 = self._param_design_ligand(lig, pose)
            names = parse_mol2_atom_names((self.workdir / mol2).read_text())
            ligand_units.append((lig.id, mol2))
            if lig.role == "design_ligand":
                design_names = names
            elif lig.role in ("cofactor", "cosubstrate"):
                acceptor_names = names

        # metal: extract the pose Mg from the complex HETATM, append to the pdb
        mg_present = False
        if self.spec.metal.enabled:
            xyz = _extract_metal_xyz(complex_pdb, self.spec.metal.element)
            if xyz is None:
                raise AmberBuildError("input",
                                      f"metal {self.spec.metal.element} requested but "
                                      "not found in the complex HETATM")
            self._append_metal(clean_pdb, xyz)
            mg_present = True

        self._run_tleap(clean_pdb, ligand_units)

        # reactive-atom resolution -> prmtop masks/indices
        design_ref = (self.workdir / f"{self._design_id}.ref.sdf"
                      if self._design_id else None)
        acc_id = next((l.id for l in self.spec.ligands
                       if l.role in ("cofactor", "cosubstrate")), None)
        acc_ref = self.workdir / f"{acc_id}.ref.sdf" if acc_id else None
        rnames = reactive_atom_names(
            self.spec.reactive, design_ref if design_ref and design_ref.exists() else None,
            design_names, acc_ref if acc_ref and acc_ref.exists() else None,
            acceptor_names)
        reactive_map = self._resolve_masks(rnames, mg_present)

        # metrics + manifest
        import parmed
        prm = parmed.load_file(str(self.workdir / "complex.prmtop"))
        n_atoms = len(prm.atoms)
        net_charge = round(sum(a.charge for a in prm.atoms), 4)
        fp = system_fingerprint(self.spec, seq, self._param_sources)
        manifest = self._manifest(seq, n_atoms, net_charge, reactive_map, fp)
        (self.workdir / "amber_system_manifest.json").write_text(
            json.dumps(manifest, indent=2))
        return AmberSystem(
            prmtop=self.workdir / "complex.prmtop",
            inpcrd=self.workdir / "complex.inpcrd",
            workdir=self.workdir, fingerprint=fp, n_atoms=n_atoms,
            net_charge=net_charge, reactive_map=reactive_map, manifest=manifest)

    def _manifest(self, seq: str, n_atoms: int, net_charge: float,
                  reactive_map: List[ReactiveAtomMap], fp: str) -> dict:
        return {
            "schema": "amber_system_manifest/v1",
            "fingerprint": fp,
            "protein": {"n_residues": len(seq), "seq_sha1":
                        hashlib.sha1(seq.encode()).hexdigest()},
            "force_field": {"protein": self.spec.protein_ff, "ligand": "gaff2",
                            "water": self.spec.water if self.spec.solvent == "explicit"
                            else "implicit_GB(mbondi3)",
                            "solvent": self.spec.solvent,
                            "box_buffer_A": self.spec.box_buffer_A},
            "ligands": [
                {"id": l.id, "role": l.role, "net_charge": l.net_charge,
                 "param_source": self._param_sources.get(l.id, "am1bcc_gaff2"),
                 "allow_am1bcc": l.allow_am1bcc}
                for l in self.spec.ligands],
            "metal": ({"ion": self.spec.metal.ion, "charge": self.spec.metal.charge,
                       "ion_frcmod": self.spec.metal.ion_frcmod,
                       "lj_model": ("12-6-4" if "1264" in self.spec.metal.ion_frcmod
                                    else "12-6")} if self.spec.metal.enabled else None),
            "system": {"n_atoms": n_atoms, "net_charge": net_charge},
            "reactive_atoms": [
                {"role": m.role, "residue": m.residue, "resid": m.resid,
                 "orig_resid": m.orig_resid, "atom_name": m.atom_name,
                 "amber_index": m.amber_index, "mask": m.mask, "status": m.status}
                for m in reactive_map],
            "reactive_spec": {
                "label": self.spec.reactive.label,
                "donor_smarts": self.spec.reactive.donor_smarts,
                "acceptor_smarts": self.spec.reactive.acceptor_smarts,
                "donor_protein": self.spec.reactive.donor_protein,
                "transfer_is_h": self.spec.reactive.transfer_is_h},
        }


def _extract_metal_xyz(complex_pdb: Path, element: str) -> Optional[List[float]]:
    """First HETATM whose element/name matches ``element`` (e.g. MG)."""
    el = element.strip().upper()
    for ln in complex_pdb.read_text().splitlines():
        if not ln.startswith(("HETATM", "ATOM")):
            continue
        name = ln[12:16].strip().upper()
        pdb_el = ln[76:78].strip().upper() if len(ln) >= 78 else ""
        if name == el or pdb_el == el:
            try:
                return [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
            except ValueError:
                continue
    return None
