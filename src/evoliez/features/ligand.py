"""Ligand preprocessing (spec section 6.2 / 11.3).

With RDKit: real parsing, 3D embedding, Gasteiger charges, per-atom chemistry.
Without RDKit (laptop / CI): a deterministic synthetic ligand derived from the
SMILES string so the rest of the pipeline has a well-formed object to work on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from evoliez.config import LigandInput
from evoliez.logging_utils import get_logger
from evoliez.types import Ligand, LigandAtom
from evoliez.utils.seeds import derive_seed

log = get_logger("evoliez.ligand")

try:  # pragma: no cover - exercised on the science env
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False


_DONORS = {"N", "O"}
_ACCEPTORS = {"N", "O", "F"}
_HYDROPHOBIC = {"C", "S", "Cl", "Br", "I"}


def parse_ligand(spec: LigandInput) -> Ligand:
    if _HAVE_RDKIT:
        try:
            return _parse_rdkit(spec)
        except Exception as exc:  # fall back rather than crash the pipeline
            log.warning("RDKit ligand parse failed (%s); using synthetic model", exc)
    return _parse_synthetic(spec)


@dataclass(frozen=True)
class LigandSpec:
    """A ligand resolved into its docking ROLE (generic — no identity hardcoding).

    ``ligand`` is the parsed :class:`Ligand`; ``role`` is free-form
    (design_ligand | cofactor | substrate | product | ion | ...). ``dock`` selects
    the docking targets, ``keep_as_context`` the chains kept as fixed receptor
    context. ``is_primary`` flags the entry from ``InputConfig.ligand``.
    """

    id: str
    role: str
    dock: bool
    keep_as_context: bool
    ligand: Ligand
    type: str
    value: str
    is_primary: bool


def resolve_ligand_manifest(primary: LigandInput,
                            extras: List[LigandInput]) -> List["LigandSpec"]:
    """Resolve the primary ligand + extras into role-tagged :class:`LigandSpec`s.

    Generic + role-driven, never keyed on a ligand name/SMILES:
      * ``role`` None -> positional default (primary = design_ligand, extra = cofactor)
      * ``dock`` None -> True iff resolved role == "design_ligand"
      * ``keep_as_context`` None -> True iff resolved role != "design_ligand"
    Explicit ``dock`` / ``keep_as_context`` always win, so any ligand can be made a
    docking target and/or a context chain regardless of its role label.
    """
    specs: List[LigandSpec] = []
    for li, is_primary in [(primary, True)] + [(e, False) for e in extras]:
        role = li.role or ("design_ligand" if is_primary else "cofactor")
        dock = li.dock if li.dock is not None else (role == "design_ligand")
        keep = (li.keep_as_context if li.keep_as_context is not None
                else (role != "design_ligand"))
        specs.append(LigandSpec(
            id=li.id, role=role, dock=bool(dock), keep_as_context=bool(keep),
            ligand=parse_ligand(li), type=li.type, value=li.value,
            is_primary=is_primary,
        ))
    return specs


# --------------------------------------------------------------------------- #
# RDKit path
# --------------------------------------------------------------------------- #
def _mol_from_spec(spec: LigandInput):
    t = spec.type.lower()
    if t == "smiles":
        return Chem.MolFromSmiles(spec.value)
    if t == "inchi":
        return Chem.MolFromInchi(spec.value)
    if t in ("sdf", "mol2", "pdb"):
        suppl = {
            "sdf": Chem.SDMolSupplier,
        }
        if t == "sdf":
            mols = list(Chem.SDMolSupplier(spec.value, removeHs=False))
            return mols[0] if mols else None
        if t == "mol2":
            return Chem.MolFromMol2File(spec.value, removeHs=False)
        return Chem.MolFromPDBFile(spec.value, removeHs=False)
    raise ValueError(f"unsupported ligand type: {spec.type}")


def _parse_rdkit(spec: LigandInput) -> Ligand:
    mol = _mol_from_spec(spec)
    if mol is None:
        raise ValueError(f"could not parse ligand: {spec.value!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    AllChem.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer()

    atoms: List[LigandAtom] = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        sym = atom.GetSymbol()
        try:
            pc = float(atom.GetDoubleProp("_GasteigerCharge"))
            if math.isnan(pc):
                pc = 0.0
        except Exception:
            pc = 0.0
        atoms.append(
            LigandAtom(
                id=f"{sym}{idx}",
                element=sym,
                coord=(float(pos.x), float(pos.y), float(pos.z)),
                formal_charge=atom.GetFormalCharge(),
                partial_charge=pc,
                hybridization=str(atom.GetHybridization()),
                aromatic=atom.GetIsAromatic(),
                in_ring=atom.IsInRing(),
                is_donor=sym in _DONORS and atom.GetTotalNumHs() > 0,
                is_acceptor=sym in _ACCEPTORS,
                is_hydrophobic=sym in _HYDROPHOBIC and not atom.GetIsAromatic(),
                pharmacophore=_pharma(sym, atom.GetIsAromatic()),
            )
        )
    return Ligand(
        id=spec.id,
        smiles=Chem.MolToSmiles(Chem.RemoveHs(mol)),
        atoms=atoms,
        formal_charge=Chem.GetFormalCharge(mol),
        n_rotatable_bonds=AllChem.CalcNumRotatableBonds(mol),
        source="rdkit",
    )


# --------------------------------------------------------------------------- #
# Synthetic fallback (no RDKit)
# --------------------------------------------------------------------------- #
def _parse_synthetic(spec: LigandInput) -> Ligand:
    s = spec.value
    seed = derive_seed(0xABCDEF, s)
    # Heavy-atom count proxy: element-starting characters in the SMILES.
    heavy = max(4, sum(1 for c in s if c.isalpha() and c.isupper()))
    heavy = min(heavy, 60)
    atoms: List[LigandAtom] = []
    rng = seed
    for i in range(heavy):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        r = rng / 0x7FFFFFFF
        element = "C"
        if r < 0.18:
            element = "O"
        elif r < 0.30:
            element = "N"
        elif r < 0.34:
            element = "P"
        elif r < 0.37:
            element = "S"
        ang = 2 * math.pi * i / heavy
        coord = (
            2.5 * math.cos(ang),
            2.5 * math.sin(ang),
            0.6 * (i % 3 - 1),
        )
        aromatic = (element == "C") and (i % 6 == 0)
        atoms.append(
            LigandAtom(
                id=f"{element}{i}",
                element=element,
                coord=coord,
                formal_charge=-1 if element in ("O", "P") and i % 7 == 0 else 0,
                partial_charge=round(r - 0.5, 3),
                hybridization="sp2" if aromatic else "sp3",
                aromatic=aromatic,
                in_ring=aromatic,
                is_donor=element in _DONORS and i % 2 == 0,
                is_acceptor=element in _ACCEPTORS,
                is_hydrophobic=element in _HYDROPHOBIC and not aromatic,
                pharmacophore=_pharma(element, aromatic),
            )
        )
    fc = sum(a.formal_charge for a in atoms)
    return Ligand(
        id=spec.id,
        smiles=s,
        atoms=atoms,
        formal_charge=fc,
        n_rotatable_bonds=max(0, heavy // 5),
        source="synthetic",
    )


# --------------------------------------------------------------------------- #
# Chemistry-graph atom mapping (expert review #6)
#
# A positional zip() between the canonical RDKit parse and a tool's re-emitted
# atoms is a SILENT correctness risk: OpenFF Molecule.from_smiles, Boltz, and
# Vina/obabel are each free to order atoms differently than RDKit's canonical
# parse (openff-toolkit explicitly documents atom order is toolkit/version
# dependent). When the orders differ but the counts match, a zip attaches
# coordinates to the WRONG canonical atom id and every id-keyed downstream
# feature (interaction descriptor, s06b family fingerprint, ligand_importance,
# IFP edges, RMSD-to-reference) is wrong with no error raised.
#
# Fix: match the two atom sets by their CHEMISTRY GRAPH (element-labelled
# connectivity) and adopt coords through that permutation. Connectivity is
# inferred from 3D coordinates via covalent radii; this is conformation
# invariant (bonded distances ~1.0-1.8 Aa, non-bonded >~2.4 Aa), so the
# canonical ETKDG conformer and a Boltz/Vina pose of the SAME molecule yield
# the SAME topology and therefore a sound graph isomorphism. Only when a
# graph match is found is the lock reported as verified.
# --------------------------------------------------------------------------- #

# Cordero (2008) single-bond covalent radii (Aa); default for the rest.
_COVALENT_R = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "NA": 1.66, "MG": 1.41, "P": 1.07, "S": 1.05, "CL": 1.02, "K": 2.03,
    "CA": 1.76, "MN": 1.39, "FE": 1.32, "ZN": 1.22, "SE": 1.20, "BR": 1.20,
    "I": 1.39,
}
_BOND_TOL = 0.45  # standard add-on so real (slightly stretched) bonds register
# Cap isomorphism enumeration so a pathologically symmetric ligand can't blow
# up; the smallest mapping among the first _MAX_ISO is picked deterministically.
_MAX_ISO = 2000


def _norm_elem(sym: str) -> str:
    return (sym or "C").strip().upper()


def _infer_bonds(atoms) -> set:
    """Element + 3D-coord -> bond set {(i, j) | i < j} via covalent radii."""
    n = len(atoms)
    rad = [_COVALENT_R.get(_norm_elem(a.element), 0.77) for a in atoms]
    bonds = set()
    for i in range(n):
        xi, yi, zi = atoms[i].coord
        ri = rad[i]
        for j in range(i + 1, n):
            xj, yj, zj = atoms[j].coord
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
            if d <= ri + rad[j] + _BOND_TOL:  # d == 0 (stacked) counts as bonded
                bonds.add((i, j))
    return bonds


def _build_graph(atoms):
    import networkx as nx

    g = nx.Graph()
    for i, a in enumerate(atoms):  # insertion order = list order (determinism)
        g.add_node(i, el=_norm_elem(a.element))
    for i, j in sorted(_infer_bonds(atoms)):
        g.add_edge(i, j)
    return g


def _graph_atom_map(canonical, parsed):
    """Permutation canonical_index -> parsed_index from an element-labelled
    graph isomorphism, or None if the chemistry graphs do not match.

    Determinism: graphs are built in list order and the lexicographically
    smallest mapping among the first ``_MAX_ISO`` isomorphisms is chosen, so
    automorphic (chemically equivalent) atoms resolve identically every run
    and across networkx versions.
    """
    n = len(canonical)
    if n != len(parsed) or n == 0:
        return None
    if n == 1:  # a single atom is trivially (and correctly) mapped
        return {0: 0} if _norm_elem(canonical[0].element) == _norm_elem(
            parsed[0].element) else None

    from networkx.algorithms.isomorphism import (
        GraphMatcher,
        categorical_node_match,
    )

    gc = _build_graph(canonical)
    gp = _build_graph(parsed)
    # With no connectivity on either side there is nothing to verify against;
    # refuse to claim a verified lock for >1 atoms (honest fall-back instead).
    if gc.number_of_edges() == 0 and gp.number_of_edges() == 0:
        return None

    gm = GraphMatcher(gc, gp, node_match=categorical_node_match("el", None))
    best = None
    for k, mapping in enumerate(gm.isomorphisms_iter()):
        cand = tuple(mapping[i] for i in range(n))
        if best is None or cand < best:
            best = cand
        if k + 1 >= _MAX_ISO:
            break
    if best is None:
        return None
    return {i: best[i] for i in range(n)}


def relabel_to_canonical(parsed, canonical):
    """Atom-index lock by CHEMISTRY GRAPH (expert review #6).

    A tool (real Boltz, OpenFF, Vina/obabel) may re-emit ligand atoms in a
    different order than the canonical RDKit parse. Every downstream module
    keys by atom id, so we keep the **canonical atom ids + chemistry** and
    adopt only the tool's coordinates - but the tool coord must be attached to
    the *chemically corresponding* canonical atom, not the positionally i-th
    one. We therefore map the two atom sets by an element-labelled graph
    isomorphism and adopt coords through that permutation.

    Return contract is unchanged: ``(atoms, locked_bool)``. ``locked`` now
    means **ids are graph-verified**, not merely "counts equal". When the
    graph match fails we fall back to the positional/heavy zip so downstream
    still has a ligand, but we log a clear "atom-order NOT verified" warning
    and return ``locked=False`` so id-keyed features can be treated as
    unreliable (callers tag the ligand source / drop RMSD accordingly).

    The mapping is computed independently per ligand, which keeps it correct
    for the queued multi-ligand / ternary-complex work (each ligand's atoms
    are matched against its own canonical parse).
    """
    from dataclasses import replace

    if not canonical:
        return list(parsed), False

    pe, ce = list(parsed), list(canonical)
    heavy = False
    if len(pe) != len(ce):
        # Structure/docking tools re-emit HEAVY ATOMS ONLY while the canonical
        # parse adds explicit H (Chem.AddHs) -> e.g. NADP 44 heavy vs 70 with
        # H. Match on heavy atoms: canonical heavy ids/chemistry + tool coords,
        # H dropped (downstream interaction features are heavy-atom based).
        ch = [a for a in canonical if (a.element or "").upper() != "H"]
        ph = [a for a in parsed if (a.element or "").upper() != "H"]
        if not (ch and len(ph) == len(ch)):
            return list(parsed), False  # genuine count mismatch: never silent
        pe, ce, heavy = ph, ch, True

    perm = _graph_atom_map(ce, pe)
    if perm is not None:
        return [replace(ce[i], coord=pe[perm[i]].coord)
                for i in range(len(ce))], True

    # Counts agree but the chemistry graphs do not -> the tool's atom order
    # could not be verified. Adopt coords positionally so the pipeline still
    # has a ligand, but flag it loudly and report locked=False.
    log.warning(
        "ligand atom-order NOT verified by chemistry graph (%d %satoms); "
        "coords adopted positionally - id-keyed features (interaction "
        "descriptor, family fingerprint, ligand_importance, IFP, RMSD) may "
        "be unreliable for this ligand",
        len(ce), "heavy " if heavy else "",
    )
    return [replace(c, coord=p.coord) for c, p in zip(ce, pe)], False


def relabel_to_canonical_template(pose_text, pose_fmt, canonical, template):
    """Atom-index lock by an RDKit SMILES TEMPLATE (conformer-robust).

    The coordinate-inferred chemistry-graph lock (:func:`relabel_to_canonical`)
    re-derives bonds from each pose's 3D coords via covalent radii. For a large,
    flexible cofactor (NADP: 48 heavy atoms, two ribose-phosphate arms) a docked
    conformer can stretch/compress a bond just enough to flip an inferred edge,
    so the two coord-built graphs are NOT isomorphic and the lock silently falls
    back to the reference (coords lost; RMSD-to-reference -> a false 0). The
    proven fix — already used to pick gnina's representative pose — assigns bond
    orders from the ligand SMILES (:func:`normalize_pose_mol`), making the
    chemistry graph CORRECT regardless of conformer, then maps the pose onto the
    template by substructure match (symmetry handled via all matches).

    ``pose_text`` is the docked pose as a molblock (``pose_fmt="sdf"``) or a PDB
    block (``"pdb"``); ``canonical`` is the canonical reference ligand (its HEAVY
    atoms must be in the SAME order as ``template``'s atoms — both come from the
    same SMILES parse, which holds for the pipeline's ``parse_ligand`` output);
    ``template`` is the RDKit template mol (``rmsd_template(smiles)``).

    Returns ``(atoms, rmsd|None)`` where ``atoms`` are the canonical reference
    HEAVY atoms (ids + chemistry kept) carrying the pose's DOCKED coordinates
    through the template correspondence, and ``rmsd`` is the symmetry-corrected
    heavy-atom RMSD to the reference (the match that minimises it). Returns
    ``(None, None)`` when RDKit / the template / the match is unavailable so the
    caller falls back to the coord-graph lock. NOTE the returned atoms are HEAVY
    only (the template has no explicit H), matching the heavy-atom contract the
    downstream interaction features already use."""
    if template is None or not canonical or not pose_text:
        return None, None
    from dataclasses import replace

    try:
        import numpy as np
    except Exception:
        return None, None
    ch = [a for a in canonical if _norm_elem(a.element) != "H"]
    if not ch or len(ch) != template.GetNumAtoms():
        # The template/reference heavy-atom counts must agree for the
        # template-index == canonical-heavy-index identity to hold; otherwise
        # this lock is not applicable -> let the caller fall back.
        return None, None
    prb, _qc = normalize_pose_mol(pose_text, pose_fmt, template)
    if prb is None or prb.GetNumAtoms() != len(ch):
        return None, None
    try:
        # match[k] = TEMPLATE atom index for pose atom k. uniquify=False keeps
        # every symmetry-equivalent mapping so we can pick the lowest-RMSD one.
        matches = template.GetSubstructMatches(prb, uniquify=False, maxMatches=5000)
    except Exception:
        return None, None
    if not matches:
        return None, None
    conf = prb.GetConformer()
    pose_xyz = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(prb.GetNumAtoms())],
        dtype=float,
    )
    ref_xyz = np.array([a.coord for a in ch], dtype=float)
    n = len(ch)
    best = None  # (rmsd, canonical_ordered_pose_xyz)
    for m in matches:
        if len(m) != n:
            continue
        canon = np.empty((n, 3), dtype=float)
        seen = np.zeros(n, dtype=bool)
        ok = True
        for k, ti in enumerate(m):
            if ti < 0 or ti >= n:
                ok = False
                break
            canon[ti] = pose_xyz[k]
            seen[ti] = True
        if not ok or not seen.all():
            continue
        r = float(np.sqrt(((canon - ref_xyz) ** 2).sum(axis=1).mean()))
        if best is None or r < best[0]:
            best = (r, canon)
    if best is None:
        return None, None
    rmsd, canon = best
    atoms = [replace(ch[i], coord=(float(canon[i][0]), float(canon[i][1]),
                                   float(canon[i][2]))) for i in range(n)]
    return atoms, round(rmsd, 3)


def _pharma(element: str, aromatic: bool) -> str:
    if aromatic:
        return "aromatic"
    if element in ("O",):
        return "acceptor"
    if element in ("N",):
        return "donor"
    if element in ("P", "S"):
        return "polar"
    return "hydrophobic"


# --------------------------------------------------------------------------- #
# Symmetry-corrected heavy-atom RMSD-to-reference (the CASF/PDBbind docking-power
# metric). EXTRACTED here from the s05 docking report so the SAME proven method
# selects gnina's representative pose AND scores the report. Faithful to the
# report: build an RDKit template from the ligand SMILES, normalize each pose to
# its LARGEST heavy-atom fragment with template bond orders (so a co-modelled
# fragment such as formate is dropped and symmetry is well-defined), then take the
# minimum RMSD over symmetry-equivalent atom mappings WITHOUT re-superposition
# (the poses are already in the receptor frame). Needs RDKit; returns None when
# RDKit/normalization is unavailable so callers fall back honestly.
# --------------------------------------------------------------------------- #
def rmsd_template(smiles):
    """RDKit template mol from the ligand SMILES (bond-order donor for pose
    normalization), or None when SMILES is missing / RDKit is unavailable."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def normalize_pose_mol(text, fmt, template):
    """Normalized ligand RDKit mol from a docked pose: the LARGEST fragment (so a
    co-modelled formate is dropped), heavy atoms only, bond orders taken from the
    SMILES ``template`` so symmetry is well-defined and shared across poses.
    ``fmt`` is ``"sdf"`` (a molblock) or ``"pdb"`` (a PDB block). Returns
    ``(mol|None, qc-dict)``."""
    qc = {"raw": None, "heavy": None, "frags": None, "templated": False}
    try:
        from rdkit import Chem
    except Exception:
        return None, qc
    full = (Chem.MolFromMolBlock(text, sanitize=False, removeHs=False) if fmt == "sdf"
            else Chem.MolFromPDBBlock(text, sanitize=False, removeHs=False))
    if full is None:
        return None, qc
    qc["raw"] = full.GetNumAtoms()
    try:
        m = Chem.RemoveHs(full, sanitize=False)
    except Exception:
        m = full
    try:
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    except Exception:
        frags = [m]
    qc["frags"] = len(frags)
    m = max(frags, key=lambda f: f.GetNumAtoms())
    qc["heavy"] = m.GetNumAtoms()
    if template is not None:
        try:
            from rdkit.Chem import AllChem
            m = AllChem.AssignBondOrdersFromTemplate(template, m)
            qc["templated"] = True
        except Exception:
            pass
    return m, qc


def no_align_rmsd(prb, ref):
    """Symmetry-corrected heavy-atom RMSD WITHOUT superposition — the poses are
    already in the receptor frame, so we must NOT re-align (that would hide a
    translation/rotation). Min over symmetry-equivalent atom mappings. ``prb`` /
    ``ref`` are normalized RDKit mols (see :func:`normalize_pose_mol`)."""
    if prb is None or ref is None:
        return None
    try:
        import numpy as np
        matches = ref.GetSubstructMatches(prb, uniquify=False, maxMatches=5000)
        if not matches:
            return None
        cp, cr = prb.GetConformer(), ref.GetConformer()
        pc = np.array([[cp.GetAtomPosition(i).x, cp.GetAtomPosition(i).y,
                        cp.GetAtomPosition(i).z] for i in range(prb.GetNumAtoms())])
        best = None
        for mt in matches:
            rc = np.array([[cr.GetAtomPosition(j).x, cr.GetAtomPosition(j).y,
                            cr.GetAtomPosition(j).z] for j in mt])
            r = float(np.sqrt(((pc - rc) ** 2).sum(1).mean()))
            best = r if best is None or r < best else best
        return round(best, 2) if best is not None else None
    except Exception:
        return None


def _atoms_to_pdb_block(atoms) -> str:
    """A minimal HETATM PDB block from a list of :class:`LigandAtom` — the SAME
    representation the s05 report normalizes the reference ligand from (so the
    selection RMSD and the report RMSD share one normalization path). Only the
    fields RDKit's ``MolFromPDBBlock`` reads (record, serial, name, resname,
    x/y/z, element) are filled; H atoms are kept (``normalize_pose_mol`` strips
    them). Element-only atom names are zero-padded so RDKit infers the element."""
    lines = []
    for i, a in enumerate(atoms):
        el = (a.element or "C").strip()
        x, y, z = a.coord
        # PDB atom name: right-justify a 1-char element into cols 13-16 the way a
        # standard HETATM line does, so MolFromPDBBlock reads the element back.
        name = f"{el:>2}" if len(el) <= 2 else el[:4]
        lines.append(
            f"HETATM{i + 1:>5} {name:<4} LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2}")
    return ("\n".join(lines) + "\n") if lines else ""


def symmetry_corrected_rmsd(pose_text, reference, smiles, *,
                            pose_fmt: str = "sdf", template=None):
    """Symmetry-corrected heavy-atom RMSD of a docked pose to a reference ligand.

    The PROVEN s05-report metric, exposed for reuse (gnina representative-pose
    selection + the report). ``pose_text`` is the pose as a molblock (``pose_fmt
    ="sdf"``) or a PDB block (``"pdb"``). ``reference`` is the reference ligand,
    either a PDB-block string or a list of :class:`LigandAtom` (the s05
    ``reference_atoms``). ``smiles`` is the ligand SMILES (the bond-order
    template); pass a pre-built ``template`` to avoid re-parsing it per pose.
    Returns the RMSD in A, or None when RDKit/normalization fails for either side
    (caller falls back). GENERIC — no ligand identity is assumed; symmetry and
    fragment selection come entirely from the SMILES template."""
    tmpl = template if template is not None else rmsd_template(smiles)
    prb, _ = normalize_pose_mol(pose_text, pose_fmt, tmpl)
    if prb is None:
        return None
    if isinstance(reference, str):
        ref_text, ref_fmt = reference, "pdb"
    else:
        ref_text, ref_fmt = _atoms_to_pdb_block(reference), "pdb"
    ref_mol, _ = normalize_pose_mol(ref_text, ref_fmt, tmpl)
    if ref_mol is None:
        return None
    return no_align_rmsd(prb, ref_mol)
