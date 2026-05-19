"""Ligand preprocessing (spec section 6.2 / 11.3).

With RDKit: real parsing, 3D embedding, Gasteiger charges, per-atom chemistry.
Without RDKit (laptop / CI): a deterministic synthetic ligand derived from the
SMILES string so the rest of the pipeline has a well-formed object to work on.
"""

from __future__ import annotations

import math
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
