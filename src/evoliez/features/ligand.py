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


def relabel_to_canonical(parsed, canonical):
    """Atom-index lock (expert review #5).

    A tool (e.g. real Boltz) may re-emit ligand atoms in a different order /
    id scheme than the canonical parse. Every downstream module keys by atom
    id (importance, IFP, edges), so when the atom count matches we keep the
    **canonical atom ids + chemistry** and only adopt the tool's coordinates.
    On a count mismatch we keep the parsed atoms but flag the ligand so the
    inconsistency is visible, never silent.
    """
    from dataclasses import replace

    if not canonical:
        return list(parsed), False
    # exact match: keep canonical ids/chemistry, adopt the tool's coords.
    if len(parsed) == len(canonical):
        return [replace(c, coord=p.coord)
                for c, p in zip(canonical, parsed)], True
    # Structure/docking tools (real Boltz, Vina, ...) re-emit the ligand as
    # HEAVY ATOMS ONLY, but the canonical parse adds explicit H (Chem.AddHs)
    # -> e.g. NADP 44 heavy vs 70 with H. Lock on heavy atoms: keep canonical
    # heavy-atom ids/chemistry + tool coords and drop the H the tool never
    # produced (downstream interaction features are heavy-atom based). Still
    # an honest no-lock if the HEAVY counts genuinely differ.
    ch = [a for a in canonical if (a.element or "").upper() != "H"]
    ph = [a for a in parsed if (a.element or "").upper() != "H"]
    if ch and len(ph) == len(ch):
        return [replace(c, coord=p.coord) for c, p in zip(ch, ph)], True
    return list(parsed), False


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
