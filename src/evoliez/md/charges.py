"""Fixed partial-charge templates for cofactors that on-the-fly AM1-BCC cannot
parameterise.

The physiological −3 NADP (tri-anionic, phosphate-rich) does NOT converge through
``antechamber -c bcc`` / ``sqm`` under the default gas-phase geometry optimisation
(the high negative charge density makes the optimiser crawl and would distort the
phosphates anyway). Both the OpenMM (GAFFTemplateGenerator) and Amber (antechamber)
routes funnel through that same sqm step, so neither could build the ligand and s10
silently skipped every NADP candidate.

Fix: derive the charges ONCE (AM1-BCC *single-point* at the bound pose, ``maxcyc=0``
— converges in ~1 min, keeps the binding geometry) and store them as a fixed-charge
template under ``params/<name>/``. At MD time the template charges are transferred
onto each pose's ligand by GRAPH ISOMORPHISM, which is robust to the per-pose HETATM
atom ordering. A partial or element-inconsistent map RAISES — charges are never
silently guessed (the anti-s06 rule: a mis-built ligand must be a loud failure).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

# Cofactors whose on-the-fly AM1-BCC charge derivation is unreliable (large,
# multiply-charged, phosphate-rich). Matched case-insensitively as a SUBSTRING of
# the ligand id, so "NADP_cofactor" / "NADPH" / "NAP" all hit. The charge+size
# heuristic in ``requires_fixed_charge_template`` is the real gate; this list is a
# convenience and a piece of documentation.
HIGH_RISK_COFACTORS = (
    "NADPH", "NADP", "NAP", "NDP", "NADH", "NAD", "NAI",
    "FAD", "FMN", "ATP", "ADP", "AMP", "GTP", "GDP", "COA", "SAM",
)


def requires_fixed_charge_template(ligand) -> bool:
    """Whether this ligand must use a fixed-charge template instead of on-the-fly
    AM1-BCC. Explicit config wins (``charges_mol2`` set, or ``allow_am1bcc`` False);
    otherwise auto-detect the high-risk cofactors so a mis-configured run fails
    loudly rather than hanging in a non-converging ``sqm``."""
    if getattr(ligand, "charges_mol2", None):
        return True
    if getattr(ligand, "allow_am1bcc", True) is False:
        return True
    name = (getattr(ligand, "id", "") or "").upper()
    if any(c in name for c in HIGH_RISK_COFACTORS):
        return True
    nc = int(getattr(ligand, "formal_charge", 0) or 0)
    if nc <= -2 and int(getattr(ligand, "n_heavy", 0) or 0) >= 30:
        return True       # tri-anionic + large: the regime where sqm won't converge
    return False


def _gaff_element(gaff_type: str) -> str:
    """Element from a GAFF/GAFF2 atom type. The first character is the element for
    every element NADP contains (c/n/o/p/h/s -> C/N/O/P/H/S); two-letter halogens
    (cl/br) are special-cased for generality."""
    t = gaff_type.lower()
    if t.startswith("cl"):
        return "Cl"
    if t.startswith("br"):
        return "Br"
    return gaff_type[0].upper()


def _parse_mol2_atoms(mol2_path) -> Tuple[List[float], List[str]]:
    """(charges, element) per atom in mol2 file order. RDKit cannot read a
    GAFF2-typed mol2 (it rejects types like ``nt``), so the charges are parsed
    directly from the @<TRIPOS>ATOM block."""
    lines = Path(mol2_path).read_text().splitlines()
    i = lines.index("@<TRIPOS>ATOM")
    j = lines.index("@<TRIPOS>BOND")
    charges: List[float] = []
    elements: List[str] = []
    for ln in lines[i + 1:j]:
        f = ln.split()
        charges.append(float(f[-1]))
        elements.append(_gaff_element(f[5]))     # col 6 = SYBYL/GAFF atom type
    return charges, elements


def load_reference(sdf_path, mol2_path):
    """``(ref_rdkit_mol, ref_charges)`` for a fixed-charge cofactor template.

    The graph comes from the sdf (RDKit reads it reliably; it cannot read the
    GAFF2-typed mol2) and the charges from the mol2. They share atom order because
    ``antechamber`` preserves the input order — ASSERTED here element-by-element so
    a reordering can never silently mis-key the charges."""
    from rdkit import Chem

    ref = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)[0]
    if ref is None:
        raise ValueError(f"unreadable charge-reference sdf: {sdf_path}")
    charges, elements = _parse_mol2_atoms(mol2_path)
    if ref.GetNumAtoms() != len(charges):
        raise ValueError(
            f"charge-reference atom-count mismatch: sdf {ref.GetNumAtoms()} vs "
            f"mol2 {len(charges)} ({sdf_path} / {mol2_path})")
    for k, a in enumerate(ref.GetAtoms()):
        if a.GetSymbol().upper() != elements[k].upper():
            raise ValueError(
                f"charge-reference order mismatch at atom {k}: sdf {a.GetSymbol()} "
                f"vs mol2 {elements[k]} — the graph and charges are not in the same "
                "atom order, refusing to mis-key charges")
    return ref, charges


def transfer_charges(pose_mol, ref_mol, ref_charges, net_charge: int) -> List[float]:
    """Partial charges for ``pose_mol``'s atoms, transferred from ``ref_mol`` by
    graph isomorphism. Bonds are made generic for the match so it is robust to
    bond-order / aromaticity perception differences between the pose build and the
    reference. RAISES on an incomplete map or any element mismatch; the result is
    normalised to sum to exactly ``net_charge``."""
    from rdkit import Chem

    qp = Chem.AdjustQueryParameters.NoAdjustments()
    qp.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(ref_mol, qp)
    match = pose_mol.GetSubstructMatch(query)        # match[i] = pose idx for ref atom i
    if len(match) != ref_mol.GetNumAtoms():
        raise ValueError(
            f"fixed-charge graph match incomplete: mapped {len(match)}/"
            f"{ref_mol.GetNumAtoms()} atoms — the pose ligand is not the reference "
            "molecule (atom count / connectivity differ); refusing to guess charges")
    out: List[Optional[float]] = [None] * pose_mol.GetNumAtoms()
    for ref_i, pose_i in enumerate(match):
        if (ref_mol.GetAtomWithIdx(ref_i).GetAtomicNum()
                != pose_mol.GetAtomWithIdx(pose_i).GetAtomicNum()):
            raise ValueError(
                f"fixed-charge element mismatch: ref atom {ref_i} -> pose atom {pose_i}")
        out[pose_i] = ref_charges[ref_i]
    if any(v is None for v in out):
        n = sum(1 for v in out if v is None)
        raise ValueError(f"fixed-charge transfer left {n} pose atom(s) unassigned")
    residual = (net_charge - sum(out)) / len(out)    # spread BCC rounding to exact net
    return [v + residual for v in out]
