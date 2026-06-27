"""WT/reference-anchored mutant builder — the core of functional-state-anchored
validation.

Instead of letting Boltz re-predict a whole complex per mutant (which re-searches
the ligand pose and so mixes the mutation effect with pose-search noise), build the
mutant by:

  1. taking the REFERENCE complex (reference backbone + reference cofactor/substrate
     poses, e.g. the WT FDH·NADP·formate),
  2. introducing ONLY the point mutation(s) in place (PDBFixer side-chain swap),
  3. keeping every ligand/cofactor/substrate atom at its REFERENCE pose,
  4. handing the result to run_md, whose tiered positional restraints provide the
     LIMITED relaxation (backbone held, mutation site + local shell free).

The downstream pose gate (md.pose_gate) then checks the cofactor stayed
reference-like; a fresh Boltz pose is treated separately as an alternative-pose
hypothesis, never as the validated structure.

Engine deps (PDBFixer/OpenMM) are lazily imported so the module is light to import.
"""
from __future__ import annotations

import copy
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_AA1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}


def _split_protein_ligand(
        pdb_text: str) -> Tuple[List[str], List[str], List[str]]:
    """(protein ATOM lines, ligand HETATM lines, CONECT lines). CONECT is kept:
    run_md's ligand reader uses proximityBonding=False, so the ligand bond graph
    comes ONLY from CONECT — dropping it leaves a bondless het that fails the
    SMILES-template match ('design ligand not found')."""
    prot, het, conect = [], [], []
    for ln in pdb_text.splitlines():
        if ln.startswith("ATOM"):
            prot.append(ln)
        elif ln.startswith("HETATM"):
            het.append(ln)
        elif ln.startswith("CONECT"):
            conect.append(ln)
    return prot, het, conect


def _atom_chain(protein_lines: Sequence[str]) -> str:
    counts: dict = {}
    for ln in protein_lines:
        c = ln[21:22]
        counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get) if counts else "A"


def _norm_mut(m) -> Optional[Tuple[str, int, str]]:
    """Accept a Mutation (wt/position/mut) or a (wt1, resseq, mut1) tuple."""
    if hasattr(m, "position"):
        return (str(getattr(m, "wt", "")).upper(), int(m.position),
                str(getattr(m, "mut", "")).upper())
    if isinstance(m, (tuple, list)) and len(m) == 3:
        return (str(m[0]).upper(), int(m[1]), str(m[2]).upper())
    return None


@dataclass
class AnchoredBuildResult:
    out_pdb: str
    applied: List[str]
    skipped: List[Tuple[str, str]]
    n_protein_atoms: int
    n_ligand_atoms: int

    @property
    def ok(self) -> bool:
        return bool(self.applied) and self.n_ligand_atoms > 0


def build_anchored_mutant_pdb(
    ref_pdb, mutations, out_pdb, *, chain: Optional[str] = None, ph: float = 7.0,
) -> AnchoredBuildResult:
    """Write a reference-anchored mutant PDB: reference backbone with the point
    mutation(s) applied in place (PDBFixer), plus the reference ligand atoms kept
    verbatim. Returns which mutations applied/skipped + atom counts.

    Mutations whose WT 3-letter name does not match the reference at that residue
    are skipped (honest — not silently forced); the rest still apply."""
    from openmm.app import PDBFile  # lazy
    from pdbfixer import PDBFixer

    text = Path(ref_pdb).read_text()
    prot, het, conect = _split_protein_ligand(text)
    if not prot:
        raise ValueError(f"{ref_pdb}: no protein ATOM records")
    ch = chain or _atom_chain(prot)

    prot_pdb = Path(out_pdb).with_suffix(".ref_protein.pdb")
    prot_pdb.write_text("\n".join(prot) + "\nEND\n")

    fixer = PDBFixer(filename=str(prot_pdb))
    applied: List[str] = []
    skipped: List[Tuple[str, str]] = []
    mutstrs: List[str] = []
    for m in mutations:
        nm = _norm_mut(m)
        if nm is None:
            continue
        wt1, pos, mut1 = nm
        w3, m3 = _AA1TO3.get(wt1), _AA1TO3.get(mut1)
        if not w3 or not m3:
            skipped.append((f"{wt1}{pos}{mut1}", "non-standard residue"))
            continue
        mutstrs.append(f"{w3}-{pos}-{m3}")

    # Apply together; if that raises (one WT mismatch aborts the batch), retry singly.
    def _apply(strs):
        fixer.applyMutations(list(strs), ch)

    if mutstrs:
        try:
            _apply(mutstrs)
            applied = list(mutstrs)
        except Exception:  # noqa: BLE001
            for ms in mutstrs:
                try:
                    fx1 = PDBFixer(filename=str(prot_pdb))
                    fx1.applyMutations([ms], ch)
                    applied.append(ms)
                except Exception as e2:  # noqa: BLE001
                    skipped.append((ms, str(e2)[:80]))
            if applied:                      # rebuild with only the ones that work
                fixer = PDBFixer(filename=str(prot_pdb))
                fixer.applyMutations(applied, ch)

    fixer.findMissingResidues()
    fixer.missingResidues = {}              # do NOT fill chain gaps (anchored = as-is)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()                 # rebuild the mutated side chain heavy atoms
    fixer.addMissingHydrogens(ph)

    buf = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, buf, keepIds=True)
    prot_out = [ln for ln in buf.getvalue().splitlines() if ln.startswith("ATOM")]

    # Keep ligand HEAVY atoms (the reference pose) re-serialised AFTER the protein
    # (so serials never collide with the rebuilt protein), and carry the heavy-atom
    # CONECT bonds remapped to the new serials. run_md rebuilds H from the SMILES;
    # the heavy het + CONECT is exactly the Boltz-style block _ligands_at_pose
    # matches (proximityBonding=False -> CONECT is the only bond source).
    def _is_h(ln: str) -> bool:
        return (ln[76:78].strip() or ln[12:16].strip()[:1]).upper() == "H"

    het_out: List[str] = []
    old2new: dict = {}
    serial = len(prot_out)
    for ln in het:
        if _is_h(ln):
            continue
        try:
            old = int(ln[6:11])
        except ValueError:
            continue
        serial += 1
        old2new[old] = serial
        # rename resName -> LIG: PDBFixer.removeHeterogens treats "UNK" as an
        # unknown PROTEIN residue and KEEPS it (then it collides with the
        # separately-added OpenFF ligand); "LIG" is a clear heterogen and is
        # dropped, matching the Boltz-complex form the engine expects.
        het_out.append(ln[:6] + f"{serial:>5}" + ln[11:17] + "LIG" + ln[20:])
    conect_out: List[str] = []
    for ln in conect:
        raw = ln.rstrip()
        serials = []
        for i in range(6, len(raw), 5):
            seg = raw[i:i + 5].strip()
            if seg.isdigit():
                serials.append(int(seg))
        if not serials or serials[0] not in old2new:
            continue                       # base atom is an H / not a kept het atom
        mapped = [old2new[s] for s in serials if s in old2new]
        if len(mapped) >= 2:               # base + >=1 heavy neighbour
            conect_out.append("CONECT" + "".join(f"{m:>5}" for m in mapped))
    out_lines = prot_out + ["TER"] + het_out + conect_out + ["END"]
    Path(out_pdb).write_text("\n".join(out_lines) + "\n")
    try:
        prot_pdb.unlink()
    except OSError:
        pass
    return AnchoredBuildResult(str(out_pdb), applied, skipped,
                               len(prot_out), len(het_out))


def build_anchored_mutant_complex(ref_complex, mutations, out_pdb, **kw):
    """Reference-anchored mutant as a Complex for run_md: deepcopy the reference,
    point structure.pdb_path at the anchored PDB, set the mutant sequence. The
    ligand object (smiles/charges/role) is inherited from the reference."""
    res = build_anchored_mutant_pdb(
        getattr(ref_complex.structure, "pdb_path", None) or kw.pop("ref_pdb"),
        mutations, out_pdb, **kw)
    mc = copy.deepcopy(ref_complex)
    mc.structure.pdb_path = str(out_pdb)
    seq = list(mc.structure.sequence or "")
    for m in mutations:
        nm = _norm_mut(m)
        if nm and 0 < nm[1] <= len(seq):
            seq[nm[1] - 1] = nm[2]
    mc.structure.sequence = "".join(seq)
    mc.method = "wt_anchored"
    return mc, res
