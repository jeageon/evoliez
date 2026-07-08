#!/usr/bin/env python
"""V6-4 helper: derive a car_spec-compatible A-domain MUTANT input from the WT
A-domain template, reproducibly.

Context: the production pipeline emits *full-protein* anchored candidate
structures (ATP + 3-HP + NADP, no explicit adenylation-site Mg, ~1188 res) in a
different frame from the ``car_spec`` A-domain system (ATP + 3-HP + Mg, ~720 res).
Those cannot feed the ``car_spec`` builder (it extracts Mg from the input and
graph-matches ATP). Rather than hand-prepare each mutant (non-reproducible), we
mutate the WT A-domain template *in place*: the ligand pose + Mg frame is
preserved exactly, and only the mutated residues are reduced to backbone so
``tleap`` rebuilds an idealized side chain (the builder's minimization then
relaxes it before MD).

This is TEMPLATE mutagenesis, not functional-state anchoring — the resulting
evidence is screening-level reaction-geometry access, same claim ceiling as the
WT run. It exists to make the CAR portfolio MD reproducible from
(WT template + mutation string), closing a V6-7 reproducibility gap.

Usage:
    python scripts/prep_car_adomain_mutant.py \
        --wt /path/wt_reverted.pdb --mutations "G430R;S433F;G407K" \
        --out /path/mut_lead.pdb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 1-letter -> Amber 3-letter (the 20 standard; the mutant residue is rebuilt by tleap)
_AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
# backbone atoms we keep for a mutated residue; tleap adds the rest of the side chain
_BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA", "H1", "H2", "H3"}


def parse_mutations(spec: str) -> dict:
    """'G430R;S433F' -> {430: ('G','R'), 433: ('S','F')}. Rejects malformed tokens."""
    out = {}
    for tok in spec.replace(",", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        wt, mut = tok[0].upper(), tok[-1].upper()
        try:
            num = int(tok[1:-1])
        except ValueError:
            raise SystemExit(f"bad mutation token '{tok}' (want e.g. G430R)")
        if wt not in _AA3 or mut not in _AA3:
            raise SystemExit(f"bad residue letter in '{tok}'")
        out[num] = (wt, mut)
    if not out:
        raise SystemExit("no mutations parsed")
    return out


def prep(wt_pdb: Path, muts: dict, out_pdb: Path) -> dict:
    lines = wt_pdb.read_text().splitlines()
    # sanity: WT residue identity must match the mutation's stated WT letter
    seen_at = {}
    for ln in lines:
        if ln.startswith("ATOM"):
            try:
                num = int(ln[22:26])
            except ValueError:
                continue
            if num in muts:
                seen_at.setdefault(num, ln[17:20].strip())
    for num, (wt, mut) in muts.items():
        got = seen_at.get(num)
        if got is None:
            raise SystemExit(f"residue {num} not found in {wt_pdb.name}")
        if got != _AA3[wt]:
            raise SystemExit(
                f"WT mismatch at {num}: template has {got}, mutation says "
                f"{_AA3[wt]} ({wt}{num}{mut}) — refusing (wrong template/numbering)")

    kept_sidechain_dropped = 0
    out = []
    for ln in lines:
        if ln.startswith("ATOM"):
            try:
                num = int(ln[22:26])
            except ValueError:
                out.append(ln)
                continue
            if num in muts:
                atom = ln[12:16].strip()
                if atom not in _BACKBONE:
                    kept_sidechain_dropped += 1
                    continue  # drop side-chain atom; tleap rebuilds it
                # relabel residue to the target type (cols 18-20, 1-indexed 18..20)
                mut3 = _AA3[muts[num][1]]
                ln = ln[:17] + f"{mut3:>3}" + ln[20:]
            out.append(ln)
        else:
            out.append(ln)  # HETATM (ATP/3-HP/Mg), TER, CRYST1, END ... preserved
    out_pdb.write_text("\n".join(out) + "\n")
    return {"mutations": {f"{v[0]}{k}{v[1]}": _AA3[v[1]] for k, v in muts.items()},
            "sidechain_atoms_dropped": kept_sidechain_dropped,
            "out": str(out_pdb)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt", required=True, help="WT A-domain template PDB (ATP+3HP+Mg)")
    ap.add_argument("--mutations", required=True, help="e.g. 'G430R;S433F;G407K'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    muts = parse_mutations(args.mutations)
    info = prep(Path(args.wt), muts, Path(args.out))
    print(f"OK  {args.mutations}  ->  {info['out']}")
    print(f"    residues relabelled: {list(info['mutations'])}")
    print(f"    side-chain atoms dropped (tleap will rebuild): "
          f"{info['sidechain_atoms_dropped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
