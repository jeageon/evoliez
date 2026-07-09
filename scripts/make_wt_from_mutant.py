#!/usr/bin/env python
"""Build a clean WT anchored complex by reverting a mutant's mutations.

The E4a run stores clean anchored complexes only for mutants (protein + 3-HP +
ATP + Mg at pose, good CONECT). There is no clean pre-solvation WT complex. This
auto-detects a mutant's substitutions by diffing its protein sequence against the
WT FASTA, then uses ``md.anchored_build.build_anchored_mutant_pdb`` to revert them
— producing a WT complex in the SAME clean format (ligand atoms kept verbatim).

Usage:
    python scripts/make_wt_from_mutant.py <mutant_anchored.pdb> <wt.fasta> <out_wt.pdb>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoliez.adapters.amber_builder import _one_letter  # noqa: E402
from evoliez.md.anchored_build import build_anchored_mutant_pdb  # noqa: E402


def protein_residues(pdb: Path):
    order, seen = [], set()
    for ln in pdb.read_text().splitlines():
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


def read_fasta(path: Path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines()
                   if l.strip() and not l.startswith(">"))


def main() -> int:
    mutant, fasta, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    wt_seq = read_fasta(fasta)
    reverts = []
    for num, res3 in protein_residues(mutant):
        have = _one_letter(res3)
        if 1 <= num <= len(wt_seq):
            want = wt_seq[num - 1]
            if have != "X" and want != have:
                reverts.append((have, num, want))      # (wt-of-mutant, pos, WT-AA) tuple
    print(f"detected {len(reverts)} substitution(s) to revert: "
          + ", ".join(f"{w}{p}{m}" for w, p, m in reverts))
    res = build_anchored_mutant_pdb(mutant, reverts, out)
    print(f"applied={getattr(res, 'applied', '?')} skipped={getattr(res, 'skipped', '?')}")
    print(f"WT complex written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
