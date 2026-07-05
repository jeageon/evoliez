#!/usr/bin/env python3
"""Regenerate the s03 MSA report in QUERY coordinates.

The original report was built over the raw mafft alignment (18012 columns for
the 1188-aa query, because mafft inserted gaps into the query row to host
divergent ANL-superfamily indels). Coverage/occupancy were on the 18012-column
axis while the query-indexed conservation.json landed on 1..1188 -> the two
tracks were on different axes and the overlay was unreadable ("strange graph").

msa_report.compute_msa_stats now projects every row onto the query's non-gap
columns, so all tracks share the 1188-residue target axis. This rebuilds the
CAR report with that fix (s03 is skipped on --resume, so it won't rebuild itself)
and overlays the catalytic + substrate-pocket positions.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from evoliez.io.msa_report import (compute_msa_stats, effective_neff,  # noqa: E402
                                   write_msa_report)

RUN = Path("runs/srcar_3hp_full")


def read_fasta(p):
    ids, seqs, cur = [], [], []
    with open(p) as fh:
        for ln in fh:
            if ln.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
                ids.append(ln[1:].strip().split()[0])
            else:
                cur.append(ln.strip())
    if cur:
        seqs.append("".join(cur))
    return ids, seqs


ids, seqs = read_fasta(RUN / "msa" / "alignment.fasta")
cons = json.load(open(RUN / "msa" / "conservation.json"))
target_len = sum(1 for c in seqs[0] if c != "-")

# catalytic machinery (locked) + substrate-pocket design residues (crystal-defined)
marked = {
    "catalytic (locked)": [268, 269, 273, 315, 507, 519, 522, 629, 970, 974],
    "substrate pocket (designable)": [263, 264, 265, 271, 274, 275, 290, 294,
                                      408, 430, 433, 438],
}

mstats = compute_msa_stats(ids, seqs, {str(k): v for k, v in cons.items()},
                           marked=marked, occ_mask=0.5)
mstats["neff80"] = effective_neff(seqs)

write_msa_report(
    RUN / "reports" / "msa_report.html",
    target_id="srcar_full",
    target_len=target_len,
    stats=mstats,
    generated="2026-07-04 (query-coordinate fix)",
    provenance="s03 integrated MSA — projected to query coordinates",
    conditions=[
        ("retrieval", "mmseqs2 vs UniRef30 (2302), identity-banded, de-duplicated"),
        ("alignment", "mafft, target-anchored"),
        ("report axis", f"query coordinates ({target_len} target positions); "
                        f"raw alignment had {len(seqs[0]):,} columns with "
                        f"query-row gaps (now projected out)"),
        ("conservation metric", "per-column Shannon entropy (gaps excluded)"),
        ("occupancy mask", "conservation hidden where <50% of rows align"),
    ],
)
print(f"n_seqs={mstats['n_seqs']:,} aln_len(report)={mstats['aln_len']:,} "
      f"(was {len(seqs[0]):,}) target_len={target_len}")
print(f"well_aligned(>50%)={mstats['well_aligned']} "
      f"low_occ_50={mstats['n_low_occ_50']} low_occ_10={mstats['n_low_occ_10']} "
      f"neff80={mstats['neff80']} mean_cons={mstats['mean_cons']}")
