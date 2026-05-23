# Xylose reductase (XR) benchmark scaffold

Literature-derived benchmark scaffold for the aldo-keto reductase
superfamily xylose reductase. Two host organisms appear in the
literature with overlapping engineering goals (NADPH → NADH cofactor
specificity switch); the residue numbering differs slightly between
them.

**Default target**: *Candida tenuis* XR (UniProt **O74237**, 322 aa)
— the structurally best-characterised XR (Kavanagh 2002 *PNAS* crystal
structure + Kratzer/Nidetzky 2006 catalytic-tetrad probing). Switch to
*Pichia stipitis* XYL1 (UniProt **P31867**, 318 aa) if the assay you
care about uses the *P. stipitis* numbering — the K270R / K274R
positions are the family-equivalent "cofactor-switch" residue between
the two hosts.

Why XR made the multi-enzyme set:
- it is **the** model NADP→NAD cofactor-specificity-engineering case
  (a useful sibling to PseFDH's NAD→NADP switch — the two together
  test the cofactor-discrimination signal in both directions);
- the **K274R/N276D/S277R** triple (Petschacher & Nidetzky 2008) is a
  well-documented engineered NADH-preferring XR with measured
  kcat / Km ratios — clean beneficial labels;
- the AKR-superfamily catalytic tetrad **Tyr51 / Lys80 / Asp46 / His110**
  is universally conserved and gives unambiguous deleterious labels
  (Kavanagh 2002 *PNAS* PMID 12446840).

## Numbering

Default rows use **Candida tenuis XR numbering** (UniProt O74237,
1-based, no signal peptide — XR is cytosolic). The four cofactor-switch
beneficial mutations come from two backgrounds:

- `K274R`, `N276D`, `S277R` — Candida tenuis XR (O74237) per
  Petschacher 2008.
- `K270R`, `K270M`, `N272D` — Pichia stipitis XYL1 (P31867) per
  Liang 2007 / Bengtsson 2009 / Watanabe 2007.

The two sets are **family-equivalent residues** (the structural loop
that contacts the 2′-phosphate of NADPH) and are kept under their
literature labels so any reader can re-find the citation. If you run
the C. tenuis FASTA, the four P. stipitis-numbered rows will fail
WT-letter validation — that's expected; switch the FASTA to P31867
to validate those rows.

A cleaner long-term option is to split the file into
`benchmark_ctenuis.csv` + `benchmark_pstipitis.csv`. Kept as one
scaffold for now so the multi-enzyme harness sees XR as one target.

## Sources used (every row cited)

### Beneficial (cofactor-specificity switch + neighbours)

| Mutation | Host (UniProt) | Source |
|---|---|---|
| `K274R` | C. tenuis (O74237) | Petschacher & Nidetzky 2008 — PMID 18752648; triple-mutant gateway |
| `N276D` | C. tenuis (O74237) | Petschacher & Nidetzky 2008 — PMID 18752648 |
| `S277R` | C. tenuis (O74237) | Petschacher & Nidetzky 2008 — PMID 18752648 |
| `K270R` | P. stipitis (P31867) | Liang/Bao/Jin 2007 — PMID 17436318 |
| `K270M` | P. stipitis (P31867) | Bengtsson 2009 — PMC2698887 |
| `N272D` | P. stipitis (P31867) | Watanabe & Kodaki 2007 — PMID 17331942 |

### Neutral (adjacent residues, modest effect)

| Mutation | Host | Source |
|---|---|---|
| `P275L` | C. tenuis | Petschacher 2008 supplementary — PMID 18752648 |
| `S275A` | P. stipitis | Liang 2007 control — PMID 17436318 |

### Deleterious (catalytic + substrate-binding ablations)

All four AKR-tetrad residues are from Kavanagh 2002 *PNAS*
(PMID 12446840):

| Mutation | Source |
|---|---|
| `Y51F` | Conserved catalytic Tyr — hydride-transfer partner |
| `K80M` | Conserved Lys — lowers Tyr pKa |
| `D46N` | Tetrad Asp — H-bonds Lys |
| `H110N` | Tetrad His — orients Tyr-OH |
| `W23A` | Kratzer/Nidetzky 2006 FEBS J — substrate-pocket Trp; PMID 16367942 |
| `F128A` | Kratzer/Nidetzky 2006 — pocket hydrophobic packing; PMID 16367942 |

## Substrate / cofactors

- **Substrate**: D-xylose
  `SMILES: O=C[C@H](O)[C@@H](O)[C@H](O)CO`
- **Cofactor (WT-preferred)**: NADPH (positive control vs PseFDH NADP+)
- **Cofactor (engineered)**: NADH (the cofactor-switch beneficial set is
  scored under NADH, not NADPH)

When running EvoLiEZ on this benchmark, point the config at NADPH
**and** NADH and compare per-cofactor scores — that's the read-out the
literature uses.

## Validate sequence consistency before using

```
# C. tenuis default:
bash scripts/fetch_target_fasta.sh O74237 xylose_reductase \
    Y51 K80 D46 H110 K274

# Or P. stipitis:
bash scripts/fetch_target_fasta.sh P31867 xylose_reductase \
    Y49 K78 D44 H108 K270
```

(P. stipitis numbering for the catalytic tetrad: Y49 / K78 / D44 / H108.)

`evoliez bench` will report any WT-letter mismatches — those mean the
row is on the wrong organism's numbering, not that the row is wrong.

## See also

- [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) — full benchmark harness.
- [`examples/pseudomonas_fdh/`](../pseudomonas_fdh/) — sibling cofactor-switch case.
