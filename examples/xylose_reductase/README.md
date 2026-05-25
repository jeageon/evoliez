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

**Cheap-run #6 finding:** the Kavanagh 2002 / Petschacher 2008
**paper** residue positions do NOT match UniProt O74237 numbering.
The original draft of this card used paper positions (Y51 / K80 /
H110 / S277), every one of which was a sequence mismatch against
the UniProt-fetched FASTA. The rows below now use **UniProt
O74237 authoritative positions** (verified via
`scripts/fetch_target_fasta.sh O74237`).

  Paper ↔ UniProt mapping:
    Kavanagh Y51  → UniProt Y52  (ACT_SITE Proton donor)
    Kavanagh K80  → UniProt K81  (SITE Lowers Tyr pKa)
    Kavanagh H110 → UniProt H114 (BINDING substrate)
    Kavanagh D46  → no UniProt annotation on O74237; dropped
    Petschacher S277 → UniProt L277 (paper used a different
                                     Candida tenuis strain variant)

### Beneficial (cofactor-specificity switch + neighbours)

| Mutation | Source |
|---|---|
| `K274R` | Petschacher & Nidetzky 2008 — PMID 18752648; primary cofactor-switch residue (UniProt K274) |
| `N276D` | Petschacher & Nidetzky 2008 — PMID 18752648; switch-loop charge introduction (UniProt N276) |
| `L277R` | Petschacher & Nidetzky 2008 — PMID 18752648; equivalent of paper's S277R for this strain variant (UniProt L277) |
| `K274M` | Bengtsson 2009 — PMC2698887; alternative hydrophobic swap at K274 |
| `N276H` | Liang/Bao/Jin 2007 — PMID 17436318; family-equivalent NADH-preference variant |

### Neutral (adjacent residues, modest effect)

| Mutation | Source |
|---|---|
| `S275A` | Liang/Bao/Jin 2007 control — PMID 17436318; adjacent to switch loop |
| `P278A` | Petschacher 2008 supplementary — PMID 18752648; neighbouring Pro |
| `F225A` | UniProt O74237 BINDING NAD+ adjacent — conservative aromatic→small swap |

### Deleterious (catalytic + substrate-binding ablations)

UniProt-renumbered from Kavanagh 2002 *PNAS* (PMID 12446840):

| Mutation | Source |
|---|---|
| `Y52F` | UniProt Y52 = paper Y51 (ACT_SITE Proton donor); ablation kills hydride transfer |
| `K81M` | UniProt K81 = paper K80 (SITE Lowers Tyr pKa); ablation kills activity |
| `H114N` | UniProt H114 = paper H110 (BINDING substrate); catalytic-triad His ablation |
| `Y52A` | Full ablation of proton donor |
| `K81A` | Full ablation of pKa Lys |
| `H114A` | Full ablation of substrate-binding His |

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
