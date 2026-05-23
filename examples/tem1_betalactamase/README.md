# TEM-1 β-lactamase benchmark scaffold

Literature-derived benchmark scaffold for *Escherichia coli* TEM-1
β-lactamase (UniProt **P62593**, 286 aa, signal peptide 1–23 + mature
24–286). This is the canonical class-A serine β-lactamase, picked as a
**no-cofactor** benchmark to balance the cofactor-heavy enzymes in the
EvoLiEZ benchmark suite (FDH, XR, P450 BM3).

Why TEM-1 made the multi-enzyme set:
- the Stiffler / Hekstra / Ranganathan 2015 *Cell* DMS gives single-codon
  fitness for nearly every position under amp ± inhibitor, so labels
  scale far beyond what we can pack into one CSV;
- the Hall & Barlow / Hall classic-engineering papers give clean
  "evolved-clinical" ESBL mutations (G238S, E104K, R164S, M69L) with
  well-documented activity changes;
- catalytic residues are perfectly defined (Ser70 nucleophile, Lys73
  general base, Glu166 Omega-loop, Ser130, Lys234, Asn170), giving the
  catalytic-avoidance metric strong signal.

## Numbering

All positions use **standard Ambler numbering** — the same convention
the literature uses for class-A β-lactamases. UniProt P62593 ships the
sequence with the 23-residue Sec signal peptide; Ambler position N
(e.g. S70) corresponds to UniProt position N+23 (S93 in UniProt
1-based). The convention used in this scaffold is **Ambler** so the
rows are directly comparable to the cited papers.

If you fetch the FASTA with
`bash scripts/fetch_target_fasta.sh P62593 tem1_betalactamase S70 K73 E166`
the script will print the residue at each named position **in UniProt
numbering** — for Ambler S70 you should see `S` at UniProt index 93
(70 + 23). Either:

1. Strip the signal peptide from `target.fasta` before validation
   (recommended; matches the mature crystal-structure numbering used
   in every β-lactamase paper), OR
2. Re-number the CSV to UniProt indices (then every row in this README
   becomes wrong vs. the paper labels).

Strip-the-signal-peptide is the convention assumed throughout this
scaffold.

## Sources used (every row cited)

### Beneficial (clinical / lab ESBL gateways + global stabiliser)

| Mutation | Source |
|---|---|
| `M182T` | Huang & Palzkill 1997 *PNAS* — PMID 9050851; global stabilising mutation that compensates fold-stability defects of resistance mutations |
| `M69L` | Stiffler/Hekstra/Ranganathan 2015 *Cell* DMS — PMID 25723163; IRT-class beneficial under amp+inhibitor, AmbF table |
| `G238S` | Hall & Barlow 2004 *J Mol Evol* — PMID 15136889; classic ESBL gateway, TEM-19 |
| `E104K` | Hall 2002 — PMC2566518; TEM-17 ESBL component, additive with G238S |
| `R164S` | Hall 2002 — PMC2566518; TEM-12 ESBL component, Omega-loop expansion |

### Neutral (DMS near-mean-fitness controls)

| Mutation | Source |
|---|---|
| `N175S` | Stiffler/Hekstra/Ranganathan 2015 DMS — PMID 25723163; buffer-region |
| `V216A` | Stiffler/Hekstra/Ranganathan 2015 DMS — PMID 25723163; buried small→small |
| `A237T` | Stiffler/Hekstra/Ranganathan 2015 DMS — PMID 25723163; near G238S but conservative |

### Deleterious (active-site catalytic ablations)

| Mutation | Source |
|---|---|
| `S70A` | Strynadka 1992 *Nature* — PMID 1538780; serine nucleophile ablation |
| `K73A` | Lenfant 1991 — PMID 1825695; general-base K73 ablation |
| `E166A` | Strynadka 1992 *Nature* — PMID 1538780; Omega-loop acid/base |
| `N170A` | Damblon 1996 *PNAS* — PMID 8757758; oxyanion-hole H-bond donor |
| `K234A` | Lenfant 1991 — PMID 1825695; SDN-loop Lys |
| `S130A` | Lamotte-Brasseur 1991 — PMID 1893393; covalent-intermediate hydrolysis |

## Substrate / ligand

TEM-1 hydrolyses β-lactam antibiotics. A reasonable default substrate
for EvoLiEZ docking is **benzylpenicillin** (penicillin G):

```
SMILES: CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O
```

For ESBL-style benchmarking, **cefotaxime** is the canonical
third-generation cephalosporin discriminator
(`CO/N=C(\C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(COC(C)=O)CS[C@H]12)c1nsc(N)n1`).

## Validate sequence consistency before using

```
bash scripts/fetch_target_fasta.sh P62593 tem1_betalactamase \
    S70 K73 E166 G238 N170
# Then strip the 23-aa Sec signal peptide from target.fasta if you
# want Ambler numbering to match the CSV directly.

evoliez bench -c <YOUR_TEM1_CONFIG>.yaml \
    --benchmark examples/tem1_betalactamase/benchmark.csv \
    --allow-no-overlap
```

`evoliez bench` runs `validate_benchmark`, which checks WT-letter
consistency. A mismatch means either the numbering scheme isn't what
you assumed, or the FASTA still has the signal peptide — fix the
input, do not silently override.

## Extending this scaffold

The Stiffler DMS is publicly available (PMID 25723163 supplementary
data). To add more rows:

1. Take a single-codon mutation with a clean fitness label (>1 = beneficial,
   ≈1 = neutral, <0.3 = deleterious) under either amp or amp+inhibitor.
2. Keep the row in Ambler numbering.
3. Append to `benchmark.csv` with `mutation,label,activity,source`.
4. Cite the DMS table / paper / PMID inline.

## See also

- [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) — full benchmark harness.
- [`examples/pseudomonas_fdh/`](../pseudomonas_fdh/) — sibling scaffold (cofactor-switch case).
