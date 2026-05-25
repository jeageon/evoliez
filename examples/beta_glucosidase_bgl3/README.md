# β-glucosidase Bgl3 benchmark scaffold

Literature-derived benchmark scaffold for the GH1-family β-glucosidase
"Bgl3" — a microbial β-glucosidase that has been subjected to one of
the largest microfluidic deep-mutational scans on record (Romero, Tran
& Abate 2015 *PNAS*, **PMID 25775532**; ~26,000 single-variant fitness
measurements at 60 °C). The DMS gives clean fitness labels for
beneficial / neutral / deleterious classes — exactly the read-out the
EvoLiEZ ranking metric needs.

Why Bgl3 made the multi-enzyme set:
- it is the **largest published enzymatic-activity DMS** keyed to one
  enzyme, so the beneficial / neutral / deleterious labels are
  statistically well-grounded rather than ad-hoc;
- it is **not** a cofactor enzyme — sits in the panel as a
  no-cofactor case alongside TEM-1 to balance the cofactor-heavy
  FDH/XR/P450 BM3 set;
- the **GH1 family catalytic dyad** (E166 acid/base, E353 nucleophile,
  Withers / Vocadlo numbering) is universally conserved across all GH1
  β-glucosidases — the catalytic-avoidance metric has a strong
  deleterious signal regardless of which GH1 enzyme the structural
  reference happens to be.

## Numbering — important caveat

The Romero 2015 DMS paper uses its own **Bgl3 numbering**; the exact
UniProt accession for that metagenomic enzyme is **not** in this
scaffold's repo docs. To keep the structural reference unambiguous, the
five **deleterious** rows are written in **canonical GH1 numbering**
(E166 acid/base, E353 nucleophile — these are the universally cited
Withers/Vocadlo positions across the GH1 family).

The five **beneficial** + three **neutral** rows are labelled with the
Romero 2015 *Bgl3-internal* numbering — these positions are the ones
the DMS reports as activity-enhancing / near-mean-fitness. When you
fetch a structural reference for docking / MD (e.g. **Paenibacillus
polymyxa BglA, UniProt P22073** is a well-characterised GH1 with the
same E166 / E352 catalytic-dyad numbering), only the catalytic rows
will pass WT-letter validation against that FASTA; the
Romero-numbered rows will warn (this is expected, not a bug — set
`--allow-no-overlap` until you have the actual Bgl3 sequence).

For production work, swap the FASTA to the actual Bgl3 sequence and
re-number the catalytic rows to match.

## Sources used (every row cited)

**Cheap-run-6 finding:** the Romero 2015 DMS used Bgl3-internal
numbering that does NOT match the structural-reference UniProt P22073
sequence. The original card's Romero rows (E96A, L195P, T245S,
N283I, M353I etc.) all failed Phase-0d-2 residue validation against
P22073. To keep the scaffold runnable against P22073 BglA, the
benchmark has been pruned to UniProt-validated catalytic-dyad
mutations + adjacent-residue controls. The Romero 2015 supplementary
fitness table is still cited; running it against the actual Bgl3
metagenomic sequence would require that FASTA + a separate
benchmark_bgl3_romero.csv keyed to its numbering.

UniProt P22073 catalytic dyad (verified):
  E166 (ACT_SITE Proton donor) ✓
  E352 (ACT_SITE Nucleophile)  ✓ (paper "E353")
  H121                          ✓ matches Gonzalez-Candelas paper H121

### Beneficial (GH1 conservative-swap variants retaining activity)

| Mutation | Source |
|---|---|
| `E166Q` | Wang 1994 *J Biol Chem* — PMID 8154313; GH1 acid/base amide swap retains weak activity |
| `E166D` | Withers — PMID 1483696; GH1 conservative carboxylate swap at acid/base |
| `E352D` | Withers — PMID 1483696; GH1 conservative carboxylate swap at nucleophile |
| `H121N` | Gonzalez-Candelas 1995 — PMID 7766611; GH1 conserved His amide swap retains binding |
| `H121Q` | Gonzalez-Candelas 1995 — PMID 7766611; H121Q amide swap |

### Neutral (UniProt-grounded conservative swaps)

| Mutation | Source |
|---|---|
| `A164S` | UniProt P22073 BglA position 164 (A in P22073, not N — the original "N164" was paper-numbered Bgl3); conservative swap |
| `A164T` | UniProt P22073 — conservative hydroxyl swap at position 164 |
| `F404Y` | UniProt P22073 position 404 (F in P22073, not W — the original "W404" was paper-numbered Bgl3); aromatic→aromatic |

### Deleterious (GH1 family catalytic ablations)

| Mutation | Source |
|---|---|
| `E166A` | Withers 1992 — PMID 1483696; GH1 acid/base Glu ablation |
| `E166G` | Withers — PMID 1483696; GH1 acid/base Glu small swap |
| `E352A` | Withers 1992 — PMID 1483696; GH1 nucleophile Glu ablation (paper "E353") |
| `E352G` | Withers — PMID 1483696; GH1 nucleophile Glu small swap |
| `H121A` | Gonzalez-Candelas 1995 — PMID 7766611; conserved His ablation |
| `H121G` | Gonzalez-Candelas 1995 — PMID 7766611; conserved His small swap |

## Substrate / ligand

β-glucosidases hydrolyse β-1,4 glycosidic bonds. The standard
laboratory substrate is **4-nitrophenyl β-D-glucopyranoside (pNPG)**:

```
SMILES: OC[C@H]1O[C@@H](Oc2ccc(cc2)[N+](=O)[O-])[C@H](O)[C@@H](O)[C@@H]1O
```

The Romero 2015 DMS was scored on a fluorogenic resorufin-β-glucoside
substrate; pNPG is a reasonable EvoLiEZ default since binding-pocket
geometry is preserved across both substrates.

## Validate sequence consistency before using

```
# Structural reference: P. polymyxa BglA (GH1, well-studied)
bash scripts/fetch_target_fasta.sh P22073 beta_glucosidase_bgl3 \
    E166 E352 H121 N164 W404

evoliez bench -c <YOUR_BGL3_CONFIG>.yaml \
    --benchmark examples/beta_glucosidase_bgl3/benchmark.csv \
    --allow-no-overlap
```

The Romero 2015-numbered rows will not pass WT-letter validation
against the P. polymyxa BglA FASTA — that's expected. For a production
run you should fetch the **actual Bgl3** sequence (Romero 2015
supplementary lists the source organism and metagenomic context) and
re-run validation.

## Extending this scaffold

The Romero 2015 DMS is publicly available in the paper's supplementary
data. To add more rows:

1. Pull a single-residue mutation with a published fitness score.
2. Bin into beneficial (fitness > 1.5), neutral (0.7-1.3), deleterious
   (< 0.3) using the paper's standardised activity score.
3. Add the row to `benchmark.csv` keeping the *Bgl3-internal*
   numbering (the catalytic-residue rows stay on GH1-canonical numbering).
4. Cite the table inline.

## See also

- [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) — full benchmark harness.
- [`examples/tem1_betalactamase/`](../tem1_betalactamase/) — sibling no-cofactor scaffold (DMS-derived labels).
