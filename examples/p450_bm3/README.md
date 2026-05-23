# Cytochrome P450 BM3 (CYP102A1) benchmark scaffold

Literature-derived benchmark scaffold for *Bacillus megaterium*
cytochrome P450 BM3 (UniProt **P14779**, 1049 aa — the heme domain is
the catalytic N-terminal ~470 aa; FMN/FAD reductase domain is fused
C-terminal). The heme-domain crystal structure is **1BU7 / 2HPD**.

P450 BM3 is the "stress test" of the multi-enzyme set: substrate
promiscuity is the whole point of the enzyme, the active site has
been engineered hundreds of different ways by different labs, and
labels in the literature are activity-on-specific-substrate (so they
genuinely depend on which substrate the EvoLiEZ config docks). Treat
this as the **hard case** — recall@K is expected to be lower here than
on the cleaner cofactor-switch enzymes (FDH / XR / TEM-1 / Bgl3).

Why P450 BM3 made the multi-enzyme set:
- it is the most-engineered self-sufficient P450 — Munro / Arnold /
  Wong / Bell groups have published hundreds of mutants;
- the **heme-thiolate Cys400** is the most rigorously documented
  catalytically-essential residue in the whole superfamily — clean
  deleterious label;
- the **F87 / A82 / V78 / L188 / A328** "active-site engineering ring"
  has very strong precedent in the literature (Arnold's 21B3 variant,
  Wong's F87V/A series, Whitehouse's A82F) — clean beneficial labels.

## Numbering

All positions use **P450 BM3 mature-protein numbering** (UniProt
P14779, 1-based, no signal peptide — P450 BM3 is cytosolic). This
matches every published paper.

The heme-domain residues used in this scaffold all sit in the
N-terminal catalytic portion: F87 / A82 / V78 are substrate-channel
gating residues, L188 / A328 are pocket-reshaping residues,
T268 / D251 / E267 are I-helix proton-relay residues, C400 is the
heme-thiolate, R398 / F393 are heme-anchoring residues.

## Sources used (every row cited)

### Beneficial (active-site / substrate-channel engineering)

| Mutation | Source |
|---|---|
| `F87V` | Oliver/Wong 1997, Graham-Lorence 1997 — PMID 9261177; F87 gating Phe, broadens substrates |
| `F87A` | Carmichael & Wong 2001 — PMID 11355923; canonical permissive mutation |
| `A82F` | Whitehouse 2009 / Huang 2007 — PMID 19288486; decoy-molecule design |
| `A328V` | Lewis 2009 / Glieder 2002 — PMID 19260689; distal-pocket reshaping |
| `L188Q` | Glieder/Farinas/Arnold 2002 *Nat Biotechnol* — PMID 12099294; 21B3 variant component |
| `V78A` | Carmichael & Wong 2001 — PMID 11355923; pocket-expanding |

### Neutral (small-effect controls)

| Mutation | Source |
|---|---|
| `T268A` | Yeom/Sligar 1997 — PMID 9224621; I-helix Thr; standard-substrate effect modest |
| `S72A` | Whitehouse/Bell/Wong 2008 *ChemBioChem* — PMID 18348137; near-mean fitness |

### Deleterious (heme + I-helix + heme-anchoring ablations)

| Mutation | Source |
|---|---|
| `C400A` | Yoshioka 2001 / Munro 2002 — PMID 11418624; proximal heme-thiolate; absolute requirement |
| `C400S` | Auclair 2001 — PMID 11589701; Cys→Ser breaks Fe-S coordination |
| `T268N` | Yeom & Sligar 1997 — PMID 9224621; disrupts proton-relay water network |
| `E267A` | Clark 2006 — PMID 16650706; distal acid/base, breaks coupling |
| `D251A` | Yeom & Sligar 1997 — PMID 9224621; I-helix proton donor for Fe-O-O-H |
| `R398A` | Munro 2002 review — PMID 12191604; propionate-anchoring Arg |
| `F393A` | Ost/Munro 2001 — PMID 11551203; tunes Fe-S covalency / heme reduction potential |

## Substrate / cofactors

P450 BM3's substrate envelope is famously broad. For the benchmark,
pick **one substrate per run** and re-score:

- **Native (long-chain fatty acid)**: lauric acid
  `CCCCCCCCCCCC(=O)O`
- **Permissive engineered substrate (e.g. for F87V)**: octane
  `CCCCCCCC` or naphthalene `c1ccc2ccccc2c1`
- **Decoy-molecule case (A82F)**: perfluorinated decoy
  `CCCCCCCCCCCC(=O)O` (lauric) with a per-fluorinated co-substrate

Cofactors are **NADPH** (electron donor) + **heme** (prosthetic Fe-porphyrin).
The catalytic cycle is `NADPH → FAD → FMN → Fe(III)-heme → O2 → product`.
For EvoLiEZ docking the relevant ligand at the heme-domain active site
is the **substrate** (not NADPH; NADPH binds the reductase domain).

## Validate sequence consistency before using

```
bash scripts/fetch_target_fasta.sh P14779 p450_bm3 \
    F87 A82 V78 L188 A328 C400 D251 T268

evoliez bench -c <YOUR_BM3_CONFIG>.yaml \
    --benchmark examples/p450_bm3/benchmark.csv \
    --allow-no-overlap
```

WT letters MUST match — if the FASTA was fetched correctly,
`F87 → F`, `C400 → C`, etc.

## Expectations

This is the panel's hardest enzyme. Reasonable expectations:
- catalytic-residue deleterious rank: should land near the bottom
  (high signal — C400 / D251 / R398 are unambiguous);
- beneficial-mutation recall: weaker than other panels because the
  beneficial labels assume **a specific substrate** that the config
  may or may not dock at the right pose;
- substrate dependence: re-score for each substrate of interest and
  compare; a single recall@K number doesn't tell the whole story.

## Extending this scaffold

The Munro / Whitehouse / Bell papers (PMID 12191604 and follow-ups)
give a curated table of BM3 variants by mutation set + substrate.
Pull rows from there if you want a deeper benchmark, but **tag each
row with the substrate it was scored under** — that's what makes BM3
benchmarks reproducible.

## See also

- [`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) — full benchmark harness.
- [`examples/pseudomonas_fdh/`](../pseudomonas_fdh/) — sibling cofactor-binding-site engineering case.
