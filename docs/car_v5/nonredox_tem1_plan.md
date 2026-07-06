# Real non-redox target — TEM-1 β-lactamase (generality: mock → proof)

Reviewer item: move platform generality from *mock* to *proof* with a **real non-redox backend run**.
TEM-1 β-lactamase (class-A serine hydrolase, UniProt P62593) is the chosen target — genuinely
non-redox, non-metal, non-CAR chemistry (nucleophilic acyl substitution).

## What is scaffolded (validated)
- `car/targets/tem1_blat.fasta` — the real 286-aa TEM-1 precursor. Catalytic **Ser68 / Lys71** (the
  class-A **STFK** motif; Ambler Ser70) — tokens match the sequence.
- `configs/tem1_acyl.yaml` — `mechanism_spec` / `nucleophilic_acyl_substitution`, benzylpenicillin
  (penicillin G) as the β-lactam substrate. **Loads, mock e2e s01→s11 completes, final report
  ClaimGuard-clean.** So the framework handles a real non-CAR non-redox target's config uniformly.

## Two follow-ups before it is a real *proof*
1. **Flip to a real backend** (server): `backend: real` + a homolog DB path (as CAR/FDH do on the GPU
   box). Then s01–s11 runs on the real TEM-1 structure/docking/validation path.
2. **Protein-nucleophile NAC extension.** TEM-1's nucleophile is a **protein** residue (Ser68 Oγ), not
   a ligand atom as in CAR (3-HP carboxylate O) or FDH (formate). The current reactive_geometry
   resolves the nucleophile via SMARTS on the *ligand*; the config's donor is therefore a placeholder
   (β-lactam ring N). A real reaction-geometry readout needs the NAC layer to accept a **protein
   catalytic-residue Oγ** as the donor. This is a bounded, generic extension (not TEM-1-specific) and
   is the honest gap for real non-redox reaction geometry.

## Success criteria (reviewer)
- Real backend runs s01–s11 (or a reduced production path) on TEM-1.
- Reports are ClaimGuard-clean; evidence + uncertainty produced with **no activity claim**.
- (Stretch) protein-nucleophile NAC gives a Ser68 Oγ → β-lactam carbonyl distance/angle observable.

Until (1)+(2), the generality claim stays **"foundation + mock proof + one real hard case (CAR)"**, not
"real non-redox proof."
