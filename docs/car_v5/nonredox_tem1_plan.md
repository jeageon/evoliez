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

## Follow-up #2 — protein-nucleophile NAC extension — **IMPLEMENTED**
TEM-1's nucleophile is a **protein** residue (Ser68 Oγ), not a ligand atom as in CAR (3-HP
carboxylate O) or FDH (formate). The NAC layer now accepts a **protein catalytic-residue atom** as the
donor (a generic serine-hydrolase / protease feature, not TEM-1-specific):
- `ReactiveSpec.donor_protein` = `'RESNAME:ATOM[:RESNUM]'` (e.g. `SER:OG:68`) + `ReactiveGeometryConfig.donor_protein`.
- `resolve_protein_donor_index(topology, spec)` resolves the nucleophile from the **MD topology**
  (duck-typed OpenMM Topology API), and `resolve_reactive_indices(..., topology=)` takes the donor
  from the protein and the acceptor (scissile carbonyl C) from the ligand.
- The read-only angle is the **Nuc–C(=O)** (Bürgi–Dunitz-like, `_acceptor_double_bonded_oxygen`); a
  protein O/S nucleophile is coerced to `transfer_is_h=False` so the angle is never the degenerate H
  path. The angle is **never restrained** — it cannot manufacture NAC.
- `configs/tem1_acyl.yaml` now declares `donor_protein: "SER:OG:68"` + the β-lactam carbonyl acceptor
  (`[CX3](=[OX1])[NX3]`), `angle_min: 95°` (Bürgi–Dunitz) — no longer a ligand-internal placeholder.
- Engine wiring: both `resolve_reactive_indices` call sites in `openmm_engine.py` pass
  `topology=modeller.topology`. Tests: parser / topology resolver / `__post_init__` run locally;
  full resolution is RDKit-gated (server).

## Remaining follow-up before it is a real *proof*
1. **Flip to a real backend** (server): `backend: real` + a homolog DB path (as CAR/FDH do on the GPU
   box). Then s01–s11 runs on the real TEM-1 structure/docking/validation path, and the protein-Ser68
   Oγ → β-lactam carbonyl distance/angle becomes a real (server-MD) observable.

## Success criteria (reviewer)
- Real backend runs s01–s11 (or a reduced production path) on TEM-1.
- Reports are ClaimGuard-clean; evidence + uncertainty produced with **no activity claim**.
- Protein-nucleophile NAC gives a Ser68 Oγ → β-lactam carbonyl distance/angle observable — **wired**;
  a real numeric readout is gated on the real-backend server run.

Until (1), the generality claim stays **"foundation + mock proof + one real hard case (CAR)"**, not
"real non-redox proof." The protein-nucleophile geometry is now built and unit-tested, so what remains
is the real-backend execution, not a missing capability.
