# SrCAR / 3-HP production run — modeling notes

**Target:** Carboxylic Acid Reductase (CAR), *Segniliparus* sp., UniProt **E5XP76**
(entry CAR_SEGRC). Full sequence = **1188 aa**. Host (for expression context):
*Corynebacterium glutamicum* ATCC 13032.

**Engineering goal (from CSV):** improve reduction of **3-HP → 3-HPA**
- 3-HP = 3-hydroxypropanoate (substrate)
- 3-HPA = 3-hydroxypropanal (product)
- overall: 3-HP + ATP + NADPH → 3-HPA + AMP + PPi + NADP⁺

---

## Data-quality issue found in the source CSV (FIXED)

The CSV lists the **3-HP substrate SMILES as `OCC(=O)[O-]`**. RDKit parses this as
**C₂H₃O₃⁻ = glycolate (2 carbons)** — a hydroxy-*acetate*, not 3-hydroxy-*propanoate*.
The product 3-HPA (`C(CO)C=O`) is correctly **3 carbons** (HOCH₂CH₂CHO). A carboxylic-acid
reductase reduces –COOH → –CHO with no change in carbon count, so the substrate of a
reaction giving 3-HPA **must be 3 carbons**: 3-hydroxypropanoate = **`OCCC(=O)[O-]`**
(C₃H₅O₃⁻). The CSV SMILES is missing one CH₂.

→ Run uses the **corrected 3-HP** `OCCC(=O)[O-]`. (Other CSV species check out:
ATP C₁₀H₁₆N₅O₁₃P₃, 3-HPA C₃H₆O₂, NADPH dihydronicotinamide.)

---

## Why we model the ADENYLATION (A) domain, not full-length

CAR is a large 3-module enzyme; the reaction proceeds in three physically separated
steps in **different domains**:

| Domain | Residues (this seq, by motif) | Role in 3-HP→3-HPA |
|---|---|---|
| **A (adenylation)** | ~1–660; A3 P-loop `TSGSTGTPKG` @ **265–274** | acid + ATP → acyl-AMP (**substrate specificity is decided here**) |
| **PCP / T** | GGDSL Ppant-Ser @ **702** | carries the acyl group as a thioester |
| **R (reductase, SDR)** | Rossmann `TGSNGWLG` @ **798**; cat. `Y966…K970` | NADPH reduces thioester → aldehyde |

3-HP is a tiny, polar, **non-native** substrate (native CARs prefer medium-chain/aromatic
acids). The bottleneck to making CAR act on 3-HP is **whether the A-domain pocket can bind
and adenylate 3-HP** — an A-domain problem. The R-domain reduces whatever thioester it is
handed (low substrate selectivity). So we model the **A + PCP di-domain (residues 1–720)**
with **3-HP (design substrate) + ATP (adenylation co-substrate)** and redesign the acid
pocket. This concentrates compute on the actual specificity determinant and avoids
mis-folding / ligand-scatter artifacts of a 1188-aa 3-domain Boltz prediction (data quality
is the priority). NADPH/R-domain reduction is a separable downstream problem.

**Truncation:** residues 1–720 (keeps native 1-based numbering; includes the complete
A-domain, the A-sub subdomain, and the PCP Ser702). A-domains fold as independent units
(ANL superfamily / firefly-luciferase / NRPS A-domain fold), so the truncation is clean.

## Catalytic / protected residues
- `S268, T269, K273` — A3 phosphate-binding loop (ATP-phosphate coordination). Protected
  from mutation. Identities verified against the sequence by `doctor`.
- The 3-HP **acid pocket** residues to redesign are discovered data-driven by the 8 Å design
  radius around the docked 3-HP (not hand-picked).

## Adenylation catalytic geometry (for the s10 NAC screen — added pre-s10)
Reaction = in-line nucleophilic substitution at the ATP **α-phosphorus** by the 3-HP
**carboxylate O⁻**. NAC term: d(3-HP carboxylate-O → ATP α-P) with near-linear
O_nuc···P···O_leaving. SMARTS to be verified against the s04 complex atom indices before s10.

## Platform
- Isolated deploy: `/mnt/data/jglee/EvoLiEZ_car` (my current code via `PYTHONPATH`); main
  server repo + its uncommitted WIP untouched. Launcher: `car/run_car.sh`.
- Real backends: Boltz-2 (complex), mmseqs2/UniRef30 (homologs), mafft (MSA), gnina+DiffDock
  (docking), LigandMPNN (design), ThermoMPNN (ΔΔG), OpenMM+AmberTools (MD/RBFE).
- GPU-first across 4× A6000; CPU capped (16-core budget) for the shared box.
- Output: `/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_prod`; results collected under `car/`.

## Bugs found & fixed during the run
1. **CSV substrate** was 2-carbon glycolate → corrected to 3-carbon 3-HP (above).
2. **`boltz.py:_to_a3m` fed Boltz a gapped-query block MSA relabeled as a3m** (query had
   8861 gap cols). a3m requires the query ungapped. Fixed by restricting every row to the
   query's non-gap columns. **pLDDT 46.6→89.2, confidence 0.50→0.90.** Affects every Boltz
   call (s04/s06b/s08b) → quality lift across the pipeline.
3. **s02 had no resume `load()`** → every staged --resume re-ran the 22-min mmseqs search.
   Added a DB-backed `s02.load()` (restores homologs from Sequence rows).

## Resolved: ATP⁴⁻ MD charges (was the s10 blocker)
`params/atp_4minus/ATP.fixed.mol2` — 43 atoms (H-added), GAFF2, single-point AM1-BCC at the
Boltz-bound pose (`sqm maxcyc=0`), charge sum −4.0000. Built by: extract ATP heavy atoms from
s04 model_0, RDKit `AssignBondOrdersFromTemplate` + `AddHs(addCoords=True)`, then antechamber.
(The tetra-anionic triphosphate converges as a single-point but crashes on gas-phase
optimization — same lesson as NADP.)

## PRE-s10 checklist (needs a config edit + `check_run_fingerprint.py --patch`)
- Wire `charges_mol2: params/atp_4minus/ATP.fixed.mol2` onto the ATP ligand.
- Add the adenylation NAC to `validation.md.reactive_geometry` (SMARTS VALIDATED against the
  real ligands — each matches uniquely):
  ```yaml
  reactive_geometry:
    enabled: true
    donor_smarts: "[OX1-]"                 # 3-HP carboxylate O (nucleophile), idx 0
    acceptor_smarts: "[PX4]([OX2][CX4])"   # ATP alpha-P only (beta/gamma excluded), idx 0
    donor_idx: 0
    acceptor_idx: 0
    transfer_is_h: false                   # O attacks P (no H transfer)
    distance_max: 3.6                      # near-attack O...P (WT rests ~7.9 A -> engineering gap)
    label: 3HP_carboxylate_to_ATP_alphaP
  ```

## Full-length (all-3-ligand) run — extra finding
4. **Boltz-2 affinity head fails on the 3-ligand complex.** `predict_affinity` (single-binder
   model) didn't generate `pre_affinity_*.npz` for the B/C/D (3HP/ATP/NADPH) system →
   DataLoader `FileNotFoundError` at predict time. The 2-ligand A-domain run was fine.
   Fix: `predict_affinity: false` (structure-only; binding via gnina/diffdock + s09). The
   1188-aa + 3-ligand structure diffusion fits in 27 GB (no OOM).

## Reference crystals (user-provided) — 5MST/5MSV/5MSW (S. rugosus CAR, exact target)
- **Validate the model**: 5MST fumarate (substrate analog) contacts = my Boltz 3-HP pocket
  residue-for-residue (W290,F294,H315,S408,loop430-434,P438). Independent confirmation.
- **Real active site** (AMP+FUM contacts): invariant Lys **K629**, R522, H315, D507/Y519,
  substrate loop 429-438. R-domain (5MSV): NADP + phosphopantetheine, SDR dyad Y970/K974.
- Used to set the protected catalytic residues (K629/D507/Y519) so the design can't break
  catalysis. To also anchor s10 (3-HP->fumarate pose, ATP->AMP pose). refs on server car/refs/.

## Boltz-vs-crystal RMSD validation (car/rmsd_vs_xtal.py)
- GLOBAL whole-domain CA-RMSD is HIGH (A-domain vs 5MST 10.6 A, R vs 5MSV 6.7 A) — this is
  CAR's known large DOMAIN DYNAMICS (subdomains/PCP in different relative orientations; the
  crystals capture specific catalytic states), NOT a fold error.
- LOCAL functional motifs are near-atomic vs 5MST: **substrate loop 429-438 = 0.32 A**,
  loop+H315/S408 = 0.36 A, A3 loop 265-275 = 0.78 A. => the 3-HP design pocket + ATP site
  are predicted with essentially crystallographic accuracy; the design rests on solid geometry.

## Full-length run — bugs fixed 5-7
5. **Boltz OOM** on 1188 aa + 3 ligands (8 parallel diffusion samples > 48 GB). Fix:
   `EVOLIEZ_BOLTZ_MAX_PARALLEL_SAMPLES=1` (serial). s04 then succeeds: pLDDT 87.5, all 3
   ligands in their correct domains (ATP->A3 loop 2.4 A; NADPH->SDR dyad Y970 2.4 A).
6. **s01.load() restored catalytic_positions from stale meta**, not the config -> a config
   change to catalytic_residues wouldn't propagate on --resume (fixed_residues DID reparse).
   Fixed: reparse catalytic from config (parity).
7. **Design mask spanned both domains** (s06 designs around EVERY co-modelled ligand -> the
   R-domain NADPH pocket was designable, off-target for 3-HP). Added
   `mutation_generation.design_around_extra_ligands` (default True); set False here ->
   51 designable positions, ALL A-domain (the 3-HP substrate pocket).

## Run decisions
- s06b `representative_max: 48` (from 80) — tractable ~2.4 h on the 2 free GPUs; robust family
  model. `gpu_pool: []` so GPUs are chosen per-launch via CUDA_VISIBLE_DEVICES (free GPUs only;
  another user's GROMACS holds GPU 0/1).
- **GNN disabled** (no CAR-family checkpoint) — xgboost interaction model + chemistry/MSA/
  LigandMPNN mutation generation carry the design.
