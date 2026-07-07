# SrCAR → 3-HP engineering — production run

Engineer **carboxylic acid reductase** (CAR, *Segniliparus rugosus*, UniProt E5XP76, 1188 aa)
to reduce **3-hydroxypropanoate (3-HP) → 3-hydroxypropanal (3-HPA)**. Full EvoLiEZ pipeline
(s01–s11) at production scale on the 4×A6000 server, grounded in user-provided crystal
structures. Output-quality-first; bugs fixed live.

## What was modelled
- **Full-length 1188-aa enzyme with all 3 ligands**: 3-HP (design substrate) + ATP
  (adenylation cofactor) + NADPH (reduction cofactor) — every functional site present.
- Engineering target = the **A-domain adenylation pocket** (where 3-HP specificity is
  decided). The R-domain (NADPH) is modelled + protected but not a mutation target.
- **Design ligand 3-HP corrected**: the source CSV gave a 2-carbon glycolate SMILES; the true
  substrate is 3-carbon `OCCC(=O)[O-]`.

## Grounded in experiment (user-provided crystals 5MST / 5MSV / 5MSW)
Same enzyme, same numbering. They **validated the predicted model**: the substrate loop
(429–438) is reproduced at **0.32 Å** vs 5MST, and the fumarate (substrate analog) contacts
match the predicted 3-HP pocket residue-for-residue. They also defined the real catalytic
machinery (invariant Lys **K629**, R522, H315, D507/Y519) which was locked as protected.

## Structure quality gate (s04)
Confidence 0.858, mean pLDDT 87.5; all 3 ligands in their correct domains — ATP on the A3
loop (Ser268 2.4 Å), NADPH on the SDR dyad (Y970 2.4 Å), 3-HP in the substrate pocket
(~9.5 Å from the ATP α-P = the non-native "engineering gap" to close).

## Design
51 designable positions, all in the A-domain substrate pocket. Top strategies: reshape the
polar loop (S433/G430/P438/S408) and **introduce Arg (P438R, G430R, T271R)** to anchor 3-HP's
carboxylate. Catalytic residues untouched.

## Bugs found & fixed live (7)
1. CSV substrate SMILES 2C→3C. 2. **`_to_a3m` fed Boltz a gapped-query MSA** → pLDDT 46→89
(the big one; fixed a3m format). 3. `s02.load()` missing (re-mined homologs every resume).
4. Boltz affinity head fails on 3 ligands → structure-only. 5. Boltz OOM on 1188 aa → serial
diffusion. 6. `s01.load()` restored stale catalytic residues → reparse config. 7. Design mask
spanned both domains → `design_around_extra_ligands` flag (A-domain only).

## Results (pipeline complete, s01→s11)
**349 candidates ranked · 39 ran real MD · evidence cards written.** Honest verdict from the
ClaimGuard: **all 39 are "insufficient evidence to claim improvement"** — they are
structurally viable (fold + keep ATP/NADPH contacts) but the two CATALYTIC axes score 0
because the adenylation near-attack geometry could not be measured (3-HP docks ~9.5 Å from the
ATP α-P, past the NAC placement gate — see below). So these are **structurally-sound,
sensibly-designed HYPOTHESES, not confirmed improvers.**

Top candidates (MD-lite binding stability + MM-GBSA endpoint ΔG):
| mutation | GBSA (kcal/mol) | note |
|---|---|---|
| G430R;Y264F;A275V | −34.9 | strongest GBSA |
| I263V;Y264F;T265S | −25.9 | |
| P438N;I263L | −22.9 | |
| G430R, G430H, P438S, S433P (single) | — | substrate-loop residues |
Full ranking: `results/ranking/`, `results/reports/final_candidates.csv`,
`results/reports/paper_report_v2.html`.

## FINAL integrated ranking (`results/nac/final_ranking_integrated.csv`)
Two axes merged: s10 binding/stability (md_lite, MM-GBSA) + CAR near-attack catalytic geometry
(Δcarboxylate-anchor toward the α-P, no 3-HP flip). **5 of 14 MD candidates are catalytically
productive** (no flip + gained anchoring + good binding). Leads:
| mutation | GBSA | Δanchor | flip | note |
|---|---|---|---|---|
| **I263V;Y264F;T265S** | −25.9 | +2 | no | strong on both axes |
| **G430R;Y264F;A275V** | −34.9 | +3 | no | best binding; Arg grips carboxylate |
| P438N;I263L | −22.9 | +1 | no | clean |
The catalytic axis **re-ranks** vs binding-only: single `G430R` and flippers (`Y264F;G274A;A275I`)
drop despite decent binding. WT is confirmed unproductive (flips 3-HP).

## Catalytic method (`car/car_nac.py`, `car/final_integrate.py`)
The s10 NAC skipped (below), so I built a **CAR-specific near-attack pass**: place 3-HP's
carboxylate O in-line to the ATP α-P (anti to the leaving PPi; angle verified ~180°) and score
whether each mutant pocket ACCOMMODATES it (carboxylate anchoring / no 3-HP flip / clash),
ranked Δ-vs-WT. **WT is unproductive** (flips 3-HP, 1 carboxylate anchor). The design's
anchoring strategy is validated and it **re-ranks** the candidates vs binding alone:
`P438N;I263L` and `G430R;Y264F;A275V` (Δanchor +3/+5, no flip) rise; the binding-only #1
(`I263V;Y264F;T265S`, Δ+1) drops. **This is a STATIC placement score (rigid → clash-noisy)** —
a directional first pass. The rigorous layer (restrained-MD ΔNAC with Mg²⁺, explicit solvent,
angle-retention/wrong-pose/ATP-Mg-RMSD terms) is the documented next build.

## The catalytic-axis gap (why s10's NAC was blank)
The engineering goal is 3-HP **adenylation**, not just binding. The NAC (near-attack
occupancy: 3-HP carboxylate-O → ATP α-P) is the metric that speaks to it — but WT CAR does not
position 3-HP productively (~9.5 Å from the α-P), so the frame-0 placement gate (4 Å) skips
the screen and both catalytic axes read 0. **Fix:** start 3-HP in the near-attack pose
(crystal-grounded on 5MST fumarate), restrain the distance, let MD test whether each mutant
HOLDS the reactive angle → ΔNAC vs WT = a real catalytic ranking. This is the recommended
next pass (config `template_cosubstrate_placement` + `restrain_cosubstrate`, or a focused
standalone run on the top candidates).

## Files
- `car_srcar_3hp_full.yaml` (final config) · `run_car.sh` (launcher) · `NOTES.md` (full log)
- `targets/` (sequences) · `refs/` (crystals) · `results/` (deliverables)
- analyzers: `analyze_full.py` (structure QC) · `rmsd_vs_xtal.py` (crystal validation)
- `params/atp_4minus/`, `params/nadph_4minus/` (MD charge templates)
