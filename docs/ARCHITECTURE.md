# Architecture & spec cross-reference

`evoliez` implements the architecture in
`EvoLigand_Enzyme_Engineer_Complete_MD_Architecture_Plan.md`. Each spec module
maps to one pipeline stage; each external tool to one adapter.

## Stage map

| Stage | Spec | Module | Adapters used |
|---|---|---|---|
| `s01_input` | §6 | `stages/s01_input_preprocess.py` | `features/ligand` (RDKit) |
| `s02_homolog` | §7 | `stages/s02_homolog.py` | `adapters/msa_tools` (mmseqs2/jackhmmer/blastp) |
| `s03_msa` | §8 | `stages/s03_msa.py` | `adapters/msa_tools` (mafft), `adapters/remote_msa` |
| `s04_complex` | §9 | `stages/s04_complex.py` | `adapters/boltz` (Boltz-2) |
| `s05_docking` | §10 | `stages/s05_docking.py` | `adapters/{vina,gnina,diffdock}` |
| `s06_graph` | §11 | `stages/s06_interaction_graph.py` | `features/graph` |
| `s06b_interaction` | §9.2+§13.3 | `stages/s06b_interaction_model.py` | `adapters/{boltz,docking}`, `ml/{pose_selection,interaction_model}`, `features/interaction_descriptor` |
| `s07_mutation_gen` | §12 | `stages/s07_mutation_gen.py` | `adapters/ligandmpnn` |
| `s08_reranker` | §13 | `stages/s08_reranker.py` | xgboost (optional) |
| `s09_nonmd` | §14 | `stages/s09_nonmd_validation.py` | `adapters/{foldx,rosetta}`, dockers |
| `s10_md` | §15 | `stages/s10_md.py` | `adapters/openmm_engine`, `md/analysis` |
| `s11_final` | §16 | `stages/s11_final_ranking.py` | `ranking/score`, `io/report` |

Persistence: `db/schema.py` = spec §17.1 tables; `io/paths.py` = spec §17.2
directory layout. Config: `config.py` = spec §19. Orchestrator: `pipeline.py`
= spec §18.1.

## Backends

`Config.backend` (or per-stage `Config.backends`) selects `mock` or `real` for
every adapter. `mock` produces deterministic synthetic structures/poses/
sequences (seeded in `utils/seeds.py`) so the full data flow, schema, scoring
and reporting run identically on a laptop. `real` shells out (via
`utils/subprocess_utils.py`) to the actual tool and parses its output; missing
tools raise an actionable error or degrade to mock per the spec §23 risk
mitigations.

## Family interaction-geometry model (`s06b`)

Self-supervised, no experimental labels (spec §9.2 + §13.3):

1. Cluster the MSA homologs; take ≤ `representative_homologs` representatives.
2. For each: predict its structure (Boltz-2) and dock the ligand as a
   `poses_per_homolog` **pose ensemble** (`adapters/docking.dock_ensemble`).
3. For each pose, build a fixed-length **ligand-atom interaction-distance
   fingerprint** (`features/interaction_descriptor`): per ligand atom, relative
   distances to the nearest enzyme residue points within `contact_cutoff` +
   interaction-type counts. Extra features: **MSA membership**,
   identity-to-target, and the **per-pose prediction score**.
4. `ml/pose_selection`: robust consensus (median + MAD). Poses within
   `pose_select_mad_z` = positives (family-consistent); beyond
   `pose_outlier_mad_z` + synthetic decoys = negatives. Each retained pose is
   one augmented training row (small pool → many rows).
5. `ml/interaction_model`: train a consensus/outlier classifier
   (xgboost → logistic → dependency-free heuristic). Persisted to
   `interaction_graphs/interaction_model.json`.
6. `s08` scores each mutant's approximate complex with this model
   (`family_interaction_score`) — a reranker feature and a weighted final-score
   term (`ScoreWeights.family_interaction`); the dominant learned signal when
   no experimental labels exist.

## Boltz outputs as features (data policy)

Boltz is run as a **diffusion-sample ensemble** (`complex_prediction.
diffusion_samples`). Its confidence/affinity metrics, per-token pLDDT/PAE/PDE,
ensemble contact frequency and WT–mutant deltas are used strictly as
**features / sample weights / weak labels / filters — never supervised
labels**. The only supervised label is experimental. This is enforced by
`ml/labels.py` and documented in [`ML_DATA_POLICY.md`](ML_DATA_POLICY.md).
Multi-level datasets are exported to `<run>/ml_datasets/` (spec §7).

## Phase coverage (spec §21)

- Phase 0 scaffold/schema/config/example/report ✓
- Phase 1 rule-based full pipeline (s01–s06, s09 stability+geometry, s10 L0, s11) ✓
- Phase 2 LigandMPNN (`adapters/ligandmpnn`) ✓
- Phase 3 reranker + weak-supervision heuristic / XGBoost (s08) ✓
- Phase 4 MD-lite restrained relaxation (`md/restraints`, openmm L1) ✓
- Phase 5 short / explicit-solvent MD + replicas (openmm L2–L3, config) ✓
- Phase 6 experimental-label supervised reranker (s08 `use_experimental_labels`) ✓

## Data flow (artifacts on `ctx`)

```
target_sequence, ligand            (s01)
 -> homologs                       (s02)
 -> msa, position_features         (s03)
 -> wt_complex                     (s04)
 -> reference_atoms                (s05)
 -> interaction_graph, designable_positions  (s06)
 -> interaction_model              (s06b: per-homolog complex+dock ensemble,
                                    consensus/outlier self-supervised classifier)
 -> candidates                     (s07)
 -> redock_candidates              (s08)
 -> validated_candidates, md_candidates       (s09)
 -> md scores on md_candidates     (s10)
 -> ranked_candidates + reports    (s11)
```
