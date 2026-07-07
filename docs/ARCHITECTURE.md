# EvoLiEZ — System Architecture

> EvoLigand-Enzyme Engineer (`evoliez`) — a ligand-aware, enzyme-family-specific protein-engineering pipeline.
> This document describes the architecture of the whole codebase (`src/evoliez/`, ~11k LOC).

> ⚠️ **The `sequence + ligand → … → final rank` contract below is the v1 contract.** It is
> being superseded by the **functional-state contract** in
> [ROADMAP_V2_FUNCTIONAL_STATE.md](ROADMAP_V2_FUNCTIONAL_STATE.md) (v2.0 epoch) — read that
> for the target flow (role-tagged reference state → functional-state graph → multi-lane →
> anchored validation → evidence-class library).

---

## 1. The system at a glance

**Purpose.** The input is a pair — *a target enzyme sequence + a ligand (cofactor/substrate)* — and the output is *a ranked list of mutation candidates with weighted, attributable evidence* together with *a focused wet-lab library (default 96-well)*. It is not a single-score predictor but a decision pipeline that fuses evolutionary constraint, ligand atom-level geometry, structural confidence, redocking consistency, fold stability, and short molecular dynamics (MD) into one **multi-objective, interpretable score**.

```
sequence + ligand
 └─> homolog search ──> MSA + evolutionary features
     └─> protein–ligand complex prediction (Boltz-2)
         └─> docking / redocking ensemble
             └─> ligand atom–residue interaction graph
                 └─> mutation generation (LigandMPNN / MSA / chemical rules)
                     └─> family-specific reranking
                         └─> non-MD validation (stability, geometry, redocking)
                             └─> OpenMM MD validation (minimization → MD-lite → short MD)
                                 └─> final multi-objective ranking + focused library
```

**Core design decisions, one line each**

| Decision | Detail |
|---|---|
| Dual backend | `mock` (notebook/CI, no GPU) and `real` (CUDA server) share the **same data flow, schema, scoring, and reports**. Only the heavy-tool adapters are swapped. |
| Label policy | Boltz/docking/MD outputs are used **only as features, sample weights, weak labels, or filters**. Supervised labels are restricted to *experimental values* (enforced by `ml/labels.py`). |
| Determinism | All synthetic data and sampling are seeded (`utils/seeds.py`). Candidate order and scores are reproducible. |
| Interpretability | The final score is an additive decomposition of named positive (+) contributions and (−) penalties. Every candidate carries a "why this rank" rationale sentence. |
| Robustness | When a tool is absent the system raises an actionable error or degrades gracefully to mock. Checkpoint-based resume guarantees fsync durability. |

---

## 2. Core runtime

The system runs on a single **stage-DAG orchestrator** and a **single context object** threaded through every stage.

### 2.1 `Pipeline` — the stage-DAG orchestrator
`src/evoliez/pipeline.py`

- Instantiates `ALL_STAGES` (13 stage classes) in order and runs them linearly.
- `from_stage`/`to_stage` allow partial runs; `resume` reloads artifacts for already-completed stages and skips them.
- On `--dry-run` it sets the `EVOLIEZ_DRY_RUN=1` environment variable but restores it in a `finally` block, so dry-run leniency cannot leak into a later real run in the same process.

```
done check: ctx.is_stage_done(name) AND stage.load(ctx) succeeds → [skip]
otherwise → stage.run(ctx) → ctx.mark_stage_done(name)
```

### 2.2 `RunContext` — the single context bus
`src/evoliez/context.py`

A single object shared by every stage; it owns:

- **Config** (`Config`), **directory layout** (`ProjectPaths`), **DB** (`Store`), **logger**, **resume state** (`_state.json`), and the **in-memory artifact bus** (`artifacts: Dict`).
- Artifact-bus API: `put`/`get`/`require` (rich in-memory objects) plus `persist_meta`/`meta` (small JSON values that survive a process restart).
- **Disk guard**: rejects the run if the filesystem holding the output directory has less than `_MIN_FREE_GB=5.0` free (protects an environment where the server root `/` is full). Bypass with `allow_small_disk=True` for small mock runs.

### 2.3 Resume / reproducibility fingerprint
`RunContext.run_fingerprint()` pins the "identity" of a run as a SHA.

- Covers the input (sequence + ligand), the full config JSON, the backend, the dry-run flag, the **content hash of the GNN checkpoint (SHA256)**, the `evoliez` version, and the ranking-formula version.
- If the fingerprint changes (config/input/backend/software version/retrained GNN weights), the resume checkpoint is invalidated and all stages re-run — **blocking silent reuse of stale artifacts at the source**.
- Writing `_state.json` is **atomic and durable**: write to tmp and `fsync` → `os.replace` → `fsync` the directory. Checkpoints corrupted by power loss or an NFS partial write are discarded and the run restarts from scratch (brick prevention).

### 2.4 Config schema
`src/evoliez/config.py` — Pydantic models that **reject unknown keys** (a typo fails immediately instead of being silently ignored).

- Global `backend` (mock|real) plus per-stage overrides `backends: {"s04_complex": "real", ...}`.
- Main sections: `input`, `homologs`, `msa`, `complex_prediction`, `interaction_model`, `gnn`, `advanced`, `mutation_generation`, `reranking`, `validation` (redocking/stability/md), `output`, `scoring` (weights).
- Supports YAML loading plus dotted-key overrides (`load_config`).

### 2.5 Shared data types
`src/evoliez/types.py` — lightweight, **framework-agnostic** dataclasses (no torch/rdkit dependency) that exist on both notebook and server:
`LigandAtom`, `Ligand`, `Residue`, `ProteinStructure`, `BoltzSample`, `Complex`, `Pose`, `Mutation`, `Candidate`.

---

## 3. End-to-end data flow (13 stages)

Each stage reads artifacts from `ctx` (`require`/`get`) and `put`s its results back. Below is the transition of artifacts on `ctx`.

```
s01_input        : target_sequence, ligand, catalytic_positions, fixed_positions
s02_homolog      : homologs                        (core 40–90% / diverse <40% stratification)
s03_msa          : msa, position_features          (conservation/entropy/PSSM/allowed AA)
s04_complex      : wt_complex                       (Boltz-2 diffusion ensemble)
s05_docking      : reference_atoms, wt_reference_poses
s06_graph        : interaction_graph, contacts, designable_positions, mechanism, ligand_importance
s06b_interaction : interaction_model, ensemble_contacts   (self-supervised family-geometry classifier)
s07_mutation_gen : candidates                       (LigandMPNN + MSA + chemical rules)
s08_reranker     : candidates(+scores), redock_candidates (heuristic/XGBoost fast rerank)
s08b_mutant_boltz: mutant_complexes                 (real Boltz Δ on top-N only)
s09_nonmd        : validated_candidates, md_candidates    (ddG/geometry/redocking filters)
s10_md           : md_candidates(+md_lite_score)    (OpenMM)
s11_final        : ranked_candidates, focused_library, reports
```

### Stage-by-stage detail

**s01 — Input preprocessing** (`stages/s01_input_preprocess.py`)
Validates and normalizes the sequence, ligand, and residue tokens. AA-character validation (non-standard X allowed), minimum length 10 aa, position-range validation, WT-character-mismatch warning. Parses the ligand via `features/ligand`, preserving canonical atom IDs.

**s02 — Homolog search** (`stages/s02_homolog.py`)
Searches with mmseqs2/jackhmmer/blastp via `adapters/msa_tools`. Stratifies by evolutionary distance (core 40–90%, diverse <40%). Warns of a weak evolutionary signal when fewer than 10 homologs are found.

**s03 — MSA & evolutionary features** (`stages/s03_msa.py`)
Aligns with mafft (or the remote MSA API `adapters/remote_msa` = ColabFold). Computes per-position conservation, entropy, gap frequency, allowed AA, and PSSM via `features/evolutionary`. Optionally `features/subfamily` derives per-subfamily specificity divergence.

**s04 — WT complex prediction** (`stages/s04_complex.py`)
Runs `adapters/boltz` (Boltz-2) as a **diffusion-sample ensemble** (`diffusion_samples`). Produces confidence/affinity, per-token pLDDT/PAE/PDE, and ensemble contact frequency — all used **only as features / sample weights / weak labels**.

**s05 — Reference docking** (`stages/s05_docking.py`)
Establishes WT reference ligand poses with multiple engines (vina/gnina/diffdock) — the anchor for later RMSD comparison. Each pose carries a score, reference RMSD, and cluster.

**s06 — Interaction graph & design mask** (`stages/s06_interaction_graph.py`)
Builds a ligand atom–residue heterogeneous graph via `features/geometry` + `features/graph`. Designable positions = within the ligand-proximity radius (design radius) + second shell, *excluding catalytic / fixed / highly-conserved (above threshold) residues*. Optionally annotates catalytic mechanism (`features/mechanism`, M-CSA), ligand-atom importance (`features/ligand_importance`), and PLIP interaction fingerprints (`adapters/plip`).

**s06b — Family interaction-geometry model (self-supervised)** (`stages/s06b_interaction_model.py`)
The core module that learns a family-consistency signal without experimental labels (see §4 ML). Per representative homolog: structure prediction + ligand-pose ensemble docking → extract a ligand-atom interaction-distance fingerprint → train a classifier on statistical consensus poses (positive) vs outliers/decoys (negative). A **subfamily-holdout AUROC** report defends against consensus circularity.

**s07 — Mutation generation** (`stages/s07_mutation_gen.py`)
Three generators: ① chemical rules (swap from amino-acid pools by ligand-atom role), ② MSA sampler (frequent AA per position), ③ LigandMPNN (`adapters/ligandmpnn`, ligand-aware neural design). After deduplication, attaches MSA permissiveness and caps with `max_candidates`.

**s08 — Reranker (fast pass)** (`stages/s08_reranker.py`)
Scores mutants with evolutionary, interaction, and structural features. **WT-delta approximation** (`_approx_mutant_complex`): instead of deep-copying a 471-residue complex per candidate, it replaces only the mutated residue and scales Boltz metrics by disruptiveness. Defaults to an interpretable heuristic; switches to supervised XGBoost when ≥8 experimental labels exist. Integrates the family interaction score (s06b) and an optional GNN score as features. Selects top-N for redocking.

**s08b — Per-mutant real Boltz re-scoring** (`stages/s08b_mutant_boltz.py`)
For cost-effectiveness, re-predicts mutant complexes with real Boltz **only for top-N**, replacing the s08 proxy Δ with the real Δ. Records honest provenance via `boltz_delta_source` (dry-run|mock|real).

**s09 — Non-MD structural validation** (`stages/s09_nonmd_validation.py`)
Fold stability (FoldX/Rosetta, ddG cap default 2.5), catalytic-geometry preservation, and redocking consistency. **Filters**: ddG > cap → reject; ligand escape (RMSD > 4.5 Å) → reject. When ddG is unavailable it is 0.0 (no reward, neutral). Optional mechanism-aware negative-design penalty. Selects the top of the passing set for MD.

**s10 — Molecular dynamics** (`stages/s10_md.py`)
Runs MD via `adapters/openmm_engine`. Prefers the real Boltz mutant structure from s08b, else a WT-derived proxy. Protocol levels 0–3 (minimization → MD-lite → short MD → explicit solvent/replicas). `md/analysis` derives **md_lite_score** from ligand RMSD, contact occupancy, catalytic distance, and energy drift. **A skip is treated as score 0.0 + health_ok=True**, so a coverage gap is not mistaken for "bad biology".

**s11 — Final multi-objective ranking & focused library** (`stages/s11_final_ranking.py`)
Weighted sum via `ranking/score.compute_final_score` → deterministic sort (`-final_score`, `candidate_id`). Optional calibration (`ml/calibration`: uncertainty + GO/investigate/reject recommendation), active-learning focused library (`ml/active_learning`: acquisition score × diversity round-robin → 96-well), 5-level ML dataset export, GNN-training graph dataset, provenance JSON, and reports (CSV/Markdown/PyMOL).

---

## 4. ML subsystem

`src/evoliez/ml/` — the layer where the system's scientific distinctiveness is concentrated.

### 4.1 Data / label policy (`ml/labels.py`) — the most important invariant
Boltz/docking/MD/PLIP outputs may be used **only as features / sample weights / weak labels / filters**; supervised labels are restricted to *experimental values* (activity, kcat, km, thermostability, tm, expression, solubility). `assert_supervised_label_allowed()` enforces this at the code level, and every column of the 5-level dataset is tagged with a role (feature/weak_label/supervised_label/delta/filter). One line; **see [`ML_DATA_POLICY.md`](ML_DATA_POLICY.md)** for the full policy.

### 4.2 Self-supervised family interaction model (`ml/interaction_model.py` + `ml/pose_selection.py`)
The module that becomes the **dominant learning signal** when experimental labels are entirely absent.

1. **Pose selection** (`pose_selection.py`): over the fingerprints of homolog docking poses, computes a **median consensus** plus a **MAD-based robust z-score**.
   - Consensus (z ≤ `pose_select_mad_z`, default 2.5) → positive (soft label, high weight).
   - Alternative band (intermediate z, optional) → soft_y = 0.5, low weight.
   - Outlier (z ≥ `pose_outlier_mad_z`, default 4.0) → negative.
   - **Synthetic decoys**: chemically impossible *easy* decoys plus *hard* decoys made by partially substituting the consensus.
2. **Classifier** (`interaction_model.py`): features are **only the ligand-atom interaction-distance fingerprint** (msa_membership·identity·pred_score are excluded from features to prevent label leakage). Auto-degrades XGBoost → Logistic → dependency-free heuristic (sigmoid distance). Uses Boltz confidence as a sample weight.
3. **Inference**: in s08 each mutant's approximate complex is scored as `family_interaction_score` — a reranking feature and a weighted term of the final score.

### 4.3 Server-grade E(3) EvoLigand-GNN (`ml/egnn.py`, `graph_dataset.py`, `train_gnn.py`, `gnn_scorer.py`)
Optional and server-only (torch dependency). When disabled / no checkpoint / no torch, it falls back to the heuristic family model (default `gnn` weight 0).

- **Graph** (`graph_dataset.py`): heterogeneous nodes of ligand atoms + nearby residues, with **relative-vector + RBF-distance** edges. Nodes are 21-dimensional (ligand chemistry 9 + evolutionary 5 + **confidence-aware 7**: pLDDT normalized/binned/window-mean/low-confidence segment/iupred/low-complexity). Edge confidence weight = `contact_freq × pLDDT_norm × ligand_iptm × exp(-ipDE/10)`. Low-pLDDT residues are removed *only when far from the ligand (>10 Å) and non-catalytic* — low-confidence loops near the pocket are kept (weighted down).
- **Model** (`egnn.py`): by default **E(3)-invariant (distance-based) message passing**, with an optional **E(3)-equivariant (EGNN coordinate-update)** mode. Seven multi-task heads (contact / interaction-type / permissiveness / native AA / reliability / risk / graph-score) — all weak/pseudo labels, no experimental labels.
- **Training** (`train_gnn.py`): coordinate-noise augmentation (larger jitter for lower pLDDT), contact masking, multi-task loss, AMP, DDP (torchrun) support. The checkpoint stores the graph-geometry parameters so inference rebuilds the identical graph.
- **Inference** (`gnn_scorer.py`): safe load with `weights_only=True`, rebuilds the graph with the same geometry as training, and returns the mean contact-head probability as `gnn_score`.

### 4.4 Auxiliary ML modules
- **`active_learning.py`**: acquisition score = improvement + β·uncertainty − δ·cost. Uncertainty = the mean of 5 independent risk signals. Diversity is enforced by cluster round-robin over (position / chemistry / ligand-target / conservation bucket).
- **`calibration.py`**: candidate uncertainty, recommendation class (strong/uncertain/reject), ECE, and a reliability diagram (rank-score reliability measurement).
- **`benchmark.py`**: internal CSV plus external ProteinGym/FLIP format support. Beneficial-mutation recall@k, deleterious-mutation avoidance, Spearman, AUROC, catalytic protection, binding-site enrichment. Compares baselines (random/conservation/msa/interaction) and per-layer ablations.
- **`datasets.py`**: 5-level table export (pose/residue/edge/mutation/variant) plus a column-role JSON.
- **`pose_validity.py`**: lightweight clash / pocket-containment checks (not a replacement for PoseBusters).

---

## 5. Adapter layer — wrapping external tools

`src/evoliez/adapters/` — each adapter offers `real` (calls the server binary) and `mock` (deterministic synthetic) behind an identical interface. Shared synthetic scaffolding, pocket placement, and minimal-PDB writing live in `adapters/base.py`.

| Adapter | Wrapped tool | real call | mock output |
|---|---|---|---|
| `boltz` | Boltz-2 | `boltz predict <yaml> --diffusion_samples ...` | synthetic complex + Boltz-like metric ensemble |
| `vina` | AutoDock Vina | obabel preprocessing + vina (center/box/exhaustiveness) | deterministic pose (disruption-scaled RMSD) |
| `gnina` | GNINA (CNN docking) | `gnina -r -l --autobox_ligand ...` | mock_redock (optimistic offset) |
| `diffdock` | DiffDock | `python -m inference --protein_ligand_csv ...` | single disrupted pose |
| `docking` | dispatcher | selects vina/gnina/diffdock via config, K-pose ensemble | mock ensemble (~25% outliers) |
| `ligandmpnn` | LigandMPNN | `run.py --model_type ligand_mpnn ...` | per-position chemically-biased sampling |
| `foldx` | FoldX | `--command=BuildModel --mutant-file` | chemistry-based ddG proxy |
| `rosetta` | Rosetta cartesian_ddg | `cartesian_ddg ... -ddg:iterations 3` | reuses the FoldX mock |
| `openmm_engine` | OpenMM | ff14SB + GAFF/Espaloma + GBSA/TIP3P, tiered restraints | disruption-scaled RMSD trajectory |
| `msa_tools` | mmseqs2/jackhmmer/blastp + mafft | search + alignment | deterministic synthetic homologs (subfamily structure) |
| `remote_msa` | ColabFold API | polls `https://api.colabfold.com` | (fallback on network failure) |
| `plip` | PLIP (not yet integrated) | falls back to geometric rules | distance-based interaction classification |
| `disorder` | IUPred2A / MobiDB-lite | `iupred2a.py long` | hydrophobicity + low-complexity heuristic |
| `mcsa` | M-CSA | user-provided JSON | role-prior heuristic |

**Important robustness pattern**: a FoldX failure returns `None` (= "stability unavailable") — it is *not* wrongly rewarded as 0.0 ("most stable"). OpenMM detects an upstream mock's CA-only PDB and records an honest skip. The Vina parser uses a heavy-atom-aware lock that tolerates the hydrogen asymmetry of PDBQT (≈54 vs the canonical ≈70 atoms).

All adapter calls go through `utils/subprocess_utils.py` (logging, timeout, dry-run, `require()` tool discovery).

---

## 6. Feature-extraction layer

`src/evoliez/features/`

| Module | Output |
|---|---|
| `evolutionary.py` | per-position conservation, entropy, gap frequency, AA frequency, PSSM, allowed AA, residue class (pure numpy) |
| `geometry.py` | hot-path distance computation, contact type (hbond/salt-bridge/aromatic/hydrophobic/vdw), catalytic distance, ligand centroid |
| `graph.py` | residue + ligand-atom heterogeneous NetworkX graph (contact/spatial/bond edges) |
| `interaction_descriptor.py` | fixed-length fingerprint per ligand atom (distance histogram + type counts + k-shell) — makes poses from homologs of different lengths comparable |
| `boltz_features.py` | ensemble contact frequency, edge confidence, confidence-weighted contacts, pocket pLDDT |
| `confidence.py` | pLDDT-derived residue confidence (normalized/binned/window-mean/low-confidence segment) |
| `delta.py` | mutant–WT Boltz-metric deltas (confidence/ptm/iptm/pLDDT/affinity/pocket/catalytic distance) |
| `ligand.py` | RDKit parsing (SMILES/InChI, 3D embed, Gasteiger charges, atom chemistry) + synthetic fallback |
| `ligand_importance.py` | per-atom importance (reactive 1.0, charged 0.9, H-bond 0.7, aromatic 0.5, hydrophobic 0.35) |
| `mechanism.py` | catalytic / acid-base / nucleophile / metal-coordinating / cofactor residues, reactive atoms, transition-state geometry score |
| `subfamily.py` | subfamily cluster partition + per-position specificity divergence (highlights cofactor/substrate-switch sites) |

---

## 7. Ranking & scoring

`src/evoliez/ranking/`

### 7.1 Multi-objective final score (`score.py`)
Deliberately **additive and interpretable**: total = Σ(named positive contributions) − Σ(named penalties). Each candidate carries a transparent rationale.

- **Positive contributions (weights)**: ml_mutation (1.0), ligand_interaction_gain (1.0), msa_permissiveness (1.0), complex_confidence (0.5), redocking_consistency (0.5), key_contact_preservation (1.0), stability (0.75), md_lite (1.0), **family_interaction (1.0)**, gnn (0.0, only when trained), catalytic_geometry_preservation (1.0), specificity_divergence (0.5).
- **Penalties**: conservation, catalytic_geometry, ddg_stability, clash, docking_uncertainty, md_instability + **negative design** (neg_catalytic_mut 4.0, neg_conserved_motif 1.5, neg_buried_core_polar 1.0, neg_catalytic_geometry 2.0, neg_overbinding 0.75, neg_pose_inversion 1.0).
- `rationale()` generates a human-readable rationale sentence (position / MSA variability / family observation / top contributions and penalties).

### 7.2 Negative design (`negative_design.py`)
Penalties that explicitly avoid risky mutations: catalytic-residue mutation, high-conservation-motif mutation, buried-core hydrophobic→charged, catalytic-geometry departure, **overbinding** (avoids inhibiting product release), and pose inversion (redocking inconsistency).

---

## 8. Persistence & outputs

### 8.1 Database (`db/`)
- A single SQLite `evoliez.sqlite` at the run root (SQLAlchemy ORM). `store.py` uses **WAL mode** (concurrent read/write on a shared server), a 10-second busy timeout, FK enforcement, and a session context manager.
- Tables (`schema.py`): Project, Sequence, MSAPosition, Structure, ComplexPrediction, DockingPose, InteractionEdge, MutationCandidate, MDSimulation. JSON-like blobs are stored as TEXT to keep a single self-contained DB.

### 8.2 Directory layout (`io/paths.py`)
`ProjectPaths` (frozen) owns every path, so stages never hand-assemble paths.

```
<run>/
  inputs/  homologs/  msa/
  structures/{target,representatives}/
  complexes/{boltz,alternative_models}/
  docking/  interaction_graphs/  mutations/  validation/  md/
  ml_datasets/            # 5-level tables
  datasets/graph_pt/      # graphs for GNN training
  checkpoints/            # GNN weights
  reports/{,figures}/     # final_candidates.csv, final_report.md, focused_library.csv, .pml
  logs/pipeline.log
  evoliez.sqlite          # run DB
  _state.json             # resume checkpoint
```

### 8.3 Reports (`io/report.py`)
- **CSV** (always): `final_candidates.csv` (rank, mutations, final_score, ml_score, ddg_fold, docking_score, md_lite_score).
- **Markdown** (optional): method summary + top candidates + rationale sentences.
- **Focused-library CSV**: 96-well (A1–H12) for synthesis.
- **PyMOL session** (optional): a `.pml` loader.

### 8.4 Provenance record (`io/provenance.py`)
`evoliez_version`, `git_sha` (±dirty), `ranking_formula_version` (currently `2026-05-19.v5`), input/ligand/config SHA1, GNN checkpoint SHA256, tool versions and binary paths, seed/backend/timestamp/platform → a JSON record for papers/patents.

---

## 9. Execution & infrastructure

### 9.1 CLI (`cli.py`)
| Command | Function |
|---|---|
| `evoliez run` | Run the full pipeline (`--backend`, `--resume`, `--dry-run`, `--from/--to`, `--stage-backend s04=real`, `--allow-small-disk`) |
| `evoliez stages` | Print the stage list |
| `evoliez show` | Validate config + print the resolved JSON |
| `evoliez train-gnn` | Train the GNN on an exported graph dataset |
| `evoliez bench` | Benchmark scoring (recall/AUROC/Spearman/calibration) + ablation |
| `evoliez doctor` | Pre-flight check of tools/dependencies/GPU/disk before a real run |
| `evoliez version` | Print the version |

### 9.2 Multi-GPU orchestration
Multi-GPU runs use one **independent job per card** (round-robin, not model-parallel) via `utils/gpu.py` (`GpuPool`) and `workflow/Snakefile` (data_pipeline → train_gnn → rerank_with_gnn). One line; **see [`SERVER_GRADE.md`](SERVER_GRADE.md)** for the GPU-pool selection logic and the Snakemake DAG.

### 9.3 Config profiles (`configs/`)
`default.yaml`, `smoke.yaml` (lightweight), `example_fdh_nadp.yaml`, `server_fdh_nadp.yaml`, `prod_fdh_nadp.yaml` — centered on the FDH/NADP cofactor-switch example.

### 9.4 Diagnostics (`diagnostics.py`)
Status codes OK/WARN/MISSING/BLOCK. Checks Python packages (rdkit, Bio, sklearn, xgboost, openmm, torch, pdbfixer …), CLI tools (boltz, mmseqs, mafft, vina, gnina, foldx …), and environment variables. Exits 1 on a blocking issue.

---

## 10. Robustness / reproducibility principles (cross-cutting)

1. **Determinism**: `seed_everything` + `derive_seed` (FNV-1a-based, stable per-entity seed) — candidate output never changes.
2. **Graceful degradation**: mock or an actionable error when a tool/model is absent. real/mock are designed to pass through the same code paths (clustering, etc.).
3. **Honest gap handling**: stability unavailable = None, MD skip = neutral (0.0, health_ok), Boltz provenance stated (dry-run/mock/real).
4. **Durable resume**: fsync-based atomic `_state.json`, fingerprint invalidation to block stale artifacts.
5. **Resource guards**: disk-free check (protects the server root `/`), Vina/docking hard timeouts (prevents runaway).
6. **Policy enforcement**: the label policy is enforced by code (`ml/labels.py`), not by documentation.

---

## 11. Scope

Implements the **entirety of Phases 0–6** of the spec (scaffold/schema/config → full rule-based pipeline → LigandMPNN → reranker/weak-supervision → MD-lite → short/explicit-solvent MD → experimental-label supervised reranking). External GPU tools are integrated as subprocess adapters (the spec treats them as external references). **Out of scope**: guaranteeing activity without labels, routine FEP, de novo design, wet-lab automation, foundation-model fine-tuning.

---

## Appendix A — Module ↔ spec ↔ adapter map

| Stage | Spec | Module | Adapters used |
|---|---|---|---|
| s01 | §6 | `stages/s01_input_preprocess.py` | `features/ligand` (RDKit) |
| s02 | §7 | `stages/s02_homolog.py` | `adapters/msa_tools` |
| s03 | §8 | `stages/s03_msa.py` | `adapters/msa_tools`, `adapters/remote_msa` |
| s04 | §9 | `stages/s04_complex.py` | `adapters/boltz` |
| s05 | §10 | `stages/s05_docking.py` | `adapters/{vina,gnina,diffdock}` |
| s06 | §11 | `stages/s06_interaction_graph.py` | `features/graph` |
| s06b | §9.2+§13.3 | `stages/s06b_interaction_model.py` | `adapters/{boltz,docking}`, `ml/{pose_selection,interaction_model}` |
| s07 | §12 | `stages/s07_mutation_gen.py` | `adapters/ligandmpnn` |
| s08 | §13 | `stages/s08_reranker.py` | xgboost (optional), `ml/gnn_scorer` |
| s08b | §13 (expert review #2) | `stages/s08b_mutant_boltz.py` | `adapters/boltz` |
| s09 | §14 | `stages/s09_nonmd_validation.py` | `adapters/{foldx,rosetta}`, dockers |
| s10 | §15 | `stages/s10_md.py` | `adapters/openmm_engine`, `md/analysis` |
| s11 | §16 | `stages/s11_final_ranking.py` | `ranking/score`, `io/report` |

Persistence: `db/schema.py` (§17.1), `io/paths.py` (§17.2). Config: `config.py` (§19). Orchestrator: `pipeline.py` (§18.1).
