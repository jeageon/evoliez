# EvoLiEZ Scientific Workflow and HTML Report Audit

Date: 2026-05-27

> **Update (2026-06): backend "silent mock" premise superseded.** Several
> findings below assumed a real run could silently degrade to mock/proxy
> artifacts. The backend now **hard-fails** under `backend=real` (`RealToolError`,
> opt out with `allow_mock_fallback`), `doctor` **blocks** on missing required
> tools, real docking/stability use the **full-atom** structure, and `s09`
> redocks the **real per-mutant** structure (see `docs/ARCHITECTURE.md` and the
> "Done / Landed" section of `docs/IMPROVEMENT_ROADMAP.md`). The remaining open
> items here are about the **HTML report layer** (stage_status labels,
> pose-ensemble gating, MSA-scale card, Foldseek implement-or-remove), which were
> not part of that backend work.

## Scope

This audit checks whether the current EvoLiEZ pipeline and the HTML report package
truthfully represent the scientific workflow we want:

1. input ligand and target sequence
2. automatic MSA and homolog discovery
3. structure prediction and docking to determine protein-ligand geometry
4. comparison against homolog or related-enzyme structures
5. ligand atom / enzyme atom interaction fingerprinting
6. enzyme-family-specific machine-learning model construction
7. WT-based candidate generation using Rosetta/FoldX, MPNN, chemistry rules, and
   related generators
8. first-pass screening of generated candidates using the learned model
9. final candidate selection using MD validation

Evidence inspected:

- Current local branch: `feat/family-interaction-model`
- Pipeline files under `src/evoliez/stages/`, `src/evoliez/adapters/`,
  `src/evoliez/features/`, `src/evoliez/ml/`, and `src/evoliez/ranking/`
- HTML report package code on `origin/feat/html-report-package`
- User-provided package:
  `/var/folders/p3/nr6s7m751zgbg5vrxjymjt0w0000gn/T/report_package.zip`

This is a code-level and report-representation audit. It is not a claim that the
specific reported mutants are experimentally valid.

## Executive Summary

The core pipeline is broadly aligned with the desired scientific flow. There are
real stages for input parsing, homolog search, MSA, Boltz complex prediction,
docking, interaction graphing, homolog-family interaction modelling, mutation
generation, reranking, mutant Boltz re-evaluation, non-MD validation, MD, and
final ranking.

The main problem is the report layer: it currently presents several sections as
if the full scientific evidence exists, even when the underlying data is mock,
proxy, unavailable, or a fallback view of the same WT structure. The provided
report package is especially clear:

- `mock_backend: true`
- MSA/homolog scale is only indirectly visible: `n_homologs = 120`
- Foldseek was not used
- `Boltz Complex` and `Pose Ensemble` both point to the same
  `wt_boltz_input_model_0.pdb`
- `Pose Ensemble` is therefore not a real ensemble in this package
- the MSA/homolog section does not show search tool, DB, raw hits, filters,
  Neff, Foldseek status, or representative-homolog count
- the fingerprint section shows derived matrices, not the raw ligand atom to
  enzyme atom coordinate evidence a reviewer would expect

The immediate priority is not more plots. The priority is report honesty:
each section must declare whether it is `real`, `mock`, `proxy`, `fallback`,
`skipped`, or `unavailable`, and must show the scale and evidence counts needed
to judge scientific validity.

## Step-by-Step Audit

| Desired step | Current implementation | Current report representation | Assessment |
|---|---|---|---|
| 1. Ligand + sequence input | Implemented in `s01_input_preprocess.py`; sequence is read/validated, ligand parsed, ligand SMILES and atom ids persisted. | Ligand 2D figure is generated from SMILES with RDKit. | Mostly correct as input QC, but not enough as mechanistic evidence. Needs ligand name, charge/protonation state, source hash, and catalytic atom overlay. |
| 2. MSA + homolog discovery | Implemented via `s02_homolog.py`, `s03_msa.py`, `adapters/msa_tools.py`; real modes support MMseqs2, jackhmmer, BLASTp; mock creates synthetic homologs. | Conservation heatmap and identity histogram. | Partially represented. Report hides scale and search method. Foldseek is configured but not implemented in the search path. |
| 3. Structure prediction + docking | WT Boltz complex in `s04_complex.py`; WT reference docking in `s05_docking.py`; adapters exist for Vina/GNINA/DiffDock. | Boltz complex full view and pocket view. | Concept is implemented, but report should distinguish Boltz model, docking pose, and fallback viewer. Current package shows one WT PDB repeatedly. |
| 4. Compare homolog structures | `s06b_interaction_model.py` picks homolog representatives and predicts representative complexes. | No clear homolog structure comparison panel. | Underrepresented. The report says homologs/pose ensemble, but does not show representative homolog structures, structural alignment, contact conservation, or homolog-vs-WT comparison. |
| 5. Fingerprint ligand/enzyme atom interactions | `s06_graph.py`, `features/graph.py`, `features/interaction_descriptor.py`, PLIP adapter, and ML datasets create contacts and fixed-length fingerprints. | Fingerprint heatmap and top contacts. | Partially correct. The implemented fingerprint is an aggregated descriptor, not a transparent table of ligand atom coordinates, enzyme atom/residue coordinates, distances, and interaction types. |
| 6. Family-specific ML model | `s06b_interaction_model.py` trains a self-supervised interaction model; `ml/interaction_model.py` uses XGBoost, logistic, or heuristic fallback. | Weakly represented through downstream scores. | Implemented, but report should show training rows, positive/alternative/outlier/decoy counts, model kind, holdout AUROC, and fallback status. |
| 7. WT-based candidate generation | `s07_mutation_gen.py` supports chemistry rules, MSA sampler, and LigandMPNN. `s09_nonmd_validation.py` uses FoldX/Rosetta for validation. | Mutation map and candidate tables. | Partially aligned. Rosetta/FoldX are currently validation/stability filters, not candidate generators. If we claim Rosetta/MPNN/CA generation, the generation stage must expose each generator separately. |
| 8. First-pass screening using learned model | `s08_reranker.py` uses the interaction model plus MSA, interaction, Boltz-delta proxy, and optional GNN; `s08b_mutant_boltz.py` upgrades top candidates with real/mock mutant Boltz. | Score waterfall and final candidate evidence classes. | Implemented but report must expose proxy vs real. In the provided package many candidates have `boltz_delta_source=proxy`; this should be obvious in the report. |
| 9. MD final selection | `s10_md.py` runs MD or records skipped/failed states; `s11_final_ranking.py` merges MD scores into final ranking. | MD plots and final library. | Implemented, but report must distinguish `md_did_run`, `md_passed`, skipped, failed, mock, force field, replicas, timestep, and protocol. |

## Detailed Findings

### 1. Input Section

Implemented:

- Sequence and ligand ingestion exist.
- Catalytic/fixed/binding-site residue tokens are parsed.
- Residue token mismatch warnings are implemented.
- Ligand atom ids are persisted for atom-index locking.
- The HTML report can render an RDKit 2D ligand graphic.

Problems:

- The ligand graphic is a 2D topology QC image, not binding evidence.
- The current report title falls back to raw truncated SMILES for NADP-like input.
- The report does not show ligand charge/protonation state or input source hash.
- `catalytic_atoms` was empty in the provided package, so the ligand reaction
  center was not marked.

Required improvement:

- Rename this section to "Input ligand 2D topology / provenance".
- Show ligand id, resolved name, SMILES/InChI, charge, protonation assumption,
  atom count, canonical atom ids, and source hash.
- Add catalytic ligand atom overlay when known.

### 2. MSA and Homolog Discovery

Implemented:

- `HomologConfig` has method, database, max sequence count, identity range,
  coverage, e-value, length ratio, clustering threshold, and `use_foldseek`.
- Real sequence search supports MMseqs2, jackhmmer, and BLASTp.
- MSA supports MAFFT and optional remote MSA.
- Conservation, entropy, amino-acid frequencies, PSSM, allowed residues, and
  subfamily-aware annotations are computed.

Problems:

- `use_foldseek` exists in config but no Foldseek adapter/search path is wired
  into homolog discovery.
- Mock homolog generation is capped at 120 by code. The provided package's
  `n_homologs=120` matches this mock cap, not necessarily a real biological
  search result.
- The report does not show raw hit count, filtered hit count, MSA depth,
  effective Neff, database name/version, search tool, e-value, coverage filter,
  identity filter, clustering threshold, or Foldseek status.

Required improvement:

- Add an MSA/Homolog scale card:
  - backend: real/mock
  - sequence-search tool
  - database path/version
  - Foldseek enabled/disabled and database
  - raw hits
  - filtered hits
  - MSA depth
  - Neff or clustered depth
  - identity min/mean/max
  - coverage min/mean/max
  - representative homolog count
  - reason for any fallback

### 3. Boltz Complex and Docking

Implemented:

- WT complex prediction uses Boltz/Boltz-2 in real mode and synthetic mock
  structures in mock mode.
- Boltz diffusion sample count is configurable.
- WT reference redocking exists through Vina/GNINA/DiffDock adapters.

Problems:

- In the provided report package, both Boltz complex viewers point to the same
  WT model file. One full view and one pocket close-up are acceptable, but this
  must be described as two views of the same model.
- The `Pose Ensemble` section also points to the same WT `model_0.pdb`, which is
  not scientifically correct for an ensemble.
- Real Boltz sample parsing records confidence files but currently reuses the
  same ligand atoms for samples; per-model ligand coordinates are not clearly
  parsed into separate sample poses.
- Report code falls back to a 3Dmol viewer when PyMOL ensemble rendering fails,
  but the fallback is a single WT pose.

Required improvement:

- Keep Section 3 as:
  - selected WT Boltz model
  - pocket close-up for the same selected model
  - model index, confidence, pLDDT/iPLDDT, ligand iPTM, affinity, source PDB
- Do not use the same single WT PDB as a `Pose Ensemble`.
- Show `n_boltz_models`, `selected_model`, `source_pdbs`, and `fallback_used`.

### 4. Homolog Structure Comparison

Implemented:

- `s06b_interaction_model.py` selects representative homologs and predicts
  representative complexes.
- Representative count and pose/sample counts are stored in pipeline metadata.

Problems:

- The report does not show homolog representative structures.
- It does not show WT-vs-homolog structural alignment, ligand pose conservation,
  contact conservation, or subfamily split.
- Report discovery looks for homolog complex directories under
  `complexes/representatives`, while `s06b_interaction_model.py` writes
  representative structures under `structures/representatives`. This path
  mismatch can make homolog-structure outputs invisible to the report.
- Documentation says docking pose ensembles are used in `s06b`, but the current
  code path uses Boltz predicted samples and does not call a docking ensemble
  through `interaction_model.docking_method`.

Required improvement:

- Add a "Homolog Structure Comparison" report block:
  - number of representative homologs
  - identities and clusters
  - structure source per representative
  - WT-vs-representative RMSD or aligned confidence
  - ligand pose RMSD to WT
  - conserved contact fraction
  - subfamily-specific contact differences
- Fix path conventions so stage outputs and report discovery agree.
- Either wire `interaction_model.docking_method` into actual docking ensembles
  or rename the implementation as Boltz diffusion-sample based.

### 5. Interaction Fingerprint

Implemented:

- WT interaction graph is built from ligand-proximal residues.
- Contacts include residue index, ligand atom id, distance, interaction type,
  and contact probability.
- A fixed-length interaction descriptor is implemented for ML:
  distance histogram, interaction-type counts, summary distances, and nearest
  shell distances.
- Multi-level datasets are exported under `ml_datasets/`.

Problems:

- The desired scientific evidence says "ligand atom coordinates and enzyme atom
  coordinates". The implemented ML fingerprint is intentionally aggregated and
  does not show the raw coordinate-pair evidence.
- The report fingerprint heatmap is not self-explanatory without a schema.
- The package references source files from the original run path but does not
  include the raw dataset tables in the ZIP.

Required improvement:

- Add an explicit atom-pair contact table:
  - candidate / model / pose id
  - ligand atom id, element, role, xyz
  - enzyme chain, residue id, residue atom or centroid id, xyz
  - distance
  - interaction type
  - contact frequency across poses
  - source structure/model
- Keep the compact fingerprint heatmap, but label it as "ML descriptor view",
  not as raw structural evidence.

### 6. Enzyme-Specific Machine Learning

Implemented:

- Self-supervised family interaction model exists.
- Training rows are built from homolog representative pose fingerprints.
- XGBoost, logistic regression, and heuristic fallback are supported.
- Subfamily holdout AUROC can be computed when there are enough groups.
- GNN dataset export exists, with GNN scoring disabled by default unless a
  checkpoint is available.

Problems:

- The report does not make model provenance prominent enough.
- A heuristic fallback can look like a trained ML model if not clearly labelled.
- The provided HTML report does not show train-row counts, consensus/outlier
  counts, model kind, or holdout validation metrics.

Required improvement:

- Add an ML model provenance card:
  - model kind: xgboost/logistic/heuristic/GNN disabled
  - training rows
  - representative homologs
  - poses/samples total
  - positive, alternative, outlier, decoy counts
  - subfamily holdout AUROC
  - feature dimension
  - whether experimental labels were used
  - whether Boltz-derived values were used only as features/weights

### 7. Candidate Generation

Implemented:

- Chemistry-rule candidate generation exists.
- MSA-sampler generation exists.
- LigandMPNN generation exists when configured.
- FoldX/Rosetta are used for non-MD stability validation.

Problems:

- If the intended story says "Rosetta, MPNN, CA build candidate pools", the
  current code only partially matches that. Rosetta/FoldX are validators, not
  generators.
- Default config does not always include LigandMPNN.
- "CA" is not a clearly named generator in the current codebase.

Required improvement:

- Separate candidate generation from validation in report wording.
- Show candidate counts by generator:
  - chemistry rules
  - MSA sampler
  - LigandMPNN
  - Rosetta design, if implemented later
  - CA/active-site chemistry generator, if implemented and named
- Do not imply Rosetta generated candidates unless there is a Rosetta design
  adapter producing mutations.

### 8. First-Pass Screening

Implemented:

- `s08_reranker.py` scores generated candidates using:
  - family interaction score
  - interaction gain
  - MSA permissiveness
  - conservation
  - approximate Boltz-delta features
  - optional GNN score
- `s08b_mutant_boltz.py` re-runs mutant Boltz for top candidates and records
  `boltz_delta_source`.

Problems:

- `s08` uses an approximate mutant complex with unchanged coordinates and
  degraded scores. This is acceptable as a proxy, but it must be labelled.
- The provided package mixes `real`, `proxy`, and blank evidence without a clear
  stage status panel.
- Evidence classes can look more definitive than the underlying computation.

Required improvement:

- Add a screening provenance table:
  - total generated
  - advanced to reranking
  - top-N mutant Boltz evaluated
  - `boltz_delta_source` counts: proxy/mock/real
  - redocking method(s)
  - pose validity pass/fail counts
  - blocked candidate counts and reasons

### 9. MD Final Selection

Implemented:

- MD stage runs OpenMM or records skipped/failed states.
- MD analysis is written per candidate.
- Final ranking includes MD contribution and MD instability penalty.
- Pipeline metadata tracks actually-ran, skipped, and failed MD counts.

Problems:

- Report plots alone do not prove MD was real.
- The provided package has MD status columns, but the visual report needs an
  explicit execution summary.
- Mock or skipped MD must not be visually equivalent to real MD validation.

Required improvement:

- Add an MD execution card:
  - backend
  - engine
  - protocol level
  - solvent
  - timestep
  - force field / ligand parameterization
  - replicas
  - candidates requested
  - candidates actually ran
  - skipped
  - failed
  - passed
  - failure reasons

## Report-Specific Problems Found in the Provided ZIP

The package at
`/var/folders/p3/nr6s7m751zgbg5vrxjymjt0w0000gn/T/report_package.zip`
contains a valid HTML package, but its scientific representation is not yet
strong enough.

Confirmed package facts:

- `visual_manifest.json` has `mock_backend: true`
- `git_sha: 2574ca4`
- `n_homologs: 120`
- identity range: approximately `46.9-95.0%`
- mean identity: approximately `68.4%`
- `Pose Ensemble` source is the same WT model used by `Boltz Complex`
- final candidate rows: `99`
- evidence classes:
  - Strong: `10`
  - Promising: `57`
  - Uncertain: `8`
  - Reject: `24`
- curated benchmark recovery recall is `0.0` at reported K values

Interpretation:

- This package should be read as a mock/demo report, not as scientific
  validation.
- The MSA scale is shallow and synthetic in this run.
- The pose ensemble section is misleading because it is a fallback single WT
  viewer.
- The report needs stage-level scale/provenance before a reader can judge
  scientific validity.

## Priority Fix Plan

### P0: Report Truthfulness and Stage Status

Add a first-class `stage_status` structure to `visual_manifest.json`:

```json
{
  "stage": "s03_msa",
  "status": "real|mock|proxy|fallback|skipped|failed",
  "tool": "mmseqs2",
  "database": "uniref30_2302",
  "inputs": [],
  "outputs": [],
  "counts": {},
  "warnings": []
}
```

Every HTML section should render this status before figures.

Acceptance criteria:

- A mock report cannot visually resemble a real report.
- A proxy feature cannot be presented as real structural evidence.
- A fallback viewer cannot be labelled as an ensemble.
- Every section shows the scale needed to judge reliability.

### P0: Fix Pose Ensemble Representation

Current bad behavior:

- If PyMOL ensemble rendering fails, Section 4 falls back to a single WT PDB.
- In the provided package, Section 4 is therefore not an ensemble.

Required behavior:

- Only render `Pose Ensemble` when there are at least two distinct pose models
  or two distinct ligand coordinate sets.
- If not, show:
  "Pose ensemble unavailable: only selected WT model_0 was available."
- Do not add `04_pose_ensemble_3dmol` for a single WT fallback.

Code areas:

- `src/evoliez/figures/three_d/pose_ensemble.py`
- `src/evoliez/figures/html/builder.py`
- `src/evoliez/figures/discovery.py`

Acceptance criteria:

- Test fails if Section 4 source files contain only the same WT model as
  Section 3.
- Manifest records `n_pose_models`.
- Manifest records `fallback_used`.

### P0: Add MSA/Homolog Scale Panel

Required visible fields:

- backend
- search tool
- database
- Foldseek on/off
- raw hits
- filtered hits
- MSA depth
- Neff
- representative homologs
- identity min/mean/max
- coverage min/mean/max
- e-value threshold
- identity/coverage filters
- clustering threshold

Code areas:

- `s02_homolog.py`
- `s03_msa.py`
- `figures/discovery.py`
- `html/templates/partials/02_msa.html`

Acceptance criteria:

- A reviewer can tell whether an MSA had 120 synthetic homologs or 20,000 real
  UniRef hits without opening provenance JSON.

### P0: Implement or Remove Foldseek Claims

Current state:

- `homologs.use_foldseek` exists.
- Docs mention Foldseek.
- The real search path does not call Foldseek.

Required behavior:

- Either implement Foldseek structural homolog retrieval and merge it with
  sequence homologs before clustering, or remove/disable the claim in reports.

Acceptance criteria:

- If `use_foldseek: true`, the run either executes Foldseek or fails/warns
  clearly.
- Report shows Foldseek database, hit count, and merged hit count.

### P1: Homolog Structure Comparison Section

Add a section or sub-section that shows:

- representative homolog table
- identity and cluster
- predicted structure path
- ligand pose RMSD vs WT
- conserved contact fraction
- contact-gain/loss table
- subfamily-specific residues and contacts

Also fix output/discovery path mismatch:

- stage writes representative structures under `structures/representatives`
- report discovery currently looks under `complexes/representatives`

### P1: Raw Atom-Pair Fingerprint Evidence

Add a bundled CSV and HTML table:

```text
pose_id,candidate_id,ligand_atom_id,ligand_element,ligand_x,ligand_y,ligand_z,
protein_chain,residue_id,residue_name,protein_atom_or_centroid,
protein_x,protein_y,protein_z,distance_A,interaction_type,contact_frequency
```

Keep the current fixed-length ML descriptor, but report it as the derived ML
feature matrix.

### P1: ML Provenance Panel

Add a model card:

- model kind
- fallback status
- train rows
- positive/alternative/outlier/decoy counts
- feature dimension
- representative homologs
- samples per homolog
- validation metric
- label policy
- GNN status

### P1: Candidate Generation Provenance

Show counts by generator and by stage:

- generated candidates
- chemistry-rule candidates
- MSA-sampler candidates
- LigandMPNN candidates
- Rosetta/FoldX validation count
- rejected by stability
- rejected by redocking
- advanced to MD

Do not claim Rosetta generated candidates unless a Rosetta design generator is
implemented.

### P1: MD Execution Summary

The MD section should start with:

- `n_requested`
- `n_actually_ran`
- `n_skipped`
- `n_failed`
- `n_passed`
- engine and protocol
- ligand force field
- parameterization status
- replicas
- timestep

## Suggested Report Section Structure

Recommended HTML report sections:

1. Run Provenance and Stage Status
2. Input QC
3. Homolog Search and MSA Scale
4. WT Structure and Ligand Binding Pose
5. Pose Ensemble Stability
6. Homolog Structural Comparison
7. Interaction Fingerprint Evidence
8. Family Interaction ML Model
9. Candidate Generation
10. Reranking and Non-MD Validation
11. MD Validation
12. Final Library

This separates "what was computed" from "what was selected", and prevents a
single WT fallback viewer from masquerading as ensemble evidence.

## Minimum Acceptance Tests

Add tests that enforce scientific honesty:

1. A mock run must display `mock` in every major report section.
2. `Pose Ensemble` must not render when fewer than two pose models are present.
3. Reusing the same source PDB across Section 3 and Section 4 must produce an
   `unavailable` ensemble state, not an ensemble viewer.
4. MSA section must render search scale metadata.
5. `use_foldseek: true` must either execute Foldseek or emit a visible warning.
6. Homolog representative output paths must be discoverable by the report.
7. Fingerprint report must include raw atom-pair evidence or explicitly state
   that only derived ML descriptors are available.
8. ML section must show model kind and fallback status.
9. MD section must show actual run/skipped/failed counts.

## Bottom Line

The computational pipeline has most of the intended stages, but the report does
not yet communicate the evidence chain at the level required for scientific
review. The highest-risk issue is not missing aesthetics; it is semantic
overclaiming:

- mock homologs presented like real evolutionary evidence
- a single WT PDB presented as a pose ensemble
- proxy mutant structures contributing to strong-looking scores
- missing scale metadata for MSA/homolog search
- Foldseek appearing in configuration/docs without an implemented execution path

Fixing these representation issues should come before adding more report
figures. The report must make uncertainty, scale, fallback, and proxy status
impossible to miss.
