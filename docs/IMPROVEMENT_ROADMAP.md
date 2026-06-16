# EvoLiEZ Improvement Roadmap

Generated: 2026-05-20. Last revised: 2026-06-16.

This document is the single source of truth for the EvoLiEZ improvement plan. It
merges two earlier views:

- the **decision + work-package + PR-sequence** plan (what to build, in what
  order, with which acceptance gates), and
- the **per-stage 2024–2026 SOTA mapping** (the evidence and citation links that
  justify each recommendation).

The goal is not to add every new model. The goal is to improve the reliability
of the EvoLiEZ enzyme/ligand-design funnel while keeping mock/real backend
separation, data-policy rules, and stage-level verification intact.

Legend for per-stage action tables: **E** = effort (S/M/L), **C** = confidence
(low/mid/high), **Pri** = priority (P0 = ship next, P1 = within quarter,
P2 = research bet).

## Scope and Assumptions

- "Our program" means the current EvoLiEZ repo and its documented stage flow.
- Boltz outputs remain features, sample weights, weak ranking signals, or
  filters. They must never become supervised experimental labels.
- Protein-ligand pose quality is judged by physical validity and interaction
  recovery, not by docking score or model confidence alone.
- Expensive methods are used as late-stage filters on small candidate sets.
- Unverified or immature models are listed as research bets, not P0 defaults.

## Done / Landed (2026-06)

A recent hardening session shipped the following. These were previously tracked
as P0/P1 items; they are now closed and removed from the active lists below.

- **Full-atom docking / honest degrade (was P0.1, WP0, First-PR #1).** Real
  docking now resolves a full-atom receptor structure and fails loudly
  (`skipped_no_full_atom_structure`) instead of silently scoring a CA-only
  receptor. `vina`/`gnina`/`diffdock` adapters share the receptor resolver;
  `write_min_pdb` is restricted to mock or ligand-only temp files.
- **Non-MD validation on real/repacked mutants (was P1.4).** `s09_nonmd_validation`
  now redocks the *real* per-mutant Boltz structure rather than a WT-coordinate
  proxy, and the redock backend is decoupled so all configured methods run.
- **`boltz_delta_source` provenance + evidence classes (the structure-source part
  of P0.5).** Reranking persists `boltz_delta_source` (`proxy` / `real_mutant_boltz`
  / `missing`) and the final stage emits evidence classes
  (`Strong` / `Promising` / `Uncertain` / `Reject`).
- **Mock degradation is no longer silent (supersedes a cross-cutting premise).**
  The earlier plan assumed real and mock shared one code path with silent mock
  degradation. That is superseded: the real backend now **hard-fails**
  (`RealToolError`) unless `allow_mock_fallback` is explicitly set, and `doctor`
  **blocks** when a required tool is missing. Mock fallback is opt-in and visible,
  not a silent default.
- **Integrated multi-source homology + ESM prior (part of P1.1).** `s02`
  `gather_homologs` merges SEQUENCE (mmseqs/jackhmmer/blastp or ColabFold remote)
  and STRUCTURE (Foldseek, ProstT5 seq→3Di) homologs into one set with a unified
  subfamily cluster space (`homologs.sources: [sequence, structure]`,
  `homologs.foldseek_database`); `s03` attaches an MSA-free **ESM2** per-position
  variability prior (`msa.esm_enabled`, `PositionFeature.esm_variability`). New
  `adapters/foldseek.py` + `adapters/esm.py` (mock/real); `doctor` requires
  Foldseek when the structure source is enabled. STILL OPEN in P1.1:
  ColabFold-style MSA-depth truncation policy and the Neff/coverage/motif card.

> Note: the monotonic-constraint and exploit/explore-split portions of the
> former P0.5 are *not* shipped and remain active below (see P0.4 and S08/S11).

## Integrated Technical Position

| Area | Integrated decision |
|---|---|
| Boltz-2 affinity head | Use it more strongly as a reranker feature, but preserve the existing ML data policy: never treat it as kcat/Km, activity, or experimental ΔΔG. Add uncertainty and disagreement features. |
| Boltz pocket steering | Verify that `pocket_constraints: true` maps to real Boltz steering/contact constraints, not just metadata. This is a P0 contract test. |
| DiffDock / DiffDock-L | Add a real-backend high-accuracy profile that uses diffusion docking as a sampler, but do not remove Vina/GNINA anchors until PoseBusters and PLIF gates are in place. |
| GNINA 1.3 | Adopt as a rescoring backend and expose `CNNscore`, `CNN_VS`, and `CNN_affinity` as features when available. |
| LigandMPNN | Make it the default generator on real GPU runs once full-atom ligand/cofactor context is confirmed. Keep rule/MSA generators as controls and diversity sources. |
| EGNN upgrade | First add PLIP edge types, physical edge features, and decoy/validity supervision. Consider EquiformerV2/MACE only after this baseline is measured. |
| Stability screening | Add a fast ML stability adapter path, then use FoldX/Rosetta on disagreement or final candidates. Treat ThermoMPNN-D/Stability Oracle/ProstaNet/STAB-DDG as candidate backends depending on availability and benchmark performance. |
| OpenMM MD | P0 is practical correctness and throughput: OpenFF Sage where possible, HMR/4 fs only after stability tests, explicit provenance for skipped parameterization, and multiple replicas for final tier. MACE ML/MM is a pilot, not a broad default. |
| Active learning | Split final 96 variants into exploit and explore libraries when experimental labels exist or after round 1. |

## P0 Plan: Ship First

Target: 1-2 weeks. These are high-leverage changes that either remove silent
failure modes or improve ranking without changing the scientific contract.

### P0.1 Boltz-2 Steering and Ensemble Contract

Current state:

- `complex_prediction.pocket_constraints: true`
- `complex_prediction.predict_affinity: true`
- `complex_prediction.diffusion_samples: 5`

Implementation:

- Add a contract test around `adapters/boltz.py` output YAML/CLI arguments:
  `pocket_constraints` must map to real pocket/contact constraints (Boltz-2's
  pair-rep scaling / contact map API) supported by the installed Boltz version,
  not just metadata.
- Increase real-backend WT diffusion samples from 5 to 25 in server configs.
  Keep smoke configs small.
- Persist per-sample:
  - ligand pose
  - confidence/affinity metrics
  - contact frequency
  - ensemble disagreement
- Feed ensemble spread directly into `s06b_interaction_model` as family-geometry
  consensus input (Boltz-sample-style multi-state coverage).

Important constraint:

- Boltz affinity is a feature and weak ranking signal only. It is not a
  supervised label and not a direct kcat/Km substitute.

Verification:

- Snapshot test for Boltz input with pocket/contact constraints.
- Regression on `runs/smoke` and `runs/demo_fdh_nadp`.
- Report ensemble disagreement for every WT complex.

Why / sources:

- Boltz-2 affinity head: Pearson R ≈ 0.95 on Boltz-ABFE; on CASP16 blind
  affinity it beats teams using weeks of custom physics+ML, at ~20 s/GPU vs
  6–12 h / ~$100 for ABFE.
- Boltz-1x → Boltz-2x **steering** adds physics-based potentials at inference
  (no retraining): kills clashes, fixes chirality, enforces contact/pocket maps.
  Pocket steering is the relevant feature for us.
- **Boltz-sample** (bioRxiv 2026.01) recovers alternative conformational states
  and ensemble coverage that single-sample inference misses — matters for
  enzymes with productive/non-productive sub-states.
- **FoldBench** ranks AF3 ≥ Boltz-2 / Chai-1 / Protenix / HelixFold-3 overall,
  but allosteric protein-ligand systems remain challenging — orthogonal
  ensembling helps.
- Boltz-2: https://pubmed.ncbi.nlm.nih.gov/40667369/ ·
  https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full.pdf ·
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/
- AlphaFold 3: https://www.nature.com/articles/s41586-024-07487-w
- Boltz-2 reliability caution: https://arxiv.org/abs/2603.05532
- Boltz-ABFE pipeline: https://www.bio-itworld.com/news/2025/10/10/boltz-upgrade--recursion-researchers-release-pipeline-combining-ai-binding-model-with-absolute-binding-free-energies
- Boltz-sample / pair-rep scaling: https://www.biorxiv.org/content/10.64898/2026.01.23.701250v1
- FoldBench: https://www.nature.com/articles/s41467-025-67127-3

### P0.2 Real Docking Profile: Diffusion Sampler + GNINA Rescoring + Vina Anchor

Integrated decision:

- The safer implementation of "switch real docking to DiffDock/DiffDock-L" is a
  high-accuracy real profile:
  - diffusion docking for pose sampling
  - GNINA 1.3 for CNN rescoring
  - Vina/GNINA as physics/scoring anchors
  - PoseBusters and PLIF gates before ranking

Implementation:

- Add config profile, for example `configs/server_high_accuracy.yaml`:
  - `interaction_model.docking_method: diffdock` (keep `vina` for mock)
  - `validation.redocking.methods: [diffdock, gnina, vina]`
- Add GNINA rescoring mode:
  - prefer GNINA 1.3 `cnn=fast` when available
  - persist `CNN_VS` / `CNN_affinity` into pose/reranker data
- Use all configured redocking methods (the DL-sample → physics-rescore → cluster
  pattern), not only the first method.
- Cluster poses after rescoring and keep method-level disagreement.

Verification:

- Redocking benchmark tracks RMSD, PoseBusters pass rate, PLIF recovery, and
  method disagreement.
- No candidate can be labeled "strong" if all physically valid poses disagree
  on key catalytic/cofactor contacts.

Why / sources:

- **DiffDock-L** improves blind-docking pose accuracy up to +50% vs DiffDock and
  Vina-class baselines; confidence correlates with < 2 Å RMSD. The DiffDock-L
  pose *distribution* is what the s06b family-geometry model is trying to learn.
- **GNINA 1.3** (Mar 2025): PyTorch backend, retrained on CrossDocked2020 v1.3,
  knowledge-distilled `CNN_VS`; consistently beats Vina and matches commercial in
  independent VS benchmarks.
- **DiffDock-Glide / Blind-VS** (bioRxiv 2025): DL pose-sampling + classical
  scoring is the dominant pattern at scale — matches "sample with DL, rescore
  with physics".
- Calibrate DiffDock-L confidence on FDH/NADP-like cofactors (reported
  cofactor-class confidence drift).
- GNINA 1.3: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-00973-x ·
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11874439/ ·
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12388557/
- DiffDock-L overview: https://310.ai/blog/diffdock-but-better
- DiffDock-Glide hybrid: https://www.biorxiv.org/content/10.1101/2025.06.02.657461v1.full
- Blind-VS at scale: https://www.biorxiv.org/content/10.1101/2025.10.10.681617v1.full
- DynamicBind: https://www.nature.com/articles/s41467-024-45461-2
- FlowDock: https://arxiv.org/abs/2412.10966

### P0.3 LigandMPNN as Real-Backend Primary Generator

Implementation:

- For real GPU/server configs, include `ligandmpnn` in
  `mutation_generation.methods` (adapter already gates by GPU presence). The
  current default `[chemistry_rules, msa_sampler]` underweights ligand geometry.
- Preserve `chemistry_rules` and `msa_sampler` as controls/diversity sources.
- Add preflight checks:
  - full-atom protein structure exists
  - ligand/cofactor atoms are present and atom-index locked
  - catalytic and fixed residues are excluded
- Add multi-state design mode when structures are available (run apo + holo +
  transition-state/product/substrate proxy and intersect designs) — avoids
  designs that bind only the ground-state pose.

Verification:

- Native/homolog sequence recovery report for ligand-contact residues.
- Candidate generation report includes source method and reason for fixed
  residues.
- No catalytic or fixed residues are mutated unless explicitly allowed.

Why / sources:

- **LigandMPNN** beats ProteinMPNN + Rosetta on ligand-interacting residue
  recovery: 63.3% vs ~50% (small molecules), 77.5% vs 36–40% (metals); 100+
  wet-validated designs, up to 100× affinity gain in redesigns.
- **EVOLVEpro** (Science 2025): pLM-guided in silico DE delivers multi-round
  wins with sparse labels.
- LigandMPNN: https://www.nature.com/articles/s41592-025-02626-1
- EVOLVEpro: https://www.science.org/doi/abs/10.1126/science.aea1820

### P0.4 Reranking Reliability: Monotonicity + Evidence Class

Current risk:

- Fast `s08_reranker` can use proxy mutant complex geometry.
- Proxy-derived candidates can look more certain than they are.

(The `boltz_delta_source` provenance and the four evidence classes themselves are
already shipped — see Done / Landed. What remains active is the monotonic
constraints and the exploit/explore split.)

Implementation:

- Add monotonic constraints to XGBoost when labels exist (XGBoost has this
  built-in; reduces overfit on tiny label sets):
  - `ddg_fold` worse -> score cannot improve
  - clash/invalidity worse -> score cannot improve
  - MD instability worse -> score cannot improve
- Require `real_mutant_boltz` or mark as `Uncertain` for final top candidates
  above a configurable rank threshold.
- Split final 96 into exploit/explore pools when uncertainty is available.

Verification:

- Final report shows why each candidate received its evidence class.
- Proxy-only candidates cannot silently enter the strongest class.
- Calibration metrics are reported when experimental labels exist.

Why / sources:

- Calibrated regression with monotonic constraints outperforms plain XGBoost
  when features have known physical sign (e.g. ΔΔG_fold ↑ → fitness ↓).
- **ALDE** (Active Learning-assisted Directed Evolution): uncertainty-aware
  acquisition lifted yield 12% → 93% in 3 rounds for a 5-residue active-site DE
  — direct match for the reranker → wet → relabel loop.
- ALDE: https://www.nature.com/articles/s41467-025-55987-8
- EVOLVEpro: https://www.science.org/doi/abs/10.1126/science.aea1820
- ML data policy in repo: `docs/ML_DATA_POLICY.md`

### P0.5 OpenMM Practical Upgrade

Implementation:

- Prefer OpenFF Sage (openff-2.x) for small-molecule ligand parameterization
  when available; keep GAFF/espaloma fallback and explicit skip reasons for
  unsupported cofactors. Already supported by `openmmforcefields`.
- Add optional HMR/4 fs mode behind config (`HBonds` constraint +
  `hydrogenMass=1.5*amu`):
  - enable only for protocol level >= 1
  - require a smoke/stability comparison against 2 fs before defaulting it on
- Increase final-tier MD replicas from 1 to at least 3 for candidates that
  reach the strongest evidence tier.
- Persist replica mean/std for ligand RMSD, pocket RMSD, contact occupancy, and
  instability score.

Verification:

- MD-lite score is calibrated against known accepted/rejected examples.
- Skipped parameterization is neutral and visible, not counted as a pass.
- Final report distinguishes "MD passed" from "MD not run".

Why / sources:

- **HMR (4 fs)** is standard in 2024+ enzyme MD pipelines — same accuracy, 2×
  wall-clock.
- **OpenFF Sage 2.x + Espaloma 2.x** outperform GAFF on benchmark ligand sets;
  `openmmforcefields` makes the swap a one-line change.
- OpenMM 8 / ML potentials: https://arxiv.org/abs/2310.03121
- OpenFF Sage: https://openforcefield.org/force-fields/force-fields/
- openmmforcefields: https://github.com/openmm/openmmforcefields
- openmm-ml: https://github.com/openmm/openmm-ml
- OpenMM protein-ligand workshop: https://github.com/openmm/openmm_workshops/blob/main/section_1/protein_ligand_complex.ipynb

## P1 Plan: Within the Quarter

Target: 4-12 weeks. These improve model capacity and throughput but need
benchmark gates before changing scientific defaults.

### P1.1 Foldseek + MSA Policy

Implementation:

- Enable Foldseek on real backends when database and executable are present
  (flip `homologs.use_foldseek` default to `true` on `real`).
- Merge sequence and structure homologs before clustering.
- Add ColabFold-style MSA truncation policy for Boltz inputs (≤ 256
  representative + 1024 extra) — cheaper and often higher Spearman than deeper
  MSAs.
- Report Neff, coverage, family/subfamily balance, and catalytic motif
  alignment.

Verification:

- Benchmark on `runs/demo_fdh_nadp` and any known family examples.
- Confirm Boltz confidence and key contact recovery do not drift downward.

Why / sources:

- **ColabFold-style shallow MSAs perform better than deep ones** for many
  mutational-landscape tasks (Δ Spearman ≈ +0.03 on DMS) and dominate the
  cost/quality frontier for AF2/AF3-class predictors.
- **Foldseek + 3Di alphabet** retrieves remote structural homologs orders of
  magnitude faster than HMMER and adds signal where sequence search saturates
  (already wired but off by default).
- ColabFold / MSA-depth: https://hal.science/hal-03907222v4/file/evad201.pdf
- Foldseek releases: https://github.com/steineggerlab/foldseek/releases

### P1.2 Interaction Graph Upgrade Before Architecture Replacement

Implementation:

- Promote PLIP interaction type to a one-hot edge feature, routed through EGNN
  as `edge_dim` (not just distance) in
  `features/interaction_descriptor.py`:
  - hydrogen bond
  - salt bridge
  - hydrophobic
  - pi-stacking
  - metal/cofactor coordination
- Add a PIGNet2-style physics-bias loss term: per-edge predicted contribution
  should respect monotonic distance/Coulomb sign (≈ free regularization), plus
  decoy augmentation (synthesize per-family decoy poses by perturbing torsions /
  rigid-body offsets as negatives during EGNN training).
- Train a pose-validity head using PoseBusters/PLIF-derived labels (clash /
  chirality / valence), fed as a pose-quality penalty into `s08`.
- Only after this baseline, evaluate EquiformerV2/MACE-style architecture
  replacement (modest but reliable lift once PLIP edge labels exist).

Verification:

- Holdout by protein family and ligand cluster.
- Report AUROC/AUPRC, calibration error, PLIF recovery AUPRC, and PoseBusters
  validity prediction.

Why / sources:

- **PIGNet2** (Digital Discovery 2024): physics-based inductive bias + augmented
  decoys lifts generalization vs vanilla EGNN/GAT, with a structured per-edge
  physics term (vdW, H-bond, hydrophobic) inside the message.
- **Edge-enhanced interaction graphs** (PMC 11977954, 2025): consistent gains by
  treating each contact type as a distinct edge class — exactly the PLIP output
  we already collect via `adapters/plip.py`.
- PIGNet2: https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00149k
- Edge-enhanced interaction GNN: https://pmc.ncbi.nlm.nih.gov/articles/PMC11977954/
- AI-driven PLI review 2025: https://www.sciencedirect.com/science/article/pii/S0959440X25000387
- PLINDER: https://openreview.net/forum?id=7UvbaTrNbP

### P1.3 Fast ML Stability Screening

Implementation:

- Add a `validation.stability.method: ml` adapter interface.
- Evaluate available backends:
  - ThermoMPNN-D (Siamese WT+mut) — best-in-class for *stabilizing* mutations
  - Stability Oracle
  - ProstaNet
  - STAB-DDG / ESMif-ddG if reproducible weights/code are available
- Use ML stability as primary screen for large candidate sets (cuts s09
  wall-clock 10–100×).
- Use FoldX/Rosetta on:
  - disagreements
  - final candidates
  - out-of-distribution structures

Verification:

- Compare against FoldX/Rosetta and any internal experimental stability labels.
- Report direction accuracy, Spearman, and false-accept rate for destabilizing
  mutations.

Why / sources:

- **STAB-DDG** matches FoldX accuracy with > 1000× speedup; the PNAS 2024
  transfer-learning approach also matches FoldX with deep learning.
- **ThermoMPNN-D** is best-in-class for the stabilizing side that matters for
  engineering. ProstaNet reports acc 0.75 vs ThermoMPNN 0.63 vs FoldX 0.71 on
  single-point.
- **ESMif-ddG**: inverse-folding ΔΔG with no extra training — near-free if ESMif
  is already run anywhere.
- STAB-DDG / PNAS 2024: https://www.pnas.org/doi/10.1073/pnas.2314853121
- ThermoMPNN comparison: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11997553/
- Stability Oracle: https://www.nature.com/articles/s41467-024-49780-2
- ProstaNet: https://spj.science.org/doi/10.34133/research.0674
- ESMif-ddG: https://www.biorxiv.org/content/10.1101/2024.06.15.599145.full.pdf

### P1.4 ESM Priors and Active Learning

Implementation:

- Add ESM-2/ESM-3 zero-shot fitness priors as features in
  `features/evolutionary.py`.
- Flow prior into:
  - residue mutability
  - `s07` candidate gate (drop candidates with ΔLL below threshold)
  - `s08` reranker
- Add active-learning acquisition (wire an ALDE-style function into a `--round N`
  mode that emits the next 96-variant library aware of model uncertainty):
  - UCB
  - Thompson sampling
  - diversity-constrained top-k
- Add EGNN scorer features (already in `ml/gnn_scorer.py`) as default reranker
  inputs, and report per-candidate calibrated uncertainty (conformal wrapper on
  XGBoost) into the final report.

Verification:

- On labeled rounds, track recall@K and hit rate of exploit vs explore pools.
- Do not let ESM prior override catalytic geometry or ligand-contact evidence.

Why / sources:

- **ESM-2 / ESM-3 zero-shot fitness** (ρ ≈ 0.50–0.60 across ProteinGym) is a
  near-free prior on which residues *can* be mutated; ESM-3 adds
  sequence+structure+annotation joint inputs.
- **ALDE** uncertainty-aware acquisition (see P0.4) and **EVOLVEpro** show pLM
  features are good base regressors with tiny labels (< 100 variants).
- **Conformal calibration** on affinity / family-consistency heads
  (PMC 12117556 / EquiScore line).
- ESM3: https://www.science.org/doi/10.1126/science.ads0018
- ESM zero-shot fitness: https://www.emergentmind.com/topics/esm-series
- ESM github: https://github.com/facebookresearch/esm
- ALDE: https://www.nature.com/articles/s41467-025-55987-8
- EVOLVEpro: https://www.science.org/doi/abs/10.1126/science.aea1820

### P1.5 MD Ensemble, ML/MM, and Flexible-State Prior

Implementation:

- Use Boltz ensemble samples (from P0.1) as alternative MD starting frames for
  top candidates instead of a single Boltz pose — improves coverage of
  productive/non-productive ensembles.
- Add a `md.engine: openmm-mlmm` path: MACE-OFF / AIMNet2 on the active site
  (residues + ligand within `design_radius_angstrom`) with MM elsewhere, via the
  `openmm-ml` plug-in. Replaces `md_lite_score` with a barrier-aware metric for
  catalytically-relevant variants. (ANI-2x is the safe baseline.)
- Add optional BioEmu-style conformational ensemble prior for flexible-loop
  risk, if model/tool availability fits local deployment.
- Add an explicit-solvent protocol for final top candidates when charged
  cofactors or active-site waters matter.

Verification:

- Report per-candidate replica variance and contact-occupancy confidence
  intervals.
- Candidates with high MD variance are marked `Uncertain`, not over-ranked.

Why / sources:

- **MLIP-in-MM** — MACE-OFF / AIMNet2 ML potentials for the QM region give
  DFT-accurate energetics at ~MM speed.
- **BioEmu** emulates conformational ensembles for flexible-state risk.
- Transition-state ML potentials (MACE-OMol25, AIMNet2): https://arxiv.org/pdf/2604.00405
- BioEmu: https://www.nature.com/articles/s41592-025-02874-1

### P1.6 Final-Ranking Pareto and Uncertainty Plumbing

Implementation:

- Add a Pareto-front selection mode (objective set: predicted activity proxy,
  stability, family-consistency, MD-lite) alongside the current weighted sum.
  Library size 96 then samples Pareto rank ≤ 2 with a diversity guard.
- Plumb the S08 calibrated uncertainty into ranking: CVaR / lower-confidence for
  risk-averse picks, UCB for exploration picks; emit two sub-libraries
  (exploit + explore).
- Apply negative-design constraints (`ranking/negative_design.py`) as hard
  filters *before* Pareto, not weight penalties — avoids "great main, terrible
  off-target" dominating the front.

Verification:

- Report exploit vs explore diversity and per-objective Pareto rank.

Why / sources:

- **Co-optimization of fitness and diversity** (Nat Commun 2024): Pareto beats
  weighted-sum on real DE deliverables, and the diversity term directly addresses
  combinatorial library design (our 96-variant output).
- **MODIFY** biases the starting library toward likely-functional variants while
  preserving diversity. Calibrated multi-objective ranking operates on the
  *posterior* score, not a point estimate.
- Co-optimization library design: https://www.nature.com/articles/s41467-024-50698-y
- ALDE: https://www.nature.com/articles/s41467-025-55987-8

## P2 Research Bets

These may be valuable, but they should not block P0/P1.

| Bet | Why it matters | Gate before adoption |
|---|---|---|
| Boltz-ABFE / OpenFE residue-FEP for top 10 | Adds kcal/mol-grade binding/free-energy tier (FEP-SPell-ABFE makes it operational) | Must outperform current final ranking on retrospective top candidates; cost must fit server budget |
| MACE / ML/MM active-site simulation | Better local chemistry for unusual cofactors, metals, charged ligands | First pilot on <= 10 candidates; compare to classical MD and known chemistry |
| EMLE or related electrostatic ML embedding | Potentially useful for enzyme barrier/electrostatics; ChemRxiv 2025 matches high-level QM/MM | Requires QM/MM benchmark and clear licensing/deployment path |
| Chai-1 / AF3 / RFAA second opinion | Cross-model disagreement can detect bad Boltz pockets | Add as optional cross-validator; do not require for standard runs |
| NeuralPLexer / FlowDock / DiffDock-L enzyme-cofactor fine-tune | Useful for induced-fit pockets; LoRA fine-tune (≤ 10k CrossDocked pairs) lifted allosteric kinases | Must pass PoseBusters/PLIF and local redocking benchmark |
| RFdiffusionAA / RFdiffusion3 / PocketGen | Useful for backbone/pocket redesign beyond point mutations | Only adopt if project scope expands from mutation design to backbone redesign |
| ZymCTRL / EnzymeFlow | Diversity generator for enzyme-class sequences (EC-conditioned) | Must project safely onto WT and preserve catalytic residues/cofactor geometry |

Sources: Boltz-ABFE residue-FEP — https://www.tandfonline.com/doi/full/10.1080/17568919.2025.2463870 ·
https://researchgate.net/publication/389519777_FEP-SPell-ABFE_An_Open-Source_Automated_Alchemical_Absolute_Binding_Free-Energy_Calculation_Workflow_for_Drug_Discovery ;
EMLE — https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2025-nw9lt ;
DiffDock-L fine-tune — https://pmc.ncbi.nlm.nih.gov/articles/PMC13014456/ ;
ZymCTRL — https://www.biopharmatrend.com/post/832-basecamp-research-introduces-zymctrl-an-open-source-ai-tool-for-enzyme-design/ ;
EnzymeFlow — https://arxiv.org/pdf/2410.00327 ;
PocketGen / VN-EGNN / GrASP — see S06 review above.

## Stage-by-Stage Roadmap

| Stage | Current direction | P0 | P1 | P2 |
|---|---|---|---|---|
| S02/S03 homolog/MSA | sequence + optional Foldseek | keep current defaults except server profile | Foldseek real default, MSA truncation, ESM priors | joint homolog DMS modelling |
| S04 complex | Boltz-2 | verify steering, increase real samples, persist uncertainty | Chai/AF3/RFAA cross-validator | Boltz-ABFE tier |
| S05 docking | Vina/GNINA/DiffDock | full-atom receptor (done), GNINA rescoring, pose-quality gate | diffusion sampler + physics rescoring benchmark | flexible docking fine-tune |
| S06/S06b graph | contact frequency + EGNN/XGBoost | consume Boltz ensemble spread | PLIP edge types, PIGNet2 loss, validity head | EquiformerV2/MACE replacement |
| S07 mutation | rules/MSA + optional LigandMPNN | LigandMPNN default on real, fixed-residue gate | ESM prior, multi-state design | RFdiffusion/PocketGen/ZymCTRL |
| S08/S08b rerank | heuristic/XGBoost + optional mutant Boltz | monotonic constraints, `boltz_delta_source` (done), uncertainty | active learning, EGNN features default | Bayesian multi-objective optimizer |
| S09 non-MD | FoldX/Rosetta/redocking | all configured redockers (done), explicit proxy status, PoseBusters gate | ML stability primary, real/repacked mutant validation (done) | high-accuracy physics validation |
| S10 MD | OpenMM MD-lite | Sage/HMR pilot, replicas for final tier | explicit-solvent tier, ensemble starts, ML/MM active site | EMLE / residue-FEP |
| S11 final | scalar rank/report | evidence class (done), exploit/explore split | calibrated uncertainty report, Pareto front | automated round planning |

## Evaluation Gates

Every phase should update benchmark output, not just code.

Required gates:

- `runs/smoke` regression remains green.
- `runs/demo_fdh_nadp` regression remains green or produces documented
  expected differences.
- Pose quality:
  - PoseBusters pass rate for accepted poses
  - PLIF/contact recovery against WT/reference contacts
  - ligand atom-order lock success
- Structure quality:
  - Boltz ensemble variance
  - pocket/catalytic residue confidence
  - source of mutant structure: proxy vs real vs repacked
- Ranking quality:
  - recall@K when labels exist
  - calibration error when labels exist
  - exploit/explore diversity
- MD quality:
  - MD actually ran count
  - skipped parameterization count and reasons
  - replica mean/std
  - ligand RMSD and contact occupancy confidence interval

## Explicit Non-Goals

- Do not train on Boltz affinity as if it were experimental activity.
- Do not replace Vina/GNINA anchors with an AI docking model until pose
  physicality and PLIF recovery gates are in place.
- Do not replace EGNN with a larger architecture before testing PLIP edge
  features and physics-biased losses.
- Do not run ML/MM for all candidates; use it only for final or research-tier
  candidates.
- Do not let proxy mutant geometry produce the strongest final evidence class.

## Proposed Work Packages

### WP1: Boltz and Docking Ensemble Upgrade

Files:

- `src/evoliez/adapters/boltz.py`
- `src/evoliez/stages/s04_complex.py`
- `src/evoliez/stages/s06b_interaction_model.py`
- `configs/server_fdh_nadp.yaml`
- optional new `configs/server_high_accuracy.yaml`

Done when:

- Boltz pocket steering is tested
- real WT sample count is increased in server config
- ensemble disagreement reaches the reranker/report
- GNINA 1.3 CNN features are extracted and persisted

### WP2: Generator and Reranker Upgrade

Files:

- `src/evoliez/stages/s07_mutation_gen.py`
- `src/evoliez/stages/s08_reranker.py`
- `src/evoliez/stages/s08b_mutant_boltz.py`
- `src/evoliez/stages/s11_final_ranking.py`

Done when:

- LigandMPNN is default in real profile
- XGBoost uses monotonic constraints when labels exist
- output library includes exploit/explore split when uncertainty exists

### WP3: Stability and MD Throughput

Files:

- `src/evoliez/stages/s09_nonmd_validation.py`
- `src/evoliez/adapters/openmm_engine.py`
- `src/evoliez/md/analysis.py`

Done when:

- ML stability adapter interface exists
- FoldX/Rosetta can be used as disagreement checks
- MD replicas and HMR/Sage settings are reported in provenance

## Recommended Next PR Sequence

(WP0 validation hardening — full-atom receptor guard, pose-quality persistence,
real-mutant redock, evidence classes — is already shipped; see Done / Landed.)

1. Boltz steering contract test and real server sample-count bump.
2. GNINA 1.3 rescoring feature extraction.
3. LigandMPNN real-profile default and fixed-residue preflight.
4. XGBoost monotonic constraints + exploit/explore split in final ranking.
5. OpenFF Sage default + MD replica provenance and HMR/4 fs pilot flag.
6. Foldseek default + ColabFold-style MSA truncation.
7. ESM-2/3 zero-shot prior wired into S02 features + S07 gate + S08 reranker.

This order increases model ambition now that the silent correctness risks are
closed.

## Open Evaluation Questions

- Which public DMS benchmark(s) mirror our deliverable best? ProteinGym
  catalytic-residue subset + EnzymeML / BRENDA assays are candidates — used to
  gate every Phase-1/2 change behind regression vs `runs/smoke` and
  `runs/demo_fdh_nadp`.
- For NADP cofactor switching specifically, is Boltz-2's training set saturated?
  Bench cofactor-class confidence drift before committing to Phase-2 fine-tunes.
- What is our error model for the MD-lite score? A miscalibrated scalar is worse
  than no scalar in the ranker.

---

*Cross-references: README pipeline diagram, [ARCHITECTURE.md](ARCHITECTURE.md),
[MECHANISM_AND_BENCHMARK.md](MECHANISM_AND_BENCHMARK.md),
[BENCHMARKS.md](BENCHMARKS.md), [SERVER_GRADE.md](SERVER_GRADE.md).*
