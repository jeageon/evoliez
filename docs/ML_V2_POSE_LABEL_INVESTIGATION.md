# v2 ML redesign — pose-label feasibility investigation (2026-06-29)

Status: **investigation complete; ML training deferred (user decision).** This document records
why the planned "MD-free functional-state (pose-preservation) triage ML" cannot be built as
originally framed for this target, and what the evidence recommends instead.

## Motivation

The current v2 candidate-scoring ML ([`ml/interaction_model.py`](../src/evoliez/ml/interaction_model.py),
24-dim ligand–pocket contact fingerprint) was found to enrich *binding-validity*, not catalysis
(memory: AUC ml→MD-pass 0.83 vs ml→NAC 0.36). The proposed redesign: an MD-free triage that
predicts whether a mutant **preserves the WT functional reference state** — multi-head
`P(reference_like) / displacement_risk / …` from cheap pre-validation features, trained on an
expensive pose label, with a strict cheap/expensive split so the model never trains on the
quantity it predicts.

## What was built (kept, reusable)

- **Learnability harness** — [`ml/learnability.py`](../src/evoliez/ml/learnability.py) (+ `tests/test_learnability.py`).
  Declares the cheap/expensive contract as an allowlist/denylist and enforces it mechanically
  (`assert_no_leakage`). Computes per-feature univariate AUC, a pure-numpy LOO logistic AUC over
  the cheap block, and the baseline AUC of the current `ml_score`, against pose/NAC/MD labels.
- **Label generator** — [`scripts/gen_functional_labels.py`](../scripts/gen_functional_labels.py)
  (+ `.sh` wrapper). Anchored-build + cheap md_lite (NAC/RBFE/binding-ΔG off) + `pose_gate` over
  *all* s07 candidates of a finished run. Purge-safe (read-only resume to s04 with the run's own
  config; in-memory MD overrides; new output file; scratch under `<run>/labels_md/`),
  position-stratified, idempotent, GPU fan-out.

## Findings (run on the finished `fdh_v2` run, 532 candidates)

### 1. The binding-validity axis IS cheaply learnable
Learnability harness, label = s09 pass (319/213, n=532):
- cheap-feature LOO logistic AUC **0.964**; without ThermoMPNN stability features still **0.893**
  (independent signal from `n_mutations` 0.87, `distance_to_design_ligand` 0.81,
  `distance_to_formate` 0.75).
- current contact-fingerprint `ml_score` vs s09 pass: AUC **0.27** (anti-predictive → replace).

### 2. The pose-preservation label is NOT cheaply obtainable
Two label routes both collapse, in opposite directions, and disagree candidate-by-candidate:

| route | n | reference_like | displaced / alt | pocket RMSD median |
|---|---|---|---|---|
| anchored MD, 0.2 ns | 150 | 122 (81%)* | **0** | 0.17 Å |
| anchored MD, **2.0 ns** | 12 | **12 (100%)** | **0** | 0.18 Å |
| de novo Boltz pose | 66 | 2 (3%) | **64 (97%)** | 11.4 Å |

\* the other 27 were `md_failed`, later shown to be GPU-context artifacts, not instability.

- **MD time is not the lever**: 10× more MD (0.2→2.0 ns) changed nothing — the anchored-start
  ligand sits in a deep reference minimum and does not displace on ns timescales.
- **`md_failed` were artifacts**: 7 of the 0.2 ns failures re-ran clean → reference_like at 2 ns;
  `failure_reason` = "OpenMM selected the CPU platform (no usable GPU)" + `fail_loud_on_cpu`.
- **de novo Boltz is unreliable**: WT de novo pose vs WT reference = reference_like 0.23 Å (Boltz
  reproduces WT faithfully), yet on the 14 candidates present in both sets, anchored says 14/14
  reference_like while de novo says displaced 11 / alternative 2 / reference_like 1. Same mutants,
  opposite verdicts → de novo over-reacts to mutations (the per-mutant pose-search instability that
  anchored validation was created to escape).
- **Cost ceiling**: 2 ns ≈ 25–30 min/candidate on OpenCL → 532 candidates ≈ 74 h on 3 GPUs.
  Long-MD labelling does not scale even if it worked.

## Conclusion

A cheap, *graded* pose-displacement label cannot be generated for this target: anchored MD never
displaces (any tested length), de novo Boltz always displaces (unreliable), MD time is not a lever,
and true cofactor displacement is a rare event (µs / enhanced sampling), not triage-cheap. This is
not a dead end — it rigorously explains the original observation (ML enriches binding-validity, not
catalysis): the catalysis/pose axis is not cheaply labelable.

## Recommendation (deferred, user decision)

Pivot the v2 ML target from "pose preservation" to **binding-validity / functional-state
retention** — the axis cheap features already predict (s09 pass, AUC 0.89–0.96) and that is
measurable. Achievable with existing data (532 candidates × s09-pass label × cheap features, no new
MD): train a leakage-guarded cheap-feature multi-head model (P(retained) + uncertainty). Keep
`learnability.py`'s cheap/expensive guard as the contract. Document pose-displacement prediction as
"not cheaply observable for this target" and shelve it (revisit only with enhanced-sampling MD or
experimental labels).

## Artifacts

- Local: `runs/fdh_real/reports/provenance/` (pulled fdh_v2 JSONs for the learnability harness).
- Server `/mnt/data/jglee/runs/fdh_v2/reports/provenance/`:
  - `functional_labels.json` — 150 anchored 0.2 ns labels. **Reinterpret as `anchored_stability`,
    NOT pose-preservation** (and note the 27 `md_failed` are GPU artifacts).
  - `denovo_pose_gate.json` — 66 de novo Boltz poses gated vs WT reference.
  - `diag_relaxed_2ns.json` — partial (12) 2 ns diagnostic; run stopped after the answer was clear.
- Known bug filed: md_batch ProcessPoolExecutor spawn workers lack PR_SET_PDEATHSIG → orphan GPU
  leak on parent kill (e639718 covered only `run()` children).
