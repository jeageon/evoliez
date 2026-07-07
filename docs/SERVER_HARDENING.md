# Server hardening (feat/server-hardening)

Correctness & operability fixes so the pipeline produces *trustworthy* results
on the production box (4× RTX A6000, 48 vCPU, 256 GB, CUDA 12.4, `/mnt/data2`,
no sudo). Driven by the multi-agent architecture audit
(`REPORT_SCIENTIFIC_FLOW_AUDIT.md` + the 2026-06 review). Every fix was verified
on the dev box (`.venv-md`, 120 passing tests + a full mock run + resume +
dry-run-real) before hand-off, per the local-first constraint.

## What was broken on the server (and is now fixed)

### P0 — the science was wrong on real data

1. **MD signal was meaningless.** `openmm_engine` restrained *every* Cα at
   5 kcal/mol/Å² (the `pocket` set was a dead always-true placeholder), so the
   whole backbone was frozen and `pocket_rmsd`/`ligand_rmsd` were ~0 regardless
   of the mutation. And `nsteps` divided by 1000 twice, so a "1 ns" request ran
   ~1 ps while `simulation_time_ns` reported the configured value.
   → tiered distance-keyed restraints (distant strong / shell weak / **pocket
   free**); correct step math; truthful reported time.
   Tests: `test_md_restraint_force.py` (real-OpenMM proof a restrained particle
   is held while a free one diffuses).

2. **Family-geometry model collapsed on the server.** The real homolog parsers
   never set `cluster_id`, so every hit defaulted to cluster 0 →
   `_pick_representatives` returned one representative → the "one rep per
   subfamily" design silently became "top-N by identity"; `cluster_identity` /
   `use_foldseek` were dead. → dependency-free k-mer Jaccard clustering
   (`assign_clusters`) applied on every real search.
   Tests: `test_homolog_clustering.py`.

3. **MD-lite weight squared + double-penalty.** `md_lite_score` was multiplied
   by `weights.md_lite` in `analysis.py` *and* again in `score.py`; integration
   health was charged both inside `md_lite` and as `md_instability`.
   → `md_lite_score` is now the raw quality metric; weight & health applied once.

### P1 — reproducibility & honest tool integration

4. **Real Boltz was non-deterministic.** No `--seed`, so the diffusion ensemble
   (and every contact-frequency / consensus / Δ / score derived from it) changed
   every run and a crash-resume mixed RNG draws. → per-label deterministic seed
   from `config.seed`.

5. **`pipeline.log` was always empty** (the early `setup_logging()` swallowed the
   later file handler). → file handler now attaches; 48-line log confirmed.

6. **Provenance couldn't reconstruct a run.** Tool versions were placeholder
   strings, no git SHA. → records the git SHA (+`-dirty`), exact package versions
   (openmm/openff/rdkit/torch/…), and each CLI tool's resolved PATH.

7. **gnina/diffdock poses had no coordinates** → `rmsd_to_reference=None` →
   `s09` read a falsely *perfect* `redocking_consistency` (1.0) and never flagged
   an escaped ligand. → shared `parse_sdf_first_pose` + `lock_pose_to_reference`
   adopt the docked pose + real RMSD for all three dockers.
   Tests: `test_docking_pose_coords.py`.

8. **Silent mock fallback in a "real" run.** A real docking tool that produced no
   output silently returned a fabricated mock pose. → still degrades (spec 23)
   but now logs a loud `NOT a real dock` warning, distinct from the dry-run
   preview. `subprocess_utils.run` also honours `EVOLIEZ_DRY_RUN` so a forgotten
   `dry_run=` can't execute a real command.

### P2 — operational robustness

9. **Resume silently truncated ML datasets.** `s06b.load()` restored only
   `interaction_model`; `ensemble_contacts`/`pose_dataset`/`edge_dataset` came
   back empty on resume. → all four persisted to `s06b_artifacts.json` and
   restored, or `load()` returns False to force a clean re-run. Verified: a
   resumed run still writes a 167-row `edge_level.csv`.
   Tests: `test_resume_s06b_artifacts.py`.

10. **Wrong ΔΔG parsers.** Rosetta averaged *every* number in the `.ddg` file;
    FoldX used "first numeric column". → Rosetta now computes
    `mean(MUT) − mean(WT)` (warns + falls back on an unrecognised format); FoldX
    locates the `total energy` column by name. Tests in `test_tool_output_parsers.py`.

11. **SQLite + checkpoint durability.** WAL + `busy_timeout` + `foreign_keys` on
    connect (so the Snakefile stage-chain / overlapping iteration WAIT instead of
    `database is locked`); `_state.json` written atomically (`tmp` + `os.replace`).

12. **PLIP honesty.** It always ran the geometry surrogate while claiming real
    PLIP; it now says so on a real backend (real PLIP deferred until a server
    fixture exists — unverified XML parsing of a weak-label feature is worse than
    an honest surrogate).

## Deliberately NOT changed here (need your call / a server fixture)

These are real, but they are research-design choices or need the server to
verify — shipping unverified changes would violate the local-first rule.

- **Multi-GPU.** `GpuPool` exists but the s06b homolog Boltz ensemble runs
  serially on one card. A 4× speedup is available, but the box is **shared with
  other users** (per your setup), so grabbing all cards by default is
  antisocial. Recommend an *opt-in* `gpu.max_devices` that fans the s06b loop
  across free cards via a process pool — implement + verify on the server.
- **Score-scale & weights.** `ml_score` is unbounded while most terms are 0..1,
  and several Boltz-derived signals enter the additive sum more than once
  (`family_interaction` in `ml_score` *and* as its own term; `complex_confidence`
  is rank-neutral). These shift rankings, so they're your call — recommend
  per-term normalisation + a sensitivity sweep against a held-out DMS set.
- **Self-supervision circularity.** In the no-experimental-label mode the whole
  ranking is a function of one Boltz prediction, and the subfamily-holdout AUROC
  re-uses the same consensus rule that made the labels (measures self-consistency,
  not predictivity). Recommend a retrospective benchmark vs a conservation-only
  baseline before claiming added design value.
- **s02–s05 resume** still re-runs (correct, just not cached); **real PLIP**;
  **declared FK columns** in the ORM schema.

## Verify on the server

```bash
bash scripts/run_pipeline.sh configs/server_fdh_nadp.yaml     # full real run
python scripts/check_real_md.py <real_boltz_model.pdb>        # real OpenMM MD-lite
# then check: reports/provenance.json has a real git_sha + tool versions,
# logs/pipeline.log is populated, and a --resume reuses s06b without zeroing
# ml_datasets/edge_level.csv.
```
