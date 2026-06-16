# EvoLiEZ Production Stage-by-Stage Playbook

Run real sequences one stage at a time and judge "is this production-grade?" at
each gate before advancing. Every acceptance check / red flag below is grounded
in the actual code (`file:line`). Config: `configs/prod_fdh_nadp.yaml`
(real backend, remote MSA, `/mnt/data2`); substitute `server_fdh_nadp.yaml` if
you use a local homolog DB instead.

## How to drive this

Four CLI flags (`src/evoliez/cli.py:25-36`):

- `--from <stage> --to <stage>` runs exactly one stage (inclusive).
- `--stage-backend s04_complex=real` (repeatable) flips ONE stage to the real
  GPU backend while the rest stay mock — cheaper than a global `--backend real`.
  A typo throws immediately (`cli.py:59`).
- `--resume` re-enters and skips already-complete stages — but see RESUME NOTE.
- `--dry-run` prints the real tool command lines without executing — confirm
  Boltz/FoldX/DiffDock invocations + DB paths before burning GPU.

> ### ⚠ RESUME NOTE — cross-process resume only reloads s01 + s06b
> The in-memory artifact bus is rebuilt across processes ONLY by stages that
> implement `load()` — today that is **s01 and s06b only** (`grep -l 'def load'
> src/evoliez/stages/`). Consequences for stage-by-stage gating:
> - **`--from <stage> --resume` fails** for any `<stage>` whose upstream isn't
>   s01/s06b: `stages[start:end]` skips the predecessors without calling their
>   `load()`, so `run()` dies on a missing `target_sequence`/`wt_complex`/...
>   (`pipeline.py:37-40`). The `--from` commands below are kept for reference
>   but **use `--to <stage> --resume` instead** (it reloads s01/s06b and
>   re-runs the rest).
> - On `--to <stage> --resume`, every stage except s01/s06b **RE-RUNS** (its
>   `load()` returns `False`). For CPU stages (s02/s03) that's free; for **s04
>   Boltz / s08b / s10 MD it re-pays full GPU cost** — `adapters/boltz.py:260`
>   re-invokes `boltz predict` (seed-reproducible, not cached).
> - **Workflow:** validate s01–s03 incrementally; then run the GPU tail **once**
>   (`--to s11`) and gate s04…s11 from their persisted `reports/`/DB output
>   (re-inspect with no recompute). `scripts/prod_validate.sh` drives exactly
>   this. Proper fix (follow-up): implement `load()` per stage, or add a
>   Boltz/MD output-reuse guard so resume is cheap.

**Gating loop:** run a stage → read its `reports/`/`_state.json`/DB → run the
2-4 acceptance checks → only then advance. Before the FIRST real run gate
everything on `evoliez doctor -c configs/prod_fdh_nadp.yaml` (must show 0 BLOCK
— it cross-checks catalytic/fixed tokens against your real sequence,
`diagnostics.py:188-223`). The one-shot driver `scripts/prod_validate.sh <stage>`
does doctor + run + auto-acceptance in a single command.

### Resolved — "real means real" (audit P0 #3 + #6)
Under `backend=real`, a missing/failed real tool no longer silently degrades to
a mock artifact: **Boltz / Vina / GNINA / DiffDock raise `RealToolError`** when
they produce no output, and **s01 raises** if RDKit cannot parse the ligand
(synthetic). Set `allow_mock_fallback: true` (→ `EVOLIEZ_ALLOW_MOCK_FALLBACK=1`,
e.g. partial-coverage smoke) to permit degradation with a loud warning. And
**`evoliez doctor` now BLOCKs** (not just MISSING) a tool/dep the config will
actually invoke — Boltz (s04 real), the configured docking engines + obabel
(s05/s09 real), FoldX (s09 real, `stability.method=foldx`), OpenMM (s10 real,
`md.enabled`), RDKit (real), mmseqs/mafft (real, non-remote MSA). So before a
real run, `doctor -c <cfg>` must show 0 BLOCK — and now that genuinely gates the
toolchain, not only the target/numbering. FoldX/Rosetta still return
`ddg_fold=None` on a missing structure (honest neutral → MD), not a hard error.

### Resolved — full-atom structures for real docking/stability (audit P0 #1)
Real Vina/GNINA/DiffDock/FoldX/Rosetta now use the FULL-ATOM Boltz structure
(`structure.pdb_path`, protein-only) via `base.full_atom_receptor_pdb()` instead
of the CA-only `write_min_pdb` trace. If no full-atom structure exists (upstream
mock / CA-only), they degrade HONESTLY: dockers return a mock pose ("NOT a real
dock" warning), FoldX/Rosetta return `ddg_fold=None` (→ `stability_unavailable`,
routed to MD) — never a fabricated score on a backbone-only receptor. So during
real s05/s09: **all-identical `docking_score` or blanket `stability_unavailable`
means s04/s08b emitted a CA-only/mock structure**, not a tool failure — verify
s04 ran real first (`method=boltz2`, pLDDT>0).

### Resolved — s09 redocks the real mutant + honours its own backend (P1 #8, #2-redock)
s09 redocking now uses the per-mutant Boltz structure from s08b (top-N; the rest
fall back to the WT proxy) — check `details.redock_structure_source ∈
{mutant_boltz, wt_proxy}`; the top-`mutant_boltz_top_n` should read
`mutant_boltz` under real. And `redock_with(stage_name=...)` uses the CALLING
stage's backend, so `--stage-backend s09_nonmd=real` makes s09 redocking real
without also flipping s05. Stability stays on the WT proxy by design (FoldX
builds the mutant from WT + mutation list).

### Resolved — reusing an output_dir can't contaminate evidence (P0 #4)
A changed run fingerprint (new target / config / backend / dry↔real / retrained
GNN) now purges the stale SQLite DB **and** artifact dirs, not just
`_state.json` — `checkpoints/` and `logs/` are kept. So you can safely point a
new target at an old `output_dir`: the stages' idempotent inserts no longer
preserve the previous target's rows. (Still simplest to use a fresh dir per
target; the purge logs `reused output_dir with a changed fingerprint`.)

### Two known limitations that look like bugs — flag, don't chase
- **#15 md_lite constant terms:** the real MD path hardcodes `hbond_occupancy`
  to 0.5 (`openmm_engine.py` real branch) for apo/no-ligand complexes, so
  `md_lite_score` can cluster near a constant even when MD genuinely ran.
  Confirm `nsteps>0` and `energy_drift` populated → documented limitation, not a
  parser failure.
- **#22 catalytic geometry under coords-unchanged proxy:**
  `catalytic_geometry_penalty` reuses the WT-proxy coordinates for the geometry
  calc (`s09` computes `mut_cat` from `mc.structure`, not the s08b mutant), so
  it stays ~0.0. Penalty-only, never a hard filter — distinct from the
  #2-redock fix above (which only switched the REDOCK structure). Out of scope.

### Remote-MSA caveat (this config)
`prod_fdh_nadp.yaml` uses `msa.remote_server: true` (no local DB). Boltz's
STRUCTURAL MSA (s04) is real via `--use_msa_server`, but s02/s03 homologs +
evolutionary/subfamily features are **SYNTHETIC** by design (you'll see a
`using SYNTHETIC homologs ...` warning). The s02/s03 "real homolog" checks below
apply ONLY if you switch to a local DB (`homologs.database`, drop
`msa.remote_server`).

---

## s01 — input (`s01_input_preprocess.py`)
`evoliez run -c configs/prod_fdh_nadp.yaml --to s01_input`

1. **Ligand is real chemistry:** ligand `source=='rdkit'`, not `'synthetic'`
   (RDKit fallback is silent, `ligand.py:36-40`).
2. **Atom order locked:** `relabel_to_canonical()` returned `locked=True`
   (`ligand.py:329-336`) — else every id-keyed feature (RMSD/IFP/family
   fingerprint) attaches to the wrong atoms.
3. **Residue tokens consistent:** no `out of range` / `asserts X at position Y`
   warnings (`s01_input_preprocess.py:33-54`); sequence ≥10 aa or `ValueError`.
4. **DB row:** `Sequence` `source='target'`, `identity_to_target=1.0`.

**Red flags:** `grep -i 'rdkit.*failed' pipeline.log` (mock ligand → all
downstream chemistry fabricated); `'non-standard residues replaced'` (in-place X
substitution); `atom-order NOT verified by chemistry graph`. **Cost:** seconds.

---

## s02 / s03 — homolog + MSA (`s02_homolog.py`, `s03_msa.py`)
`evoliez run -c configs/prod_fdh_nadp.yaml --to s03_msa`

> Remote MSA (this config): EXPECT `using SYNTHETIC homologs` — the checks below
> are the LOCAL-DB criteria. Switch to `homologs.database` if you need them.

1. **Enough real homologs:** `n_homologs >= 10` (`s02_homolog.py:62-63`).
   Headers not `mock_homolog_*` under real + local DB.
2. **Clustering live:** `Sequence.cluster_id` ≥3 distinct values, NOT all 0
   (`msa_tools.py:312-333`). All-zero collapses s06b to one family rep — fatal.
3. **Real alignment has gaps:** `awk '/^[^>]/ {print length}' alignment.fasta |
   sort -u` shows >1 distinct length (synthetic = uniform).
4. **Conservation non-degenerate:** `0.0 < mean_conservation < 1.0`,
   `msa_depth >= 10` (`s03_msa.py:64-69`).

**Red flags:** indel homologs scoring ~0.20 not >0.90 (frame-shift, fixed —
feed a known-indel seq to confirm); `gap_frequency` always 0.0; subfamily count
scaling with `max_sequences`. **Cost:** minutes-tens of min CPU (local DB).

---

## s04 — Boltz-2 complex (`adapters/boltz.py`) — the central stage
`evoliez run -c configs/prod_fdh_nadp.yaml --stage-backend s04_complex=real --from s04_complex --to s04_complex`

1. **Real method:** `cx.method=='boltz2'`, not `'boltz-mock'` (mock fallback
   logs `"Boltz produced no prediction ... mock fallback"`, `boltz.py:275`).
2. **pLDDT populated:** per-residue pLDDT non-zero on >90% of residues (guards
   the G1-2 fix); `confidence_score ∈ [0.15, 0.95]` and DIFFERS from
   `complex_plddt`.
3. **Distinct poses:** ligand coords differ across diffusion samples (G1-3 fix);
   ensemble `contact_frequency > 0` for ≥1 (residue, ligand-atom) pair — the
   priority-#1 feature.
4. **Affinity present** (if `predict_affinity=true`): `affinity_pred_value` set;
   all METRIC_KEYS present (`boltz.py:34-39`).

**Red flags:** `grep 'mock fallback'` / DB `method='boltz-mock'`; all per-residue
pLDDT == 0.0 (npz not parsed); `Boltz ligand atom ids NOT verified` →
`source|reindexed`. Keep `use_kernels=false` (stock pip Boltz lacks
`cuequivariance_torch`). **Cost:** ~min/complex on A6000; scales with
`diffusion_samples` (prod = 40).

---

## s05 docking → s06/s06b — interaction graph + family model
`evoliez run -c configs/prod_fdh_nadp.yaml --from s05_docking --to s06b_interaction_model --resume`

**s06b is where the family model goes live.** `family_interaction_score` must
NOT be constant — `max-min > 0.05` for N≥100 candidates
(`InteractionModel.score_complex`). Constant 0.5 ⇒ `interaction_model.enabled=
false` or fingerprint extraction failed (depends on s02/s03 clustering being
live — so under remote-MSA synthetic homologs this signal is weaker by design).
`interaction_model_kind` (xgboost/heuristic/null) is recorded here → surfaces in
s11 provenance. **Real docking checks** (s05): `docking_score` varies ±0.2-0.5
(not identical = mock); pose RMSD-to-reference computed (not None →
`redocking_consistency=1.0`, the M12 fix).

---

## s08 / s08b — reranker + per-mutant Boltz (`s08_reranker.py`, `s08b_mutant_boltz.py`)
`evoliez run -c configs/prod_fdh_nadp.yaml --from s08_reranker --to s08b_mutant_boltz --resume`

1. **family_interaction_score live:** `>5` distinct values for N≥100 (sqlite:
   `json_extract(details,'$.features.family_interaction_score')`).
2. **Honest delta provenance:** the top-N (`mutant_boltz_top_n`) have
   `boltz_delta_source=='real'` under real (`s08b:63-73`); `'proxy'`/`'dry-run'`
   otherwise. <95% real on the top-N = FAIL.
3. **No double-count:** vary `family_interaction_score` 0.2→0.8 →
   `(final_hi−final_lo)/0.6 == w.family_interaction (1.0)`, not ~2.5x
   (`score.py:38-39`).
4. **Spread:** `std(ml_score) > 0.05`; per-mutant deltas DIFFER across candidates
   (proxy-conservation fix — identical deltas = broken).

**Red flags:** all deltas identical; `gnn_status='trained'` but checkpoint mtime
< run start (stale); `grep 'too few labelled examples; heuristic'` (xgboost
silently heuristic, needs ≥8 labels). **Cost:** s08b ~2-5 min/mutant →
top_n=40 ×5 samples ≈ several hours, 20-40 GB VRAM.

---

## s09 — non-MD validation (`s09_nonmd_validation.py`)
`evoliez run -c configs/prod_fdh_nadp.yaml --from s09_nonmd --to s09_nonmd --resume`

1. **ddG not fabricated:** `pstdev(ddg_fold) > 0.001` — OR mean 0.0 ONLY if >50%
   carry `stability_unavailable=True` and were routed to MD (M14/M28 honest
   neutral). Filter active: <95% pass `ddg_fold <= max_ddg_allowed` (2.5).
2. **Redocking discriminates:** `redocking_consistency` not uniformly 1.0
   (`pstdev>0.05` or `median<0.95`); a few `ligand_escape=True` (RMSD>4.5 Å).
3. **Survival sane:** `n_after_nonmd / n_input ∈ [0.2, 0.95]`.
4. **Log honesty:** `'non-MD validation [real]: X/Y passed'` (mode = real).

**Red flags:** all `ddg_fold==0.0` with no `stability_unavailable` (parser failed
silently — but M14/M28 now surface it); `docking_score` all identical (mock);
`catalytic_geometry_penalty` all 0.0 = EXPECTED under the proxy (limitation #22).
**Cost:** GPU redocking dominates (min/candidate); Vina path is CPU.

---

## s10 — MD (`s10_md.py`, `md/analysis.py`, `adapters/openmm_engine.py`) — heaviest
`evoliez run -c configs/prod_fdh_nadp.yaml --from s10_md --to s10_md --resume`
(needs s04 + s08b real so MD runs the ACTUAL mutant, not the WT proxy.)

1. **MD actually ran (P0 gate):** `grep 'MD real-execution: ' run.log` → `N/M`
   with **N>0** (`s10_md.py:106-109`). N=0 = degraded, FAIL even if skip counts
   are high. (The smoke `md` step already enforces this.)
2. **Stability passes:** `ligand_rmsd_mean < 4.5`, `pocket_rmsd_mean < 2.75`,
   `energy_drift <= 0.5` (`analysis.py:33-36,96`).
3. **nsteps correct:** `actual_ns` derived from `nsteps` (cap `_LITE_MAX_STEPS=
   25000` for L1/L2, floor 50, `openmm_engine.py` `_production_nsteps`). A "1 ns"
   L3 request showing 0.001 ns ⇒ the 1000× bug returned.
4. **Catalytic geometry** (when `key_distances` present): `hbond_occupancy >=
   0.25`, `catalytic_distance_mean <= 7.0`.

**Red flags / expected-skips:** `skipped_parameterization` dominant under real →
ligand FF unavailable / unsupported cofactor (**EXPECTED for NADP/large
cofactors**; neutral pass — verify intentional, needs `pdbfixer` + ambertools,
both now in the env). `skipped_no_*_structure` → s04/s08b emitted CA-only/mock.
All `md_lite_score ≈ 0.5` while `nsteps>0` → limitation #15 (not a failure).
`ligand_rmsd_series` all-zero → trajectory not parsed. **Cost:** heaviest GPU
stage; min-hours/candidate by `protocol_level`.

---

## s11 — final ranking (`s11_final_ranking.py`, `ranking/score.py`)
`evoliez run -c configs/prod_fdh_nadp.yaml --from s11_final --to s11_final --resume`

1. **Decomposition complete:** all 13 contributions + 11 penalties present and
   non-constant (`score.py:26-67`); `final_score = sum(contrib) − sum(penalties)`
   — hand-check the top candidate.
2. **Non-degenerate:** `variance(ml_score) > 0.05` across top 50; top vs
   bottom-50 differ in ≥8 of 13 contributions (a constant term ⇒ silent upstream
   fallback).
3. **Deterministic:** sorted by `(-final_score, candidate_id)` (L43 tie-breaker);
   `rank` monotonic; re-run yields identical top-3.
4. **Provenance honest:** `provenance.json` has non-empty `ligand_atom_ids`,
   `gnn_status ∈ {disabled, heuristic_fallback, trained}` matching
   `gnn_checkpoint_sha256` null-ness (M17), `interaction_model_kind`, reproducible
   `run_fingerprint`. `final_report.md` Model-provenance section states GNN status.

**Red flags:** `md_lite_score` all 0.0 with no `n_md_skipped` (MD silently
mocked); `gnn_status='trained'` but `gnn_checkpoint_sha256=null`; duplicate wells
in `focused_library.csv` (M18 round-robin); Boltz columns tagged
`supervised_label` in `roles.json` (ML-data-policy violation). **Cost:** 5-15 min.

---

## GNN train (`evoliez train-gnn` / `torchrun --nproc_per_node=4`)
`evoliez train-gnn -c configs/prod_fdh_nadp.yaml` (after s11 builds the graph dataset)

1. Graph dataset built (`graph_dataset_samples > 0` in meta).
2. Training loss decreases; checkpoint saved with `rbf_n` + `graph_geom` + sha
   pinned (L39/L35/M17).
3. Scorer loads (`weights_only`) and `gnn_score ∈ [0,1]` varies per candidate;
   inference graph geometry + disorder match training (M19/L35 — else 2/21 node
   features go to 0 at inference).

**Red flags:** `gnn_status=heuristic_fallback` when GNN expected; loss flat;
disorder columns zero at inference. (torch is server-only.)

---

## Quick gate summary
| Gate | One-line pass test |
|---|---|
| s01 | ligand `source=rdkit` + `locked=True`, no residue-token warnings |
| s02/s03 | local DB: `n_homologs≥10`, `cluster_id` ≥3 distinct, alignment has gaps (remote MSA = synthetic by design) |
| s04 | `method=boltz2`, pLDDT>0 on >90%, poses distinct, `contact_frequency>0` |
| s06b | `family_interaction_score` max−min > 0.05 |
| s08/s08b | `boltz_delta_source='real'` ≥95% of top-N, no 2.5× double-count |
| s09 | `pstdev(ddg_fold)>0.001` OR honest `stability_unavailable→MD`; survival 0.2-0.95 |
| s10 | `MD real-execution: N/M` with N>0; RMSD/energy_drift in bounds |
| s11 | 24 score terms non-constant, ranking monotonic, provenance honest |
| gnn | dataset>0, loss↓, checkpoint pinned, `gnn_score` varies |

Derived by a multi-agent pass over `src/evoliez/` (every `file:line` checked
against source). Pair with `SERVER_RUNBOOK.md` (env/GPU/storage) and
`SERVER_SMOKE.md` (plumbing smoke before production).
