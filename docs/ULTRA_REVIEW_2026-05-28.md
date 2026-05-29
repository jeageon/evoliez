# EvoLiEZ Ultra-Review (2026-05-28)

Full line-by-line review of the entire codebase (23,612 LOC src / 122 files,
branch `feat/pymol-pocket-style` @ `7212e53`), partitioned into 9 clusters and
reviewed in parallel, with the highest-stakes findings re-verified by direct
reading + arithmetic. Fixes landed on branch `fix/ultra-review-p0p1`.

Verification legend: ✅ verified by direct read/arithmetic · ✅✅ cross-found by
two independent reviewers · ○ single-reviewer (plausible, not independently
re-verified).

---

## Headline: real MD validation was scientifically inert (now fixed)

Three independent bugs combined to make every "MD validated" result meaningless:

1. **nsteps 1000× too few** (openmm_engine.py:1163) ✅ — `(ns*1000)/(fs/1000)/1000`
   ran ~0.5 ps instead of the configured 0.5 ns, while still *reporting* the full
   ns. Per-candidate wall-time (~300–500 s) was dominated by setup, not dynamics.
2. **catalytic geometry never checked** (openmm_engine.py:1200) ✅ — real path
   hardcoded `key_distances={}`, so `analysis.CAT_DIST_MAX` was dead on the real
   path (the mock path populated it).
3. **whole backbone frozen** (openmm_engine.py:1101–1107) ✅ — every Cα restrained
   at 5 kcal/mol/Å²; the `pocket` set was a dead placeholder (`abs(ca[0])>=0`,
   always True, never used). Pocket RMSD forced ~0 → everything "ok".

Consequence: the celebrated "12/12 MD validated" runs (TEM1/XR/Bgl3) actually
validated *minimize + frozen-backbone + ~0.5 ps + no catalytic check*, reported
as "0.5 ns validated".

**Fix (Group A):** correct ns→steps formula + report actual simulated ns;
populate `key_distances` with per-frame catalytic-Cα↔ligand-centroid distances;
wire `md/restraints.build_restraint_plan` so only DISTANT backbone is restrained
and the pocket + active-site is free to move. `restraints.py` was previously
dead code (never imported).

---

## All findings by severity

### CRITICAL — silently wrong science

| ID | finding | location | verify | status |
|----|---------|----------|--------|--------|
| C1 | MD nsteps 1000× too few | openmm_engine.py:1163 | ✅ | FIXED (A) |
| C2 | catalytic distances never computed (real path) | openmm_engine.py:1200 | ✅ | FIXED (A) |
| C3 | whole-backbone freeze + dead pocket placeholder | openmm_engine.py:1101 | ✅ | FIXED (A) |
| C4 | family-interaction "holdout AUROC" circular (decoys from held group's own median) → AUROC ~1.0 is decoy-separation, not predictive power; heuristic saturates 0.98±0.008 ignoring 3 advertised features | s06b:281, pose_selection.py:112, interaction_model.py:123 | ✅✅ | FIXED (E) |
| C5 | config accepts negative/zero production_ns / temperature_K / timestep_fs / replicas | config.py:140 | ✅ | FIXED (A) |
| C6 | mock run stampable as "real" (mock_backend from build config, not run state) | builder.py | ○ | FIXED (B) |

### HIGH — latent bugs / overclaim / silent fallback

| ID | finding | location | verify | status |
|----|---------|----------|--------|--------|
| H1 | dead `ligand_escape` REJECT branch (never written to scores) | s11:42 | ✅ | FIXED (C) |
| H2 | "unstable" MD → Promising/accepted (not failed/skipped) | s11:39 | ✅ | FIXED (C) |
| H3 | s09 skipped-primary-docking scored as perfect redock (consistency=1.0) | s09:220 | ✅ | FIXED (C) |
| H3b | non-MD candidates read as phantom "MD ok" | s11:39 | ✅ | FIXED (C) |
| H4 | rosetta/foldx feed CA-only PDB to real binaries (no full-atom guard) | rosetta.py, foldx.py | ○ | FIXED (D) |
| H5 | ligandmpnn CA-only input | ligandmpnn.py:84 | ○ | FIXED (D) |
| H6 | vina/gnina/diffdock real-tool-no-output → silent mock (no skip flag) | vina/gnina/diffdock | ○ | FIXED (D) |
| H7 | msa_tools / remote_msa silent mock / partial-a3m fallback | msa_tools.py, remote_msa.py | ○ | FIXED (D) |
| H8 | score_waterfall reads `ddg_fold`, CSV column is `stability_ddg` → ddG bar silently dropped | score_waterfall.py:21 | ✅ | FIXED (B) |
| H9 | pose_ensemble wrong path + no n≥2 guard → single WT pose shown as "ensemble" | pose_ensemble.py:49 | ✅ | FIXED (B) |
| H10 | discovery `complexes/representatives` vs actual `structures/representatives` | discovery.py:268 | ✅ | FIXED (B) |
| H11 | diagnostics tool checks are PATH-only (`which`), not "working" | diagnostics.py:108 | ○ | FIXED (F) |
| H12 | benchmark `_spearman` no tie-correction (`[5,5,5]`→1.0); `_auroc` empty-class→0.0 (worst, not N/A) | benchmark.py:125,145 | ○ | FIXED (E) |

### MEDIUM — robustness / honesty (representative; fixed where in-scope)

- subprocess grandchild reaping (GPU VRAM leak on timeout) — subprocess_utils.py — FIXED (F)
- diffdock CSV injection (raw f-string SMILES) — diffdock.py — FIXED (D)
- sqlite no busy_timeout/WAL (database-locked under concurrency) — db/store.py — FIXED (F)
- `mutation` vs `mutations` column drops the column in report tables — builder.py — FIXED (B)
- HomologConfig inverted identity range silently → 0 homologs — config.py — FIXED (A, validator)
- interaction_model `scale` collapse when `mn<=mp` (hard step) — interaction_model.py — FIXED (E)

### Known-correct (re-flag avoidance)

cofactor chemistry (NAD/NADP/NADH/NADPH SMILES/charge/heavy all correct);
atom-index lock (graph isomorphism, honest `locked=False`); evidence_class
proxy→never-Strong gate; `validated_candidates is None vs []` distinction; atomic
state write; DB idempotency; label-policy enforcement (Boltz-derived ≠ supervised
label); no `shell=True` anywhere; self-contained path-traversal guard. Plumbing is
solid; the science layer was the weak point.

---

## What this branch fixed (Groups A–F)

- **A (MD + config):** nsteps formula + actual-ns reporting; per-frame catalytic
  distances; restraint plan wiring (pocket free); `Field(gt=0/ge=0)` bounds on all
  MDConfig numerics; HomologConfig identity-range validator.
- **B (figures honesty):** `stability_ddg` column (alias `ddg_fold`); `mutations`
  column; discovery `structures/representatives`; pose_ensemble tree-walk +
  n≥2-distinct-model guard; mock detected from run `_state.json`.
- **C (ranking honesty):** `ligand_escape` propagated from s09/s10 so the REJECT
  branch fires; "unstable" MD → Reject + gate-blocked; non-MD candidates →
  `md_status="not_run"` (cannot be phantom-Strong); skipped-primary-docking →
  neutral signals, not perfect.
- **D (adapters):** full-atom guard on rosetta/foldx/ligandmpnn; real-tool-no-output
  → skipped pose (not laundered mock); diffdock CSV via `csv.writer`; remote_msa
  timeout → None; msa_tools real-empty → loud + `synthetic_real_fallback` tag.
- **E (ML):** subfamily holdout re-implemented as leave-one-group-out with
  TRAIN-consensus decoys, renamed `consensus_generalization_auroc`
  (off-manifold junk: OLD ~0.92–1.00 → NEW ~0.00); heuristic scale fix + honest
  docstring; `_spearman` tie-correction + None; `_auroc` None on empty class.
- **F (infra):** diagnostics `--version` probes (WARN on broken binary);
  subprocess `start_new_session` + process-group SIGKILL on timeout; sqlite
  WAL + busy_timeout.

**Tests:** +5 files (`tests/test_ultra_review_{adapters,figures,infra,ml,ranking}.py`)
plus the existing suite. Full suite: **668 passed, 2 skipped, 0 failures**
(baseline 598).

---

## Not yet addressed (follow-up)

- All-atom `ProteinStructure` (Boltz output is full-atom; only the parser
  down-projects to Cα+centroid) → would enable atom-level (not residue-centroid)
  interaction features and activate the exported GNN path.
- `explicit` solvent option is silently treated as implicit (openmm_engine).
- FoldX/Rosetta output parsers are heuristic (mean-of-tokens) — parse the real
  format.
- Foldseek config knob still has no execution path (remove or implement).
- diagnostics residue-token-vs-sequence consistency runs only under `doctor`,
  not under `run`.

These are recorded for a later pass; none block the P0/P1 fixes above.
