# V6 Amber-GPU Platform — Live Server Verification Report

**Date:** 2026-07-08 · **Host:** admin1-SYS-740GP-TNRT (evo), 4× RTX A6000, CUDA 12.4, driver 550.163.01
**Scope:** live re-verification of every V6 gate on the real server using the CAR (SrCAR→3-HP) project.
**Discipline:** 30-min-cadence deep analysis, stop-and-fix on every error, claim-safe throughout.

## Environment (reproducible)

```
source /mnt/data/jglee/envs/evoliez/amber.sh          # AMBERHOME + AmberTools25 bin
export EVOLIEZ_PMEMD_CUDA=/mnt/data/jglee/pmemd24/bin/pmemd.cuda_SPFP
export EVOLIEZ_AMBERTOOLS_BIN=/mnt/data/jglee/envs/evoliez/bin
PYTHONPATH=src  /mnt/data/jglee/envs/evoliez/bin/python  # 3.10
```

## Bugs found + fixed live (stop-and-fix)

1. **Stale server code — energy-drift parser.** A full sha1 parity sweep of
   `src/evoliez` found `amber_engine.py` on the server was missing the committed
   `A V E R A G E S`-truncation fix (server `d82e6235` vs committed `3b3e7803`).
   Proven on real `prod_0.out`: buggy parser read the trailing RMS-fluctuation Etot
   as the last frame → **100.31 %** drift on a run whose true drift is **0.60 %**.
   Re-synced. Affects the s10 `run_md_amber` path; the standalone CAR runner
   (`amber_scheduler`) already had the fix. *Lesson: sha1-sweep server code before
   trusting a run (macOS/Linux `sort` differ — join on path, not `diff`).*
2. **Candidate-input reproducibility gap.** Pipeline anchored candidates are
   full-protein (~1188 res, ATP+3-HP+**NADP**, no Mg, frame ~27 Å off) → cannot feed
   the `car_spec` A-domain builder. Fixed with reproducible in-place template
   mutagenesis (`scripts/prep_car_adomain_mutant.py`, commit `f69244d`). See
   `docs/car_v6/adomain_mutant_prep_and_md.md`.

## Gate-by-gate live evidence

| Gate | Live result | Status |
|------|-------------|--------|
| **V6-0** Amber capability | `tleap→pmemd.cuda→cpptraj` round-trip: implicit GB (42 atoms, RMSD 0.51 Å) **and** explicit PME (9924 atoms, RMSD 0.22 Å); GPU DEVICE INFO parsed → no CPU fallback | **PASS** |
| **V6-1** system builder | WT + 2 mutants built (72k-atom explicit systems, ATP+3-HP+Mg, reactive atoms mapped); ambiguous inputs fail loud (UNK blob, missing Mg both rejected) | **PASS** |
| **V6-2** GPU MD executor | WT + G430R + LEAD explicit-solvent pmemd.cuda trajectories (250 frames, 0.2 ns), cpptraj recomputes access dist / angle / Mg / RMSD / drift; one job per GPU | **PASS** |
| **V6-3** PMF | 13-window umbrella ran on GPU; non-converged → recorded `classification: diagnostic_only`, "prohibited from ranking candidates" | **PASS** (diagnostic) |
| **V6-4** CAR portfolio | reproducible A-domain mutant prep + live WT/G430R/LEAD deconvolution MD; EvidenceCards L0 | **PASS** |
| **V6-5** QM/MM-lite | server records (`qmmm_wt_test.jsonl`), claim ceiling "reaction-core PLAUSIBILITY … NOT activity/kcat/barrier" | **PASS** (smoke-grade) |
| **V6-6** non-CAR proof | TEM-1 β-lactamase real Amber build + GPU MD; Ser68 Oγ→β-lactam carbonyl geometry resolved (8.51 Å), drift 0.72 %; committed `net_charge=0` reproducible | **PASS** |
| **V6-7** hardening | full local test suite green; server code re-synced to committed; reproducibility demonstrated (below) | **PASS** |

## Reproducibility

- **WT re-run** (0.2 ns, GPU 0, fresh build) vs prior WT (0.1 ns, prior session,
  different GPU): O→P min **5.16 vs 5.18 Å** (Δ0.02), Mg–Pα **3.35 vs 3.33 Å**
  (Δ0.02), near-attack occ 0.0 both. Energy drift 0.22 %.

## Claim-safe deconvolution (WT / G430R / lead), 0.2 ns explicit pmemd.cuda

| variant | O→P min (Å) | in-line angle (°) | near-attack occ | drift |
|---------|------|------|------|------|
| WT | 5.16 | 90.6 | 0.0 | 0.22 % |
| G430R | 10.27 | 122.8 | 0.0 | 0.06 % |
| LEAD | 7.25 | **167.3** (occ 1.0) | 0.0 | 0.87 % |

The lead reaches a near-linear in-line attack angle but **not** closer distance
access; near-attack occupancy is 0 for all → distance-access and in-line-angle
observables disagree → the Amber PMF tier is the correct escalation. **No activity,
kcat, or catalytic-superiority claim** — screening-level mechanism-probe geometry
only.

## Replicate (independent seed) — reproducibility of the deconvolution

| variant | O→P min r1/r2 (Å) | in-line angle r1/r2 (°) | near-attack occ |
|---------|------|------|------|
| WT | 5.16 / 4.95 (Δ0.22) | 90.6 / 92.3 (Δ1.7) | 0.0 / 0.0 |
| G430R | 10.27 / 8.07 (Δ2.20) | 122.8 / 104.2 (Δ18.5) | 0.0 / 0.0 |
| LEAD | 7.25 / 7.47 (Δ0.22) | 167.3 / 157.4 (Δ9.9) | 0.0 / 0.0 |

**Qualitative signal reproduces** — the lead is more in-line than WT in *both*
independent seeds (167°/157° vs 90°/92°). **Magnitudes are noisy** (G430R swings
±18°), and near-attack occupancy is 0.0 in every run → the signal is correctly
**screening-grade** (ordering reproducible, magnitude not) and PMF-tier escalation
is the right next step. Nothing here is an activity claim.

## Gate 4 hole found + fixed (ClaimGuard enforcement)

Testing enforcement (not just labels) revealed ClaimGuard caught only **2 of the 6**
ROADMAP §13 forbidden phrases — the activity-improved and kcat-improved forms were
caught, but the catalytic-validation, validated-lead labeling, activation-barrier
lowering, and experimental-activity forms all **leaked**. Added the two missing
categories (`catalytic_validation`, `activation_barrier`) with negation-safe EN+KO
patterns, unlockable only by replicated wet-lab evidence; a regression test locks all
six. Now every §13 phrase is rejected pre-wet-lab. (`src/evoliez/ranking/claim_guard.py`)

## Known limitations (honest)

- Portfolio MD is screening-grade: 0.2 ns, template-mutagenesis side chains.
- CAR PMF is non-converged (diagnostic-only, ranking-prohibited) — the documented
  rugged-landscape case; converged access-cost needs longer / 2-D sampling.
- QM/MM-lite is smoke-grade (single-frame, `qm_atoms` metadata thin).
- These are boundaries, not overclaims — every artifact's claim ceiling reflects them.
