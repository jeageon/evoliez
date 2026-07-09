# V6-2 — Amber GPU MD Executor (server-verified)

**Phase:** V6-2 · **Status:** ✅ PASS · **Modules:** [`amber_scheduler.py`](../../src/evoliez/adapters/amber_scheduler.py) (new), [`amber_engine.py`](../../src/evoliez/adapters/amber_engine.py) (extended)
**Driver:** [`scripts/run_amber_md_car.py`](../../scripts/run_amber_md_car.py) · **Provenance:** `reports/provenance/amber_gpu_jobs.jsonl` · **Tests:** [`tests/test_amber_scheduler.py`](../../tests/test_amber_scheduler.py)

## Goal

Run production MD through Amber `pmemd.cuda` and return results in the same
evidence schema EvoLiEZ uses — with GPU job fan-out (one independent pmemd per
GPU), restart safety, blocked CPU fallback, and cpptraj reaction-geometry
diagnostics feeding EvidenceCard/ClaimGuard.

## Design

`AmberGpuScheduler` fans a list of built Amber systems (from V6-1) across a GPU
pool. Each job runs its staged protocol pinned to a single GPU via a thread-safe
GPU-queue allocator (ROADMAP_V6 §2.2 — independent jobs over multi-GPU single
jobs). Per job:

- **Restart-safe staged protocol**: `min → heat → equil → prod`; a stage is
  skipped when its restart file already exists (resume after interruption).
- **Robust 2-stage minimization** (min1 backbone-restrained clash relaxation with
  a tiny initial step + no SHAKE, then min2 broader) — see "bugs fixed" below.
- **Reaction-geometry cpptraj analysis** from the V6-1 manifest reactive masks:
  O_nuc→Pα **access distance**, in-line O_nuc–Pα–O_leaving **angle** (reported
  SEPARATELY, ROADMAP_V6 §6), **Mg bridge** retention, ligand/backbone RMSD,
  energy drift. Every result carries the screening-level `claim_ceiling`.
- **Provenance**: one JSONL record per job (`amber_gpu_job/v1`).

## Success conditions — validation

| Condition | Result |
|---|---|
| CPU fallback blocked unless explicitly requested | ✅ empty GPU pool + `allow_cpu=False` → `skipped_no_gpu` ("refusing silent CPU fallback") |
| Restart / resume reliable after interruption | ✅ re-running a completed job → `stages_remaining=[]`, 0.5 s, no MD re-execution (analysis only) |
| GPU execution, no silent CPU fallback | ✅ each stage asserts pmemd printed a `CUDA Device Name` block |
| cpptraj recomputes ligand/Mg retention, O→P dist, angle, RMSD, energy | ✅ (see analysis schema) |
| EvidenceCard / ClaimGuard compatible | ✅ structured `analysis` + fixed screening-level `claim_ceiling` |
| WT + ≥2 candidates → explicit trajectories on pmemd.cuda | ✅ (3-job run below) |

## Bugs found + fixed live (stop-and-fix)

The first smoke run **failed at heat** — pmemd minimization diverged to
−8.6×10⁹ kcal/mol and the 1-4 nonbonded term overflowed. `cpptraj check` on the
fresh system found the cause:

1. **V6-1 transplant bug (ATP hydrogens)**: my V6-1 coordinate transplant moved
   ATP *heavy* atoms to the pose but left *hydrogens* at the curated reference
   positions → ATP H clashed with the pocket (ILE335/361, MET364 at 0.65–0.72 Å).
   **Fixed** in `amber_builder.transplant_pose_coords`: `AddHs(addCoords=True)`
   rebuilds H on the pose heavy-atom frame, then all atoms are transplanted.
2. **Minimizer robustness**: the mutant anchored structures carry severe
   side-chain clashes from the crude anchored swap (PHE433 ring collapsed,
   CG↔HZ = 0.18 Å) — OpenMM's L-BFGS absorbed them; pmemd CG overshot into a
   blow-up. **Fixed** with the robust 2-stage min (min1: backbone-restrained,
   `ntmin=1`, `dx0=0.001`, `ntc=1/ntf=1`).
3. **`energy_drift` metric bug**: the drift regex also caught pmemd's trailing
   RMS-fluctuation `Etot` (612), reading a stable run as ~100% drift. **Fixed**
   (truncate at the `A V E R A G E S` block) in both `amber_scheduler` and the
   shared `amber_engine._analyze`.

After the fixes, `mut_00000` (a clashy real mutant) ran end-to-end
(min+heat+equil+prod) in 66 s at 0.01 ns, with a stable ligand (RMSD 0.65 Å) and
corrected energy drift.

## 3-job run — WT + 2 candidates (explicit solvent, 0.1 ns, GPUs 0/1/2)

All three built (V6-1) and ran the full `min1→min→heat→equil→prod` cycle,
**concurrently, one job per GPU** (~150–160 s wall each, `reports/provenance/amber_gpu_jobs.jsonl`):

| job | status | GPU | O_nuc→Pα mean (min) | in-line angle | Mg–Pα | ligand retention / RMSD | energy drift |
|---|---|---|---|---|---|---|---|
| wt | ok | 0 | 6.13 Å (5.18) | 88.8° | 3.33 Å | 1.0 / 1.97 Å | 0.006 |
| mut_00000 | ok | 1 | 12.62 Å (10.99) | 119.3° | 3.35 Å | 1.0 / 1.52 Å | 0.005 |
| mut_00001 | ok | 2 | 7.40 Å (7.40) | 121.7° | 3.31 Å | 1.0 / 2.55 Å | 0.010 |

**Deep analysis (claim-safe):**
- All three are **stable** — ligand retention 1.0, energy drift 0.5–1 % (the *corrected*
  metric; energy is conserved), backbone intact.
- **Near-attack occupancy = 0.0 for all three** — unbiased explicit MD never reaches
  the productive ≤3.6 Å O_nuc→Pα window, and the in-line angle stays well below 150°.
  This is **not** a failure: it reproduces the E4a physics that the near-attack
  conformation is transient, not a stable minimum. It is the direct motivation for
  V6-3 — the right observable is the **free-energy cost to access** near-attack
  (PMF), not whether unbiased MD sits there.
- Mg sits ~3.3 Å from Pα (coordinating the phosphates) but 9–15 Å from the
  nucleophile — it cannot bridge because the two anions are 5–13 Å apart in the
  relaxed state. Consistent, not a build error.
- The three distinct fingerprints (`7f8e7306…`, `d0da9033…`, `f35c3dd3…`) confirm
  three genuinely different systems.
- **No ranking or activity claim is made** — occupancy differences at 0.1 ns
  unbiased MD are not evidence of catalytic difference; the ceiling stays
  screening-level.

## Analysis schema (EvidenceCard inputs)

```
access_distance:   {series_min_A, series_mean_A, near_attack_A, near_attack_occupancy}
inline_angle:      {series_mean_deg, angle_min_deg, productive_angle_occupancy}
mg_retention:      {mg_onuc_mean_A, mg_pa_mean_A, bridge_occupancy}
ligand_retention, ligand_rmsd_final_A, backbone_rmsd_final_A, energy_drift
claim_ceiling:     "screening-level reaction-geometry access evidence (NOT activity/kcat/activation-barrier)"
```
