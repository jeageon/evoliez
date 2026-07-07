# Corrected-Mg smoke — PR #8 server verification (reviewer Day 1-2)

Run: `runs/srcar_3hp_v5_cmg`, 3-candidate manifest (WT + G430R;S433F;G407K + P438N), implicit, 0.05 ns,
1 replica, RBFE/GBSA off, **2 GPUs so the s10 batch fan-out (`run_md_batches`) is exercised**. Code:
`feat/server-hardening` + PR #8 (metal fan-out fix) + PR #9 (s08b empty-bucket fix).

## Purpose
NOT candidate evaluation — a real-server check that PR #8 (thread `metal_requested` through the
multi-GPU `run_md_batches` path) actually puts Mg into **candidate** MDs, which the batch bug had
silently dropped (WT-only Mg confounded the focused run).

## Result: PASS

| criterion | required | observed |
|-----------|----------|----------|
| WT + candidate PDBs contain MG | all | WT=2, mut_00000=2, mut_00001=2 (was 0 for candidates) |
| metal_setup.status | valid_metal_setup | valid_metal_setup |
| openff_parameterized | false | false (Amber ion) |
| O→Pα angle finite | yes | n_angle_computed=2, n_angle_nan=0 |
| E3 "no MG in topology" | gone | gone — E3 now computes real Mg metrics |

**Every candidate now gets Mg on the real multi-GPU batch path.** The focused-run confound (candidates
had no bridging cation) is resolved; the fix is server-verified.

## E3 on the corrected trajectories (Mg now tracked)
| system | mgNucO | mgLeaO | bridgeOcc | mode |
|--------|--------|--------|-----------|------|
| WT | 0.00 | 0.00 | 0.00 | substrate_left_first |
| G430R;S433F;G407K | **1.00** | 0.00 | 0.00 | substrate_left_first |
| P438N | **1.00** | 0.00 | 0.00 | substrate_left_first |

Mg now coordinates the 3-HP nucleophile O in the mutants (`mgNucO=1.00`), but the ATP phosphate side
still drifts (`mgLeaO=0.00`, `substrate_left_first`) in **implicit** solvent — the same diffusion E3
diagnosed, now on Mg-consistent data. This re-confirms the E1 (explicit-solvent) escalation is the
right next step, and it can now run without the Mg confound.

## Provenance verdict
`CONDITIONAL_PASS` — Mg machinery valid across WT + candidates; the co-substrate does not yet stay
productive in implicit solvent (→ E1). Not a candidate ranking; no lead/activity claim.

## Two bugs fixed en route
- **PR #8** metal_requested fan-out (candidates lost Mg on >1 GPU).
- **PR #9** s08b empty GPU buckets (2 mutants / 4 GPUs → `boltz predict` on a missing dir crashed the
  stage); relaunched on 2 GPUs. Also cut s06b to 3 representative homologs (~56 min → ~5 min) — pure
  overhead for a Mg-verification smoke.
