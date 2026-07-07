# CAR V5 — next-action decision (reviewer Day 6)

**Focused verdict = CONDITIONAL_PASS (no discrimination; WT co-substrate diffused).**
→ Reviewer track = **sampling escalation** (NOT EvidenceCard A/B, which needs a PASS; NOT
integration-fix, since the machinery is valid).

## Root-cause hypothesis (ranked)

1. **Implicit solvent (primary).** s10 runs GBSA implicit solvent. A 3-HP⁻ / ATP⁴⁻ / Mg²⁺ cluster is
   dominated by explicit water + counter-ion electrostatics; GBSA screens them crudely, so the
   co-substrate diffuses and the Mg bridge is not held. **The distance floor (~4.8 Å) and the WT
   diffusion both point here.**
2. **No productive anchor.** The V4 active-state reference ensemble (anchor the Michaelis pose) is not
   wired into this MD path, so the trajectory samples away from the productive geometry.
3. **Trajectory length is NOT the bottleneck** — median distance barely changed 0.05 ns → 2 ns, so
   "just run longer" is unlikely to help on its own.

## Recommended escalation (in order)

| step | change | cost | expected effect |
|------|--------|------|-----------------|
| E1 | **Explicit solvent** (TIP3P box + Na⁺/Cl⁻ neutralize, PME) for the s10 NAC path, top candidates only | high GPU | stabilize the anion+Mg cluster; hold the co-substrate → valid WT baseline + real ΔNAC |
| E2 | **Active-state reference ensemble** seed (V4 `guided/`) to anchor the productive Michaelis pose | med | keep sampling near the reactive geometry instead of diffusing |
| E3 | restraint-free **Mg retention diagnostic** (does Mg stay bridging over the traj?) | low | confirm/refute hypothesis #1 before big compute |
| E4 | (only if E1–E3 still flat) **QM/MM-lite** single-point on the best explicit-solvent frames | high | electronic-structure check of the attack geometry |

**Do E3 first (cheap diagnostic), then E1 on a handful of candidates (WT + lead + 2–3 controls)** before
committing the full 18 to explicit solvent. This keeps the shared box courteous and follows the
diagnostic-first principle.

## What is NOT changing

- Angle stays **read-only** — no O→P angle restraint (that manufactures NAC).
- Mg stays a **structure-level Amber ion** (never OpenFF), co-folded then re-placed.
- Claim discipline holds: nothing is a "lead" on geometry until an escalated tier discriminates.

## Deliverable routing

- Full `md_candidates.json` + trajectories stay on the server (`runs/srcar_3hp_v5`) → sync to
  **Google Drive `EvoLiEZ_runs/car`**, NOT GitHub.
- Small provenance summaries (this file, acceptance report, geometry table, fingerprint, verdict json)
  live in `docs/car_v5/`.

## ADDENDUM (2026-07-06) — corrected sequence after the E3 Mg bug

E3 found the focused candidate MDs lacked Mg (batch fan-out dropped `metal_requested`; fixed in PR #8).
So E1 is NOT the immediate next step — the corrected Mg path must first be verified on the real
multi-GPU path before spending explicit-solvent compute:

```
A. corrected-Mg smoke  : WT + lead + P438N, implicit, 0.05 ns, >=2 GPUs (force run_md_batches)
B. E3 re-diagnostic    : confirm WT AND candidates both have MG in topology
C. (only if A/B clean) : E1 explicit-solvent subset
```

A is a **verification smoke** (does PR #8 put Mg into candidates on the batch path?), not an attempt to
get a scientific answer from implicit solvent. Launch: `bash scripts/run_car_v5.sh corrected-smoke`.

## Open decision for the user

E1 (explicit-solvent s10 path) is a **real feature + a multi-hour compute escalation**. Recommend
sanctioning **E3 → E1(subset)** next; hold the full-18 explicit-solvent run until the subset shows the
co-substrate is retained.
