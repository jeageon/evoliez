# CAR V5 — evidence package (SrCAR → 3-HP adenylation)

**Status: banked, claim-safe.** This is the consolidated CAR V5 triage output. EvoLiEZ is a
mechanism-configurable, claim-safe enzyme-variant **triage** framework — it quantifies structural
viability, reaction-geometry evidence, and uncertainty to prioritise experiments; it is **not** an
activity predictor. Read every result below through that lens.

Code: `feat/server-hardening` (PRs #8–16 merged). Server runs reproducible from the merged commit +
configs. Large run artifacts stay on the server / Google Drive, not GitHub.

## 1. What was built and server-verified (the machinery)
- **Adenylation mechanism, config-declared** (`mechanism.reaction: adenylation_phosphoryl_transfer`,
  `metal_state: Mg2+_bridged`, `cofactor_state: ATP`) — no core-code edit per enzyme.
- **Mg²⁺ co-fold** in Boltz (`ccd: MG`) + **structure-level Amber Mg ion** at MD (never OpenFF),
  threaded through the WT reference **and** the multi-GPU candidate fan-out.
- **Explicit-solvent MD path** (TIP3P + PME + Na⁺/Cl⁻ neutralize) — a real, server-verified capability
  (was an implicit-only stub).
- **Reaction-geometry (NAC) as evidence** — O_nuc→Pα distance + in-line angle, angle **read-only**
  (never restrained → never manufactured).
- **ClaimGuard** blocks overclaim across reports; **manifest-seeded** focused runs; **multi-lane**
  selection preserves low-ML / mechanism-positive candidates.

## 2. The investigation arc
| stage | result |
|-------|--------|
| E3 diagnostic | Found a real bug: candidate MDs ran **without Mg** on the multi-GPU batch path (WT-only Mg confounded the earlier focused run). **Fixed** (PR #8). |
| Corrected-Mg smoke | **PASS** — every candidate now gets Mg on the real batch path (valid_metal_setup, openff=false, n_angle_nan=0). |
| E1 explicit-solvent | **CONDITIONAL_PASS** — explicit solvent **fixes the co-substrate diffusion** implicit GBSA suffered (WT retention 0.52→1.00, escape now False) → a **meaningful WT baseline** for the first time. But O→Pα relaxes to ~5 Å (not productive ~3 Å); no clean discrimination. |
| E2 crystal-grounded | Crystal **5MST** (AMP + fumarate + metal) sets the productive target **O→Pα ~2.8–3.0 Å, in-line**. The explicit MD holds the **angle** (WT always ≥150°) but **never the productive distance** (occupancy of ≤3.6 Å = 0 for all). |

## 3. Banked statement (fixed wording)
> **CAR V5 produced valid, Mg-consistent, explicit-solvent adenylation-geometry evidence, but no
> candidate-level catalytic ranking. The classical MD tier identified a method limitation: the
> productive O→Pα near-attack distance is not occupied under the current *unbiased* explicit-MD
> observable.**

The word *unbiased* matters: "not occupied under unbiased MD" is not the same as "the FF is wrong."
The near-attack conformation is a **transient** reactive configuration, so the correct next observable
is the *free-energy cost to access it* (E4a umbrella/PMF, §7), not whether unbiased MD sits there.

## 3b. Honest verdict — the CAR reaction question
EvoLiEZ produced **valid, Mg-consistent, explicit-solvent reaction-geometry evidence** with a
crystal-grounded productive reference. The finding is a **clear mechanism/method limitation**: the
classical fixed-charge force field maintains the in-line **angle** but cannot hold the productive
**~3 Å attack distance** — expected, since the reactive-complex (partial P–O bond formation / charge
transfer) is inherently quantum. **Resolving the reaction-distance question requires QM/MM (deferred).**
This is **not** a solvent defect (E1 fixed that) nor a reference defect (E2's crystal target is correct).

## 4. Claim status (ClaimGuard-clean)
**Allowed / stated:**
- "Explicit solvent restored a meaningful WT baseline that implicit GBSA could not."
- "Crystal-grounded productive target is ~2.8–3.0 Å in-line; classical explicit MD holds the angle
  but not the distance — a force-field limitation requiring QM/MM to resolve the reaction distance."
- "Screening-level, Mg-consistent adenylation-geometry evidence; hypothesis-grade only."

**Withdrawn / forbidden (no experimental data):**
- The earlier focused 2 ns candidate ranking is **withdrawn** (it ran candidates without Mg —
  confounded). **Candidate-level rankings from that run must not be used.**
- NOT "validated catalytic lead", "activity improved", "productive mutant confirmed", "control beats
  lead", "WT is best". Retention and angle are **not** catalysis. **No wet-lab plate is recommended.**

## 5. CAR acceptance conditions — met
- ✅ Mg-consistent trajectories, WT + all candidates.
- ✅ Explicit-solvent subset → finite, interpretable O→P geometry.
- ✅ Meaningful WT baseline.
- ✅ A clear sampling/mechanism limitation is reported (classical FF cannot hold the productive
  distance → QM/MM).
- ✅ No wet-lab or activity claim made without experimental data.

## 6. Bugs fixed en route (PRs #8–16)
metal fan-out drop (#8) · **systemic empty-GPU-bucket** s08b (#9) + s06b×2 (#13) · s11 UTF-8 locale-C
(#12) · E1 explicit-solvent path (#11) · active-state ensemble builder (#15) · s06b 40→3 homologs ·
E2 crystal reference + accommodation (#16). Corrected-Mg (#10) and result docs (#7/#14) merged.

## 7. Deferred / next — correct escalation order
- **E4a — O→Pα umbrella / PMF access-barrier screen (do FIRST, before QM/MM).** Measure the
  free-energy cost for each candidate to *access* the near-attack geometry, using the existing
  classical FF (cheap, low-risk). Only if that PMF is flat / uninformative does QM/MM become
  justified. Built: `evoliez/md/umbrella.py` (WHAM verified) + `run_md(umbrella=…)` +
  `scripts/e4a_umbrella_pmf.py`; server run pending. See `e4a_umbrella_pmf_plan.md`.
- **QM/MM-lite** — only *after* E4a, and only to check reaction-core electronic plausibility on the
  best frames; never a final activity claim (needs wet-lab calibration).
- **Generality → a REAL non-redox backend run.** The mechanism-configurable framework is demonstrated
  across 3 classes, but FDH + metalloenzyme are **mock**; a real non-redox target (TEM-1 / glycosidase)
  is needed to move from *foundation* to *proof*.

## 8. Positioning — "commercial-platform FOUNDATION", not "commercial-ready"
The three completion criteria are met at the **architecture-proof / foundation** level: the framework
is mechanism-configurable, claim-safe, and reproducible-from-commit. It is **not** a shippable product:
still open are a real non-redox backend run, license/packaging/CI, artifact-storage policy, the 9
pre-existing test failures, and ≥2 real benchmarks (see `docs/PRODUCTIZATION_FOUNDATION.md`). Describe
the current state as **"commercial-platform foundation + one real hard case (CAR) + mock generality
proof"**, not "commercial-ready."

Deliverables: `docs/car_v5/` — corrected_mg_*, e1_explicit_*, e2_* (report, geometry/accommodation
tables, reference ensemble, verdict JSONs).
