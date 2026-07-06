# CAR V5 — sign-off: a classical-force-field mechanism-limitation case

**Status: BANKED.** SrCAR → 3-HP adenylation is closed out not as an activity result but as a clean,
claim-safe demonstration that the platform (a) computes Mg-consistent reaction-geometry *evidence*,
(b) discovered and fixed a real platform bug through its own diagnostics, and (c) honestly reports the
mechanistic tier at which classical MD stops being informative. No candidate ranking, no wet-lab plate,
no activity claim is made or implied.

Consolidates: E3 diagnostic + PR #8 confound fix · corrected-Mg verification · E1 explicit solvent · E2
crystal-grounded active state. Detail lives in `e3_mg_retention_report.md`, `corrected_mg_smoke_report.md`,
`e1_explicit_subset_report.md`, `e2_active_state_report.md`, `CAR_V5_EVIDENCE_PACKAGE.md`.

## The arc (what happened, in order)

| step | finding | disposition |
|------|---------|-------------|
| **Focused 2 ns (prod)** | looked like CONDITIONAL_PASS, 18/18 finite angles | **WITHDRAWN — confounded** |
| **E3 diagnostic** | WT had Mg but **all 18 candidate MDs ran without it** | real bug found |
| **Root cause / PR #8** | `run_md_batches` fan-out never forwarded `metal_requested` (WT ran serial, kept Mg) | fixed + regression |
| **corrected-Mg smoke** | WT **and** candidates now carry MG; `valid_metal_setup`, `openff=false`, 0 NaN | fix server-verified |
| **E1 explicit solvent** | explicit TIP3P/PME **fixes gross diffusion**: WT retention 0.52→1.00, meaningful WT baseline | CONDITIONAL |
| **E2 crystal reference** | crystal (5MST) productive O→Pα = **2.8–3.0 Å in-line**; but **no MD system occupies ≤3.6 Å** | force-field limit |

## The verdict — a force-field limitation, not a pipeline/solvent/reference defect

The crystals prove the productive near-attack basin exists at ~2.8–3.0 Å. Explicit solvent removed the
gross-diffusion artifact and restored a real WT baseline. Yet even from a near-productive start the
fixed-charge MD relaxes the reactive distance to ~5 Å and never re-enters the productive window
(WT holds the in-line *angle* but not the *distance*). This is the signature of a **classical
fixed-charge force field being unable to represent the partial P–O bond / charge transfer of the
reactive complex** — the mechanistic tier where classical MD is expected to fail. Deeper CAR would
require **QM/MM-lite** on the best in-line frames; that is *not* pursued here because the platform
priority is generality, and CAR has already served its purpose as the acceptance case.

## Withdrawn / forbidden (claim discipline — permanent for this run set)

- Withdrawn: *"Previous focused candidate comparison is withdrawn due to inconsistent Mg inclusion.
  E3 found and fixed a real multi-GPU MD batch bug. Candidate-level rankings from that run must not be used."*
- Forbidden: "lead failed", "control beats lead", "validated catalytic lead", "productive mutant
  confirmed", "activity improved". Retention and angle occupancy are **not** catalysis.

## What CAR proved for the platform (the reason it was worth doing)

1. The diagnostic layer (E3) **caught a real confound the headline verdict hid** — evidence-integrity
   working as designed.
2. Mg co-fold + Amber-ion MD + O→Pα geometry + explicit-solvent path are **server-verified machinery**.
3. ClaimGuard held throughout: a genuinely negative/limited result was reported **without overclaim**.
4. The platform can say *"here is the mechanistic tier at which our method stops being informative,"*
   which is exactly what a claim-safe triage framework should do.

## PRs
#8 (metal fan-out) · #9/#13 (empty-bucket) · #10 (corrected-Mg) · #11 (explicit solvent) · #12 (UTF-8) ·
#16 (E2 crystal reference) · #17 (evidence package). Server runs reproducible from `feat/server-hardening`;
large trajectories kept out of GitHub.

## Next (platform, not CAR)
CAR is closed. Main focus → **generality demonstration**: FDH (hydride) + CAR (adenylation) + a non-redox
enzyme, all triaged through the **same MechanismSpec → EvidenceCard → ClaimGuard** framework with no
core-code edits, proving the commercial-platform criterion.
