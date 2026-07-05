# E1 explicit-solvent subset — acceptance report (reviewer Day 5)

Run: `runs/srcar_3hp_v5_e1`, merged code `feat/server-hardening` (PRs #8–13). WT + 4 candidates,
**explicit TIP3P + PME + Na⁺/Cl⁻ (0.15 M)**, 0.3 ns, 1 replica, RBFE/GBSA off, Mg-consistent (PR #8).
Co-substrate retention restraint identical to the implicit baseline (controlled solvent contrast).

## Verdict: **CONDITIONAL_PASS** — with a clear positive: explicit solvent fixes the diffusion

The explicit-solvent feature ran end-to-end for all 5 systems (~95k waters each, Mg as an Amber ion,
`solvent_mode=explicit`, `openff=false`, `n_angle_computed=4/4`, `n_angle_nan=0`, s11 clean).

### The headline — explicit solvent RETAINS the co-substrate that implicit lost
| system | retention (impl→expl) | escape | d_min Å | angle° |
|--------|-----------------------|--------|---------|--------|
| WT | **0.52 → 1.00** | **False** | 4.70 | 169 |
| G430R;S433F;G407K | → **0.88** | **False** | 4.96 | 98 |
| P438N | → 0.12 | True | 5.27 | 112 |
| G430H;P438L;A275I | → 0.54 | True | 5.47 | 131 |
| T265S;G274A;A275V | → 0.14 | True | 5.41 | 95 |

**WT retention rose 0.52 → 1.00 and no longer escapes** — implicit GBSA could not hold the charged
3-HP⁻/ATP⁴⁻/Mg²⁺ cluster, explicit water + counter-ions does. This gives a **meaningful WT baseline**
for the first time (the focused/implicit WT diffused → no baseline). The E1 hypothesis and the E3
diagnosis are confirmed: the diffusion was a solvent-model artifact, not a pipeline defect.

### The remaining gap — retained, but not yet productive
Even retained, the reactive O→Pα distance stays ~4.7–5.5 Å (not the productive ~3 Å), and angles are
mixed (WT 169°, others 95–131°). The provenance judge reports `discriminates=false`: the candidates do
not cleanly separate on the NAC geometry metric at 0.3 ns / 1 replica. There IS retention variation
(WT + lead escape=False; three others escape=True), but that is **retention, not catalysis**, and is
noisy at this tier.

## Honest classification
Reviewer CONDITIONAL_PASS, **"finite + Mg-tracked + meaningful WT baseline, but no clean discrimination"**
branch — with the important upgrade over implicit that the WT baseline is now real. Next tier (Day 6):
**active-state reference ensemble** (anchor the productive Michaelis pose so sampling starts and stays
near the ~3 Å attack geometry) and/or **QM/MM-lite** on the best-retained frames; optionally longer
sampling / more replicas.

## Claim status (ClaimGuard-clean)
- ✅ "explicit solvent retains the co-substrate that implicit GBSA lost (WT 0.52→1.00, escape now False);
  a meaningful WT baseline is established."
- ✅ "screening-level adenylation-geometry evidence, Mg-consistent, explicit-solvent — retained but not
  yet at the productive ~3 Å distance."
- ❌ NOT "validated lead", NOT "activity improved", NOT "WT/lead is best" (retention ≠ catalysis, noisy),
  NOT "productive mutant confirmed". No wet-lab plate, no EvidenceCard flip — the discrimination gate is
  not passed. Prior focused ranking stays withdrawn.

## What E1 proved for the platform
The explicit-solvent MD path (TIP3P/PME/neutralize, Amber Mg, read-only angle) is a **working,
server-verified capability**, and it demonstrably changes the physics (retention) in the diagnosed
direction. The CAR reaction-geometry question now has a valid, Mg-consistent, explicit-solvent evidence
base to escalate from.
