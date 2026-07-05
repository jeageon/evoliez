# CAR V5 focused 2 ns run — acceptance report (reviewer Day 5)

- Run: `runs/srcar_3hp_v5`, merged code `feat/server-hardening @ 4d0eb7e` (PR #4/#5/#6).
- Tier: protocol_level 3, **2 ns × 3 replicas**, RBFE off. Candidate set: reviewer-locked 19 (WT + 18).
- Manifest wiring validated in production: s07 seeded **exactly 18** (0.1 s), s08b folded **18** (not 41).

## Verdict: **CONDITIONAL_PASS** — machinery valid, geometry does NOT discriminate

| check | value |
|-------|-------|
| metal_status | `valid_metal_setup` (inserted, `amber14/tip3p ion (MG)`) |
| openff_parameterized | **false** (Mg is an Amber ion, never OpenFF) |
| geometry_source | `legacy_reactive_geometry_from_mechanism_spec` |
| n_candidates | 18 |
| n_angle_computed / n_angle_nan | **18 / 0** (every candidate got a finite O→Pα angle) |
| wt_baseline_present | true, **but WT co-substrate DIFFUSED** (angle/dist = None) |
| **discriminates** | **false** |

## What the 2 ns tier actually showed

**The evidence is valid but flat.** All 18 candidates produced finite reaction geometry (angle
117–160°, median 147°; O→Pα distance 3.74–5.43 Å, median 4.85 Å) with a correctly-tracked Amber
Mg²⁺ and zero NaN — so the s10 NAC + metal machinery is confirmed working at production tier. But:

1. **No discrimination.** Distances cluster in a 1.7 Å band with 3-replica noise; the judge finds no
   meaningful candidate separation — the same `discriminates=false` the 0.05 ns smoke reported. The
   median distance barely moved (smoke 4.6 Å → focused 4.85 Å), so co-folding Mg + a 20× longer
   trajectory did **not** close the substrate toward the productive ~3 Å.
2. **ΔNAC is undefined.** The WT reference co-substrate **diffused out of the site** in 2 ns, so there
   is no valid WT geometry baseline → `nac_delta_vs_wt = None` for every candidate. The catalytic-power
   axis (mutant vs WT) could not be computed.
3. **The raw geometry ordering does not support the prior lead.** The integrated lead
   `G430R;S433F;G407K` has the *largest* O→Pα distance (5.43 Å) while a scalar control
   (`G430H;P438L;A275I`) has the smallest (3.74 Å). Given `discriminates=false`, this ordering is
   **noise, not an inverted ranking** — it must NOT be read as "the control beats the lead."

## Honest classification

This is the reviewer's CONDITIONAL_PASS **"reference state / metal not physiologically stable at this
tier"** branch, not a pipeline failure. Diagnosis: the highly charged **3-HP⁻ / ATP⁴⁻ / Mg²⁺** cluster
is under-stabilized in the **implicit-solvent (GBSA)** s10 path — the co-substrate drifts, so neither
the WT baseline nor a productive candidate pose is held long enough to yield discriminating evidence.

## Claim status (ClaimGuard-safe)

- ✅ "screening-level adenylation-geometry evidence computed for all 18 candidates at 2 ns; machinery
  (Mg co-fold + Amber-ion MD + O→Pα geometry) validated."
- ✅ "no candidate is prioritized on geometry yet — the 2 ns implicit-solvent tier does not discriminate."
- ❌ NOT "validated catalytic lead", NOT "activity improved", NOT "productive mutant confirmed",
  NOT "control outperforms the lead" — none are supported.

**No wet-lab plate is recommended from this run** (recommending one off non-discriminating geometry
would be an overclaim). See `next_action_decision.md`.
