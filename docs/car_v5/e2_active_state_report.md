# E2 — crystal-grounded active-state reference ensemble (reviewer Day 2–4)

E1 showed explicit solvent fixes gross diffusion but the pose relaxes to a non-productive ~5 Å O→Pα.
The diagnosis moved to "the reference/initial pose is not in the true active-state basin." E2 grounds
the productive target in **crystal structures** (SrCAR, Gahloth et al.) instead of the relaxed MD.

## Step 1 — the productive geometry from the crystals (real, not a fixture)
`scripts/e2_build_reference_ensemble.py` on `car/refs/*.cif`:
- **5MST** = AMP (adenylate) + **FUM** (fumarate, the carboxylate substrate analog) + Ca — the
  productive Michaelis-like state. Extracted O_nuc(carboxylate)→Pα(AMP): **2.79 Å (chain A) / 3.05 Å
  (chain B)**, in-line 120–145°. (Crystal Ca ions are peripheral, 21 Å away — no catalytic metal
  resolved; the O→Pα arrangement is the anchor.)
- 5MSV = NADP/PCP domain (not the A-site); 5MSW = AMP-only (no reactive pair).

**Productive reference window: O→Pα ≈ 2.8–3.0 Å, in-line.** Reference ensemble built from the 2 crystal
copies (`e2_reference_ensemble.json`, `L0_uncalibrated`, honestly flagged `insufficiency = only 2
measured frames < 8` — 5MST has only 2 A-site copies with the reactive pair).

## Step 2 — accommodation readout vs the crystal window (`scripts/e2_accommodation.py`)
Scored the existing E1 explicit trajectories by occupancy of the productive window (≤3.6 Å **and**
≥150°) — no new MD:

| system | prodOcc | distOcc (≤3.6 Å) | angOcc (≥150°) | d_min | d_med |
|--------|---------|------------------|----------------|-------|-------|
| WT | 0.00 | 0.00 | **1.00** | 4.70 | 5.13 |
| G430R;S433F;G407K | 0.00 | 0.00 | 0.00 | 4.96 | 5.69 |
| P438N | 0.00 | 0.00 | 0.00 | 5.27 | 6.40 |
| G430H;P438L;A275I | 0.00 | 0.00 | 0.00 | 5.47 | 5.98 |
| T265S;G274A;A275V | 0.00 | 0.00 | 0.00 | 5.41 | 6.51 |

## Finding — the force field holds the ANGLE but not the productive DISTANCE
- **No system occupies the productive window** (`prodOcc=0` everywhere): the E1 MD started at 3.2 Å
  and relaxed to ~5 Å, never re-entering ≤3.6 Å. The classical explicit-solvent force field does **not**
  stabilise the tight ~3 Å in-line attack distance the crystals show is productive.
- **WT holds the in-line angle** (`angOcc=1.00`, always ≥150°) while the others do not — a partial
  geometric distinction on *angle*, but this is WT (the reference) and angle-only is **not catalysis**.

## Honest classification & claim status
The crystal reference is correctly constructed (~2.8–3.0 Å productive) — this is **not** a
reference-construction failure. The relaxation is a **force-field limitation** for the reactive-complex
geometry (partial P–O bond formation / charge transfer that a fixed-charge FF cannot represent). Per the
reviewer's E2 branch, "geometry finite but the productive distance is not held" → **E4 QM/MM-lite** on
the best in-line frames is the indicated next tier.

- ✅ "the crystal-grounded productive target is ~2.8–3.0 Å in-line; classical explicit MD holds the
  angle but relaxes the distance to ~5 Å — a force-field limitation, not a solvent or reference defect."
- ❌ NOT "validated lead / activity / WT is best / control beats lead" — none supported; retention and
  angle are not catalysis; the focused ranking stays withdrawn.

## Caveat (what would make this rigorous)
This readout uses the E1 trajectories (anchored to a Boltz-docked ~3.2 Å pose, not the crystal pose). A
**crystal-anchored E2 smoke** — place the co-substrate at the 5MST productive arrangement and re-run
short explicit MD — would confirm whether a *crystal-basin* start is held any better. The E1 evidence
(a near-productive 3.2 Å start still relaxed) is a strong prior that it will not, which is why E4 is the
leading recommendation — but the crystal-anchored smoke is the clean confirmation if desired.
