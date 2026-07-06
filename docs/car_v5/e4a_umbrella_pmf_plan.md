# E4a — O_nuc→Pα umbrella / PMF access-barrier screen

## Why this before QM/MM
E2 found unbiased explicit MD never occupies the productive ~3 Å window. But the near-attack
conformation (NAC) is a **transient reactive configuration, not a stable minimum**, so "does it sit at
3 Å?" is the wrong observable — and expecting a rare NAC in 0.3 ns unbiased MD is an ill-posed
measurement. The right question is the **free-energy cost to ACCESS the near-attack geometry**, which
the existing classical FF can answer via umbrella sampling along the O_nuc→Pα distance + WHAM —
cheaper and lower-risk than QM/MM, and the correct next tier. QM/MM-lite is entered only if this PMF
screen is flat or the reference/protonation is implicated.

## What was implemented (verified core)
- `evoliez/md/umbrella.py`: `umbrella_windows()`, `add_umbrella_bond_restraint()` (harmonic bias on the
  O_nuc–Pα **distance only** — the angle is never biased, so NAC cannot be manufactured), a pure-python
  iterative **WHAM** (`wham()`, no pymbar dep), and `access_free_energy()` (near-attack access cost).
  `test_v5_umbrella_pmf.py` verifies WHAM **recovers a known harmonic PMF** from synthetic biased samples.
- `run_md(umbrella=(window_A, k))`: adds the harmonic O_nuc→Pα bias for one window and writes its DCD
  (reuses the whole explicit-solvent + Mg build).
- `scripts/e4a_umbrella_pmf.py`: the analysis driver — reads the per-window DCDs, extracts the O_nuc→Pα
  timeseries, WHAM → PMF → per-candidate access cost, ClaimGuard-safe wording.

## How to run (server, per the reviewer's E4a spec)
Subset: WT · G430R;S433F;G407K · P438N · G430H;P438L;A275I · T265S;G274A;A275V.
1. **Sample** (server, OpenMM). For each candidate, for each window `d0` in
   `umbrella_windows(2.8, 5.4, ~14)`: `run_md(..., solvent=explicit, umbrella=(d0, k=10))` into
   `runs/car_v5_e4a/<cand>/window_<d0>/`. (k ≈ 10 kcal/mol/Å² is a starting force constant; windows
   overlap check before WHAM.)
2. **Analyse**: `PYTHONPATH=src python scripts/e4a_umbrella_pmf.py runs/car_v5_e4a --k 10 --window-nac 3.6`.

## Success / conditional / fail (reviewer)
- **PASS**: PMF converges, and the relative access cost to the ≤3.6 Å window **separates candidates**
  (spread > ~1 kcal/mol) → screening-level access-barrier evidence (NOT a rate/activity claim).
- **CONDITIONAL**: PMF converges but candidate costs are flat (~0 spread) → classical FF cannot
  discriminate the reaction distance → escalate to **QM/MM-lite** on the best frames.
- **FAIL**: PMF unstable / poor window overlap → re-check restraint k / window spacing / protonation /
  Mg coordination.

## Claim discipline
Access cost is **not** kcat and **not** activity. Lower cost = the candidate reaches the near-attack
geometry more cheaply under this FF — a prioritisation signal, not a rate. No wet-lab plate, no
"validated lead". The prior focused ranking stays withdrawn.
