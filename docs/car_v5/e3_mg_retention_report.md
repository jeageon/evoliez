# E3 — Mg-retention diagnostic (reviewer Day 2)

Run analysed: `runs/srcar_3hp_v5` (focused 2 ns, merged code `4d0eb7e`). Read-only, from the saved
trajectories + per-candidate `analysis.json`. Tooling: `scripts/e3_mg_retention.py`, table in
`e3_mg_retention_table.csv`.

## Headline: E3 found a real BUG, not just a sampling limit

The reviewer asked E3 to separate three causes of the focused CONDITIONAL_PASS: Mg-bridge collapse,
substrate pose diffusion, or implicit-solvent sampling. The trajectories answer it — and expose a
confound that invalidates the WT-vs-mutant comparison.

### Finding 1 — every system starts productive, every system diffuses
From the 19 `analysis.json` NAC blocks: **all 19 (WT + 18) start at `distance_initial = 3.20 Å`,
`angle_initial = 180°`** — the anchored build places every system in a perfect near-attack pose. But
**0 / 19 retain it** (`retention_fraction ≥ 0.8`); every one escapes (`escape = true`), drifting to a
mean O→Pα of 5.5–6.8 Å. Retention: WT 0.52, median 0.50, min 0.04.

### Finding 2 — the WT reference had Mg; NONE of the 18 candidates did (BUG)
Inspecting the MD topologies: `_wt_reference_minimized.pdb` contains an `MG` residue; **every
`mut_XXXXX_minimized.pdb` contains zero Mg**. So the WT MD ran with the bridging cation and all 18
candidate MDs ran **without it**.

Root cause: `s10` runs the WT reference through a direct `run_md(..., metal_requested=True)` but fans
every candidate through `run_md_batches` (multi-GPU). That fan-out **never forwarded
`metal_requested`** — the payload tuple and worker dropped it — so candidates defaulted to
`metal_requested=False`. PR #5 threaded the flag into `run_md`/`_run_real` but missed the batch path.
It only bites a >1-GPU run, which is every real run; the smoke had it too.

**Consequence:** the focused "no discrimination / lead has the worst distance / a control has the best"
result is **confounded and not interpretable** — the candidates had no Mg, so unopposed 3-HP⁻/ATP⁴⁻
repulsion (not the mutation) drove their diffusion. The lead's 0.04 retention (worst) is consistent
with "most-charged design, no counter-cation," not with a real mutation effect.

## Fix
`md_batch.py`: thread `metal_requested` through `run_md_batches` → payload → `_md_chunk_worker` →
`run_md`; `s10_md.py`: pass it at the fan-out call site. Regression locked in `test_md_batch.py`
(worker must forward `metal_requested`). Suite: 9 pre-existing, 0 new.

## What this means for the plan
- The focused run must be treated as **invalid for candidate comparison** (WT-only Mg). Any prior
  candidate-level reading is withdrawn.
- **But** the WT — which *did* have Mg — still diffused (retention 0.52, productive pose lost before
  the first saved frame). So Mg presence alone does not hold the cluster in implicit solvent: the
  E1 explicit-solvent escalation is still indicated.
- Correct sequence: (fix landed) → re-run the MD with Mg in **all** systems, and do it in **explicit
  solvent** (E1) rather than repeating implicit (WT already shows implicit diffuses). The E1 subset
  run therefore tests both fixes together — no separate wasted implicit re-run.

## Claim status
Unchanged and now firmer: no candidate is prioritized; the focused geometry is **withdrawn as
confounded**; next is a corrected explicit-solvent subset run. No "lead", no "control beats lead".
