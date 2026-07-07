# E4a — O_nuc→Pα umbrella / PMF access-barrier: results & verdict

**Run:** SrCAR → 3-HP adenylation, V5. Server `evo`, explicit solvent (TIP3P+PME) + Mg²⁺, k in kcal/mol/Å².
**Verdict (all 4 sampling attempts):** **NOT CONVERGED → the access costs are NOT interpretable.** This is
an honest, claim-safe outcome — the convergence guard fired every time and never let an apparent ranking
through as a result.

## TL;DR
The classical fixed-charge force field **cannot cleanly converge a 1D umbrella PMF along the O_nuc→Pα
distance for this reaction.** With a strong restraint (k=60) the individual windows are well-behaved
(monotonic, tight), but a **coverage gap persists and *shifts* when infilled** (≈4.0 Å → ≈4.7 Å) — the
signature of a reaction coordinate coupled to a slow hidden variable (co-substrate metastable pocket
sub-states) that 40 ps windows cannot bridge. Consistent with E1 (explicit solvent) and E2 (crystal
reference): **classical FF holds the near-attack *angle* but not the productive ~3 Å *distance*.**

## Why E4a (recap)
E2 showed unbiased explicit MD never occupies the productive ~2.8–3.0 Å window (the crystal target, 5MST:
O→Pα **2.79 Å**). But the near-attack conformation is a *transient* state, so "does it sit at 3 Å?" is the
wrong question. E4a asks the right one — the **free-energy cost to *access*** the near-attack geometry —
via umbrella sampling along the distance + WHAM. The reactive **angle is never biased** (only the
distance), so the bias cannot manufacture the in-line NAC.

## The four sampling attempts

| run | k | windows | window pinning | residual gap | verdict |
|---|---|---|---|---|---|
| k=10 | 10 | 30 | ✗ windows slide into local basins (non-monotonic) | many | not converged |
| k=60 / 24w | 60 | 72 | ✅ monotonic, std ~0.09 Å | one, ≈4.0 Å | not converged |
| k=60 + infill (3.6–4.6) | 60 | 114 | ✅ | one, **moved to ≈4.7 Å** | not converged |
| k=60 combined (canonical) | 60 | 129 | ✅ | one, ≈4.7 Å | not converged |

Each infill demonstrably closes its own region — and the last unbridged transition simply shifts. That
"whack-a-mole" behavior is the diagnostic: it is *not* random noise (all three candidates show the gap at
the **same** distance), so it reflects a real coordinate-coupling / equilibration limit, not under-tuning.

## Final numbers (canonical 129-window set) — reported, NOT interpreted
`docs/car_v5/e4a_pmf_access_table_FINAL_129w.csv`

| candidate | windows | access cost (kcal/mol) | in-line angle occ | overlap_min | converged |
|---|---|---|---|---|---|
| WT (`_wt_reference`) | 53 | 5.53 | 0.62 | **0.00** | **NO** |
| mut_00000 (lead G430R;S433F;G407K) | 38 | 2.90 | 0.80 | **0.00** | **NO** |
| mut_00001 (P438N) | 38 | 10.22 | 0.22 | **0.00** | **NO** |

The access costs swung wildly across k=10 ↔ k=60 ↔ infill (e.g. mut_00000 7.71 → 0.14 → 2.90) — that
instability is itself proof of non-convergence. **`converged=NO` ⇒ these numbers must not be ranked or read
as an ordering.** The `access-barrier` and in-line `NAC occupancy` columns are kept separate by design: the
umbrella biases distance only, so a short distance is not necessarily the in-line NAC.

## What *did* work
- **Machinery validated:** per-window sampling produced correct biased distributions; angles were always
  finite and in-line (~163°, unbiased) — confirming the V5-1 O_nuc–Pα–O_leaving angle math and that the
  angle is never restrained.
- **The near-attack region (≤3.6 Å) is well-sampled and overlapping** (the low-distance windows pin in
  3.1–3.7 Å with good overlap). Only the *global* PMF — which references the relaxed ~5 Å basin — crosses
  the ≈4.7 Å gap, so it is the global reference that is unreliable, not the near-attack coverage.
- **The convergence guard did its job:** it refused to interpret every non-converged run, so no false lead
  was ever produced. That is the intended behavior of a claim-safe triage platform (not an activity
  predictor). The prior focused ranking stays **withdrawn**.

## Honest endpoint & next-tier options
Classical FF methods are now exhausted for the productive ~3 Å geometry (E1 + E2 + E4a agree). The next
tier is a decision, not an automatic step:
1. **Consolidate + pivot to generality** — accept the FF-limitation finding as the E4a result and move to
   the platform's stated priority (demonstrating generality on other mechanisms: FDH / TEM-1 / …).
2. **QM/MM-lite** — the reviewer's designated deeper-CAR tier: QM/MM on the best-sampled near-attack frames
   (the ≤3.6 Å region *is* well-covered, so those frames exist). New capability + GPU cost.
3. **Longer windows / 2D coordinate** — add the hidden variable (co-substrate pocket state) as a 2nd
   dimension and lengthen per-window MD. Uncertain payoff; the FF's inability to hold ~3 Å may persist.

## Claim discipline
Access cost is a distance-coordinate free energy under a fixed-charge FF — **not** an activation free
energy (that needs QM/MM), and **not** kcat/activity. Allowed: "screening-level reaction-geometry access
evidence." Forbidden: catalytically-improved / lowers-activation-barrier / increases-kcat / validated lead.

## Data provenance
- `e4a_pmf_access_table_FINAL_129w.csv` — canonical verdict (129 windows, k=60).
- `e4a_per_window_diagnostic.csv` — per-window center/mean/std/min/max/angle (the sampling behavior).
- `e4a_pmf_access_table_{k10_nonconverged,k60_24w_gap,k60_38w_gap47}.csv` — the sampling-tuning history.
- Raw per-window `umbrella_samples.json` (129) stay on the server under
  `runs/srcar_3hp_v5_e4a/md/<cid>/e4a_k60mixed_archive/` (→ GDrive, not GitHub).
- A shared-box OOM (a different user's 258 GB process) killed the run mid-sweep once; the idempotent sweep
  + backups made it fully recoverable (no data lost).
