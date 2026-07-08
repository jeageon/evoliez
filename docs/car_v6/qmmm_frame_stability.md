# QM/MM-lite frame-stability check (optional next-tier, not a V6 gate)

**Question (reviewer):** is the QM/MM-lite in-line-angle difference between the CAR
variants a **single-frame artifact**, or does it hold across representative frames?

**Method:** 3 representative frames per system extracted from the unbiased 0.2 ns
production trajectory (`prod_0.nc`, frames 62/125/187 of 250); each QM/MM-relaxed
(QM = 3-HP + ATP + Mg, PM6 in sander; protein MM, backbone restrained) and the
QM-relaxed in-line angle (O_nuc–Pα–O_leaving) measured with cpptraj. ~66 s/frame,
server (evo). Provenance: `reports/provenance/amber_qmmm_stability_{wt,LEAD,P438N}.jsonl`.

## Result — the signal is stable

| system | in-line angle, 3 frames (°) | mean ± sd (°) | O_nuc→Pα (Å) |
|--------|------|------|------|
| WT | 93.3 · 94.8 · 85.3 | **91.1 ± 4.2** | ~5.9 |
| P438N | 121.0 · 120.5 · 127.4 | **123.0 ± 3.1** | ~7.3 |
| G430R;S433F;G407K (lead) | 163.1 · 169.8 · 166.7 | **166.5 ± 2.7** | ~8.5 |

- **Every mutant frame exceeds every WT frame**; within-system spread is tight
  (±3–4°) while between-system separation is large (WT 91° → lead 167°).
- The ordering **lead > P438N > WT** is consistent across all frames.
- **Verdict: not a single-frame artifact** — the in-line-angle ordering seen in MD
  survives QM inspection and is reproducible across the ensemble.

## What this does — and does not — establish

- It establishes that the **angle-ordering is stable**, strengthening the
  mechanism-probe rationale for placing the lead (and P438N) on the panel.
- It does **not** establish that the lead is mechanistically superior. The mutants
  sit at **larger** O_nuc→Pα distance (lead 8.5 Å vs WT 5.9 Å) yet show more in-line
  angles — the **angle and distance still disagree**. A more in-line angle at a
  longer distance is not unambiguously more productive.
- These are **unbiased-ensemble** angles (O→P ~6–9 Å), **not near-attack** (~3.2 Å)
  angles. The mechanistically-relevant near-attack in-line angle (the earlier
  single-frame QM/MM read WT ~69°) would require matched near-attack frames for the
  mutants — a restrained-pull/umbrella step deferred as a further tier.
- Semiempirical (PM6), single-trajectory frames, no wet-lab calibration.

## Claim ceiling

Screening-level reaction-geometry evidence — reaction-core plausibility only.
**No activity, kcat, activation-barrier, or validated-lead claim.** The lead and
P438N remain hypothesis-grade mechanism-probe candidates; the panel still requires
its deconvolution variants and controls to be assayed together.
