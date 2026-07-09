# QM/MM-lite frame-stability check (optional next-tier, not a V6 gate)

**Question (reviewer):** is the QM/MM-lite in-line-angle difference between the CAR
variants a **single-frame artifact**, or does it hold across representative frames?

**Answer:** partly an artifact. At a **non-reactive** distance the variants separate
cleanly; at the **reaction-relevant near-attack** distance the separation largely
**dissolves**, and the earlier single-frame WT value was an outlier. This is a
claim-safe *cautionary* result — it argues against over-reading the QM/MM angle
signal, and keeps CAR firmly hypothesis-grade.

**Method:** 3 frames per system, each QM/MM-relaxed (QM = 3-HP + ATP + Mg, PM6 in
sander; protein MM, backbone restrained), QM-relaxed in-line angle (O_nuc–Pα–O_leaving)
measured with cpptraj (~66 s/frame, evo). Two frame sources:
(a) **production** frames from the unbiased 0.2 ns trajectory (O→P ~6–9 Å);
(b) **near-attack** frames from a short O_nuc→Pα-restrained pull to ~3.2 Å.
Provenance: `reports/provenance/amber_qmmm_frame_stability.jsonl` (18 records).

## (a) Production frames (unbiased ensemble, O→P ~6–9 Å)

| system | in-line angle, 3 frames (°) | mean ± sd (°) | O→P (Å) |
|--------|------|------|------|
| WT | 93.3 · 94.8 · 85.3 | 91.1 ± 4.2 | ~5.9 |
| P438N | 121.0 · 120.5 · 127.4 | 123.0 ± 3.1 | ~7.3 |
| lead (G430R;S433F;G407K) | 163.1 · 169.8 · 166.7 | 166.5 ± 2.7 | ~8.5 |

Clean, tight separation (lead > P438N > WT, every frame ordered). **But this is at a
non-reactive distance** — and the more in-line variants also sit *further* out, so the
angle and distance disagree.

## (b) Near-attack frames (restrained to O→P ~3.2 Å — the reaction geometry)

| system | in-line angle, 3 frames (°) | mean ± sd (°) | O→P (Å) |
|--------|------|------|------|
| WT | 137.9 · 154.1 · 136.0 | **142.7 ± 8.1** | ~3.1 |
| lead (G430R;S433F;G407K) | 153.0 · 156.7 · 158.4 | **156.0 ± 2.3** | ~3.6 |
| P438N | 100.9 · 109.5 · 116.8 | **109.1 ± 6.5** | ~3.5 |

**At the reaction geometry the story changes:**
- **WT is not robustly bent.** The earlier single-frame QM/MM read WT ≈ 69° at
  near-attack (from a PMF umbrella window); across 3 restrained-pull near-attack
  frames WT is **142.7 ± 8.1°**. Whether the discrepancy is frame sampling or the
  different restraint protocol, the WT near-attack in-line angle is **not reproducibly
  bent** — so the "WT is bent, lead is in-line" contrast is not robust.
- **lead (156°) ≈ WT (143°)** — the distributions overlap (a WT frame reached 154°);
  the large production-frame gap does **not** carry to the reaction distance.
- **P438N (109°) is *below* WT** at near-attack — the opposite of its production-frame
  ranking.

## Conclusion (claim-safe)

- The apparent QM/MM in-line-angle advantage of the lead was **substantially a
  frame/geometry artifact**: it is large only at a non-reactive distance and the WT
  baseline it was measured against (69°) was an outlier frame.
- At the reaction-relevant near-attack geometry the variants are **not robustly
  differentiated** by in-line angle (WT 143 ± 8, lead 156 ± 2, P438N 109 ± 7 —
  overlapping/mixed, P438N worse).
- This **does not kill the panel** — it reinforces the claim boundary: the CAR
  candidates remain **hypothesis-grade mechanism probes** whose ranking cannot be
  established computationally. The deconvolution variants and controls must be assayed
  together; only wet-lab data can differentiate them.

## Claim ceiling

Screening-level reaction-geometry evidence — reaction-core plausibility only.
**No activity, kcat, activation-barrier, or validated-lead claim.** The multi-frame
check *weakened* the computational case for differentiating the CAR variants; it did
not create one.
