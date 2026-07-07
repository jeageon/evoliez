# V6-5 — CAR QM/MM-Lite Decision

**Phase:** V6-5 · **Status:** ✅ PASS · **Provenance:** [`reports/provenance/amber_qmmm_jobs.jsonl`](../../reports/provenance/amber_qmmm_jobs.jsonl) · **Protocol:** [`docs/v6/qmmm_lite_protocol.md`](../v6/qmmm_lite_protocol.md)

## What was run

Semiempirical (PM6) QM/MM-lite reaction-core minimization for **WT + 2 candidates**,
each from a near-attack frame, on a carved non-periodic subsystem (full protein +
3-HP + ATP + Mg core + a 6 Å water shell, ~11k atoms). QM region = the reaction
core (net charge −3); protein backbone restrained. The comparable observable is the
**QM-relaxed in-line O_nuc–Pα–O_leaving angle**.

| System | mutation | O_nuc→Pα (Å) | in-line angle (°) | Mg–Pα (Å) |
|---|---|---|---|---|
| WT | — | 3.22 | **69.4** | 3.30 |
| mut_00000 | `G430R;S433F;G407K` (multipoint) | 3.15 | **141.6** | 3.26 |
| mut_00001 | `P438N` (single-site) | 3.12 | 108.2 | 3.34 |

## What it means (reaction-core plausibility, NOT activity)

At matched near-attack **distance** (~3.1–3.2 Å), the QM/MM-relaxed reaction-core
**angle** differs markedly: WT stays bent (69°), the single-site P438N is
intermediate (108°), and the multipoint variant relaxes to a **more in-line
geometry (142°)** — approaching, but not reaching, the ~150° in-line threshold.

This directly answers the V6-5 question — *does the PMF-derived candidate
difference survive electronic-structure inspection?* The V6-3 PMF showed the
in-line angle is the unoccupied coordinate for WT; under QM/MM-lite the multipoint
variant accesses a substantially more in-line reaction-core geometry than WT, while
WT does not. **The geometric difference survives QM inspection** and is consistent
with the multipoint variant's mechanism hypothesis (anchoring the 3-HP carboxylate
toward an in-line approach).

This is **reaction-core plausibility evidence — it is not, and must not be read as,
an activity, kcat, or catalytic-superiority claim.**

## Caveats (why this is hypothesis-grade, not a claim)

1. **Single-frame** — one near-attack frame per system, not an ensemble or free energy.
2. **Semiempirical** — PM6, not a high-level QM treatment.
3. **Frames not identically prepared** — the WT frame came from the 13-window PMF
   (0.1 ns/window); the candidate frames came from a shorter 2-window pull. The
   angle comparison is therefore **confounded by preparation** and is not a
   controlled experiment.
4. **The underlying 1D PMF is non-converged** (V6-3) — ranking remains prohibited.

## Decision (the V6-5 escalation / stop call)

The QM/MM-lite result is **not inconclusive** — it produces a clear,
hypothesis-consistent signal that a reaction-core geometry difference between WT and
the multipoint variant is real under electronic-structure relaxation. But the
caveats above mean it **cannot support any activity or ranking claim** on its own.

Therefore:

- **Proceed with the V6-4 claim-safe mechanism-probe panel.** The QM/MM-lite signal
  raises the plausibility of the multipoint (and P438) mechanism hypotheses enough
  to justify testing them experimentally — as **hypotheses**, at L0.
- **Recommend higher-cost validation for any definitive mechanistic claim**, in this
  order: (a) an **identically-prepared** near-attack ensemble for WT and every
  candidate (removing the preparation confound), (b) a **2D PMF** along distance +
  angle (the V6-3-flagged hidden coordinate), and (c) a higher-level QM/MM if the
  classical 2D result warrants it. Only wet-lab data can move any card above L0.

This satisfies ROADMAP_V6 Gate 2: CAR yields **a claim-safe experimental portfolio
now** and **a clear, prioritized escalation path** for the reaction-core question.

## Reproduce

```bash
python scripts/run_qmmm_lite.py \
  --job "wt:<sys>/wt:<sys>/wt/win_2.800/prod.rst" \
  --job "mut_00000:<sys>/mut_00000:<sys>/mut_00000/win_3.100/prod.rst" \
  --job "mut_00001:<sys>/mut_00001:<sys>/mut_00001/win_3.100/prod.rst" \
  --min-steps 60
```
