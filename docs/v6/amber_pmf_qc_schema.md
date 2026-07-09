# V6-3 — Amber PMF QC & Claim Schema (server-verified)

**Phase:** V6-3 · **Status:** ✅ PASS · **Module:** [`src/evoliez/amber/pmf.py`](../../src/evoliez/amber/pmf.py)
**Driver:** [`scripts/run_amber_pmf.py`](../../scripts/run_amber_pmf.py) · **Tests:** [`tests/test_amber_pmf.py`](../../tests/test_amber_pmf.py) · **Provenance:** `reports/provenance/amber_pmf_jobs.jsonl`

## Goal

Replace the weak observable "does the system sit at 3 Å?" (V6-2 showed it never
does — near-attack is transient) with "what is the **free-energy cost to access**
near-attack geometry?" — an Amber `pmemd.cuda` umbrella sweep along O_nuc→Pα +
WHAM, with rigorous convergence QC and strict claim discipline.

## Reuse, not reinvention

The WHAM / access-cost / overlap / angle-occupancy **core is the engine-agnostic
`md.umbrella`**, reused verbatim so the OpenMM E4a tier and this Amber tier report
identical quantities. V6-3 adds only the Amber-native restraint + GPU orchestration.

## Restraint convention (physics-critical)

Amber's `&rst` distance restraint energy is `rk·(r-r0)²`, but `md.umbrella.wham`
expects the `0.5·k·(r-r0)²` convention. Therefore:

> **Amber `rk2 = rk3 = 0.5·k_kcal`**, and the SAME `k_kcal` is passed to WHAM.

Getting this wrong scales the PMF by 2×. It is unit-tested and **validated on real
MD**: in the smoke each window's sampled distance localized at its restraint centre
(3.0→3.17, 3.5→3.59, 4.0→4.10 Å), confirming the harmonic well is centred correctly.

The umbrella biases **distance only** — the reactive angle is never restrained, so
the bias cannot manufacture an in-line near-attack conformation.

## QC schema (`amber_pmf/v1`)

| Field | Meaning | Gate |
|---|---|---|
| `overlap.mean/min_overlap` | Bhattacharyya overlap of adjacent-window histograms | `min ≥ 0.10` → `sufficient` |
| `convergence.access_first_half / second_half_kcal` | near-attack access cost from each temporal half (independent WHAM) | drift `≤ 1.0 kcal/mol` → `halves_consistent` |
| `convergence.access_drift_kcal` | \|first − second\| | (as above) |
| `equilibration_discard_frac` | fraction of each window's leading samples discarded | 0.2 default |
| `near_attack_angle_occupancy` | of frames whose **distance** reached ≤ near-attack, the fraction also **in-line** (≥ angle_min) | reported, never gated (access vs NAC kept separate) |
| `windows_with_samples / n_windows` | window completion | ≥ 2 to attempt WHAM |

## Classification → claim ceiling

```
converged      = overlap.sufficient AND convergence.halves_consistent
```

| Verdict | Classification | Claim ceiling |
|---|---|---|
| `converged` | `screening_prioritization` | "screening-level near-attack **access-cost** evidence (converged PMF; NOT activity/kcat/activation-barrier)" |
| `diagnostic_only` | `diagnostic_only` | "**DIAGNOSTIC-ONLY** non-converged PMF — **prohibited from ranking** candidates (NOT activity/kcat/access-cost)" |
| `failed` | `diagnostic_only` | (as diagnostic) |

A non-converged PMF can never rank candidates — ROADMAP_V6 §6. This is enforced in
code (`classify_pmf`), not just documentation.

## Server validation

**Smoke (3 windows, 0.02 ns):** end-to-end tier works — biased pmemd per window
(one per GPU), distance localized at each centre, WHAM + QC + a `diagnostic_only`
verdict (3 windows 0.5 Å apart → 0 overlap, correctly detected).

**Production PMF (WT, 13 windows 2.8–4.6 Å, k=60, 0.1 ns/window, GPUs 0–3):**

| QC | Value | Gate |
|---|---|---|
| window overlap | mean 0.765, **min 0.435** | ✅ sufficient (≥0.10) |
| windows sampled | 13/13, 400 samples each | ✅ |
| half-convergence | 1st-half access 9.56, 2nd-half 8.16 kcal/mol, **drift 1.40** | ❌ inconsistent (>1.0) |
| in-line angle occupancy | **0.0** over 1827 near-attack-distance frames | reported (not gated) |
| **verdict** | **`diagnostic_only`** — access cost 8.39 kcal/mol is diagnostic, NOT a free energy | ranking prohibited |

Every window localized at its restraint centre (2.8→3.14, 3.4→3.58, 4.6→4.64 Å),
so the restraint + the `rk = 0.5·k` conversion are correct across the whole range.

**Deep analysis (the scientifically important part):**
1. **The 1D umbrella is well-executed** — adjacent windows overlap strongly (min
   0.435). This is *not* a sparse-window artifact; the sampling is adequate.
2. **Yet the PMF does not converge** — the first- and second-half access costs
   differ by 1.40 kcal/mol. Good overlap **with** failed temporal convergence is
   textbook evidence of a **hidden slow variable orthogonal to the O_nuc→Pα
   distance** (the ROADMAP_V6 §6 trigger for 2D escalation — e.g. Mg-bridge
   coordination or substrate orientation).
3. **The angle occupancy = 0.0 is the mechanistic diagnosis**: even when the
   umbrella *forces* the distance to near-attack (≤3.6 Å in 1827 frames), the
   in-line O_nuc–Pα–O_leaving angle is **never** productive (0/1827 ≥ 150°). The
   near-attack *distance* is accessible under bias; the near-attack *geometry* is
   not — the carboxylate approaches the α-phosphate off-axis. This is exactly why
   distance access and in-line NAC must be reported separately: the distance PMF
   alone would be a misleading "8.4 kcal/mol access cost."

**Outcome:** the Amber PMF tier works and delivers an honest, claim-safe diagnosis
— the 1D distance PMF is well-sampled but non-converged, and the in-line angle is
the missing productive coordinate. Because ranking on a non-converged PMF is
prohibited, this **directly drives the V6-4 decision toward a mechanism-probe
portfolio** (not a false "best mutant") and flags 2D / QM/MM escalation for V6-5.
It reproduces and sharpens the E4a five-run non-convergence history.

## Escalation note (1D → 2D)

If a well-overlapped 1D PMF still fails the half-convergence test, that is evidence
of a **hidden slow variable** (e.g. Mg bridge coordination, substrate orientation)
orthogonal to the O_nuc→Pα distance — the ROADMAP_V6 §6 trigger for 2D
enhanced-sampling escalation or the V6-5 QM/MM tier. The classification stays
diagnostic-only until a converged free energy is in hand.

## Success conditions — checklist

- [x] PMF windows generated, scheduled, and analyzed on GPUs (one pmemd per GPU).
- [x] Window overlap, equilibration discard, half-convergence, and (via replicate-free half split) consistency reported.
- [x] Distance access cost reported **separately** from in-line NAC occupancy.
- [x] Non-converged PMFs classified **diagnostic-only** and prohibited from ranking (enforced in `classify_pmf`).
- [x] Converged PMFs would carry only screening-level access-cost claims (path in place).
