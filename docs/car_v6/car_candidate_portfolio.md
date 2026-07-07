# V6-4 — CAR Candidate Portfolio (claim-safe mechanism-probe panel)

**Phase:** V6-4 · **Status:** ✅ PASS · **Generator:** [`scripts/build_car_v6_portfolio.py`](../../scripts/build_car_v6_portfolio.py)
**Deliverables:** [`outputs/car_v6/wetlab_plate_24.csv`](../../outputs/car_v6/wetlab_plate_24.csv) · [`outputs/car_v6/evidence_cards.json`](../../outputs/car_v6/evidence_cards.json)

## What this is — and what it is not

This is a **19-variant mechanism-probe portfolio** for CAR (SrCAR A-domain →
3-HP adenylation). It is a claim-safe, **L0-uncalibrated** experimental panel for
exploring structural/binding viability, reaction-**geometry evidence**, and
uncertainty. It is deliberately **not**:

- a ranking of candidates,
- a "best mutant" or lead selection,
- any activity, kcat, or catalytic-superiority claim.

Every EvidenceCard is born `L0_uncalibrated`; the whole panel and every rationale
string is gated through ClaimGuard (`assert_report_clean` under a default,
no-wet-lab `ClaimProvenance`).

## Why a probe panel, not a ranking (the V6 decision)

Two independent lines force this framing:

1. **V6-3 PMF is non-converged.** The 13-window O_nuc→Pα umbrella overlapped well
   (min 0.435) but failed the half-convergence test (access drift 1.40 kcal/mol),
   and — decisively — the **in-line angle is never productive even when the
   umbrella forces the near-attack distance** (0 of 1827 near-attack-distance
   frames). Ranking on a non-converged PMF is prohibited (ROADMAP_V6 §6). The
   near-attack *distance* is accessible; the near-attack *geometry* (in-line
   angle) is the unoccupied coordinate.

2. **The banked CAR V5 sign-offs** (`docs/car_v5/CAR_V5_SIGNOFF.md`,
   `CAR_V5_EVIDENCE_PACKAGE.md`) withdrew the earlier focused ranking and
   recommended no activity plate, because catalytic geometry was method-limited.

This portfolio **supersedes the "no plate" stance only on narrow, claim-safe
grounds**: an L0 exploratory mechanism-probe panel is not a catalytic-lead plate.
It resurrects **no** ranking and makes **no** activity ordering. It is exactly the
"honest exploratory mechanism-probe panel" ROADMAP_V6 §7 calls for when the
computational evidence is non-discriminating.

## The mechanistic hypothesis the panel tests

V6-3 localized the problem to the **in-line attack angle**, not the distance. The
multipoint and deconvolution variants each **hypothesise** an anchoring change that
could reposition the 3-HP carboxylate toward an in-line α-phosphate approach — a
hypothesis to be **tested experimentally**, never predicted here. The clean
single-site P438 probes test the same pocket at minimal structural risk. The
controls separate binding from geometry and probe the scalar-vs-geometry divergence.

## Panel composition (19 variants, 24-well plate)

| Lane | n | Purpose |
|---|---|---|
| baseline control (WT) | 1 | fixes Mg placement + O_nuc→Pα reference; plate-QC anchor |
| multipoint mechanism hypothesis | 1 | `G430R;S433F;G407K` — the multipoint geometry hypothesis; anchors the deconvolution series |
| deconvolution probes | 6 | the 3 singles + 3 pairs of the multipoint set — which site(s) drive the geometry change |
| single-site probes | 6 | `P438{N,R,K,Q,S,T}` — cleanest design position, low structural risk |
| multipoint hypotheses | 2 | `Y264F;T265S;G274A`, `P438R;G407H` — alternative geometry hypotheses |
| objective-conflict / binding controls | 3 | scalar-top geometry-≈0 variants + a binding/stability-only control |

Two variants carry `assay_priority: A` (the multipoint hypothesis + its Arg430
single); priorities are **information-gain ordering for the assay queue, not an
activity ranking**.

## Evidence attached to each card

- **Structural viability** — V6-2 explicit-solvent Amber MD for the three systems
  actually run (`WT`, `G430R;S433F;G407K`, `P438N`): all retained the ligand
  (retention 1.0, energy drift 0.5–1 %). The remaining variants carry design-level
  structural priors only.
- **Reaction-geometry accommodation** — held **low for every variant** (score
  0.15): there is no converged geometry evidence to discriminate them (V5 NAC
  status = none for all; V6-3 non-converged). The card provenance carries the V6-3
  diagnosis verbatim. Differentiation across the panel is by **test purpose**, not
  by any geometry score.
- **Pose uncertainty** — high (0.75): per-mutant structure-prediction noise plus
  the non-converged PMF.
- **Experimental calibration** — empty by construction → every card stays L0.

## V6 Gate 2 decision (CAR)

CAR produces a **claim-safe 19-variant mechanism-probe experimental portfolio**
with an explicit rationale and an L0 EvidenceCard for every variant. It does **not**
produce a ranked "best mutant" — the PMF evidence is non-discriminating and ranking
is prohibited. **Higher-cost validation is recommended**: the V6-5 QM/MM-lite tier
(or a 2D enhanced-sampling escalation) should address the in-line-angle question the
1D PMF surfaced, before any activity interpretation. This satisfies ROADMAP_V6 Gate 2
("a claim-safe 16–24 candidate experimental portfolio" **or** "a clear decision that
QM/MM-lite is required before any experiment") — here, both: the portfolio is
actionable for structural/binding/geometry-evidence exploration now, and QM/MM-lite
is the recommended escalation for the reaction-core question.

## Reproduce

```bash
python scripts/build_car_v6_portfolio.py   # regenerates the CSV + evidence_cards.json (ClaimGuard-gated)
```
