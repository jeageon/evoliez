# CAR V7 — Mechanism-Ranked Portfolio (ROADMAP_V7)

**Target:** Carboxylic Acid Reductase adenylation domain (SrCAR, UniProt E5XP76, residues
1–720) engineered toward 3-hydroxypropanoate (3-HP) adenylation.
**Reaction modeled:** the A-domain half-reaction — 3-HP carboxylate-O⁻ in-line attack on the
ATP α-phosphorus (acyl-adenylate + PPi), with a bridging Mg²⁺.
**Status:** computational triage only. **No activity, kcat, or validated-lead claim is made
before wet-lab data** — every output is ClaimGuard-gated (see §5).

This is *platform foundation*, not a commercial-ready product: it produces a statistically
calibrated, mechanism-stratified experimental panel to test, not a predicted winner.

---

## 1. What V7 produces

A `<50`-variant experimental portfolio built from a **seven-axis evidence ledger** computed
for the full generated candidate universe, with **statistically calibrated evidence bands**
replacing top-N ranking, and **expensive GPU tiers reserved for decision-changing subsets**.

The seven axes (`src/evoliez/portfolio/ledger.py`):

| # | Axis | Cheap (all-variant) | Expensive (subset) |
|---|------|---------------------|--------------------|
| 1 | structural viability | ΔΔG / stability priors | Amber/OpenMM relaxation |
| 2 | evolutionary tolerance | MSA freq, conservation, ESM, subfamily | PLM (optional) |
| 3 | ligand/cofactor/metal accommodation | role-shell proximity, contacts | Boltz complex, MD retention |
| 4 | reaction-geometry access | *(deferred — needs MD)* | Amber MD NAC: O→P distance **and** angle, kept separate |
| 5 | mechanism-hypothesis consistency | generator, multipoint order, deconvolution, catalytic-residue proximity | active-state reference fit |
| 6 | uncertainty / learning value | sparse-MSA, mutation load, subfamily support | seed/engine disagreement |
| 7 | portfolio calibration / controls | lane membership | — |

Each axis carries: `score`, `effect_size`, `empirical_p`, `q_value`, `band`, `confidence`,
`provenance`, `claim_ceiling`, `tier`, `subset_level`. Missing evidence is **deferred**, never
a silent zero — a candidate is never sunk by one absent axis.

## 2. Statistical evidence bands (§4 of the roadmap)

A score is selectable only after calibration against a per-axis null:

- **Null:** a robust-Gaussian empirical null (median / 1.4826·MAD, Efron-style) fit to the
  population and resistant to the signal tail; the analytic tail p-value uses `math.erfc`
  (no scipy). A count-based empirical p (floor `1/(n+1)`) cannot clear BH multiplicity, so a
  genuine outlier could never band — the robust-normal null fixes that.
- **Bands:** `strong` q≤0.03, `significant` q≤0.05, `consensus` q≤0.10 in ≥2 axes, plus
  `exploratory` / `control` / `unresolved` / `deferred`. Benjamini–Hochberg FDR per axis,
  with a per-axis minimum effect-size gate so a tight-but-tiny effect is never oversold.
- **Subset-level vs full-population (Gate 5):** an axis computed on only a subset (e.g.
  reaction-geometry over the MD subset) is flagged `subset_level`; the report labels those
  q-values distinctly from full-population ones.
- The control-derived null is used **only** for the reaction-geometry axis and **only** from a
  large (≥30), unbiased negative set — never from score-selected weak controls (which would
  bias the null low and manufacture significance).

## 3. Multi-fidelity cascade (§5)

- **Tier 0 (all candidates):** cheap seven-axis ledger. CPU/light, deterministic.
- **Tier 1–3 (subsets):** Boltz complexes → Amber/OpenMM MD (reaction-geometry / NAC) → PMF /
  QM-MM-lite. The allocator (`allocator.py`) assigns each candidate the cheapest
  decision-changing tier and records **why** each expensive calculation was run
  (`reports/provenance/v7_tier_plan.json`). The expensive tiers never run on the full universe.
- Reaction-geometry evidence is filled for the MD subset only, and only when the MD produced a
  **valid** near-attack signal (a diffused co-substrate → deferred, not a fake 0).

## 4. How to reproduce (server)

```bash
# on the GPU server, in /mnt/data/jglee/EvoLiEZ_car
bash scripts/run_car_v7.sh preflight   # V7 unit tests + config load (no GPU)
bash scripts/run_car_v7.sh prep        # reuse cached s01-s06b from srcar_3hp_v5
bash scripts/run_car_v7.sh gen         # s07 full-universe generation (halt-on-bug gate)
bash scripts/run_car_v7.sh run         # s07->s11 (subset Boltz/MD) + build the V7 portfolio
# or, on an already-finished run:
bash scripts/run_car_v7.sh portfolio
```

Config: [`configs/car_srcar_3hp_v7.yaml`](../../configs/car_srcar_3hp_v7.yaml) — full
candidate generation (no seeded manifest), subset caps `s08b top-24 / s09 top-64 / s10 top-16`
(widened by the selection-lane union), MD reactive-geometry O→P NAC with Mg²⁺ + co-substrate
retention restraint. Reuses the candidate-independent Boltz structure / MSA / interaction model
from the finished v5 run (config `input`/`homologs`/`msa` sections are byte-identical, so the
cached artifacts stay valid; the fingerprint is patched, not purged).

Outputs: `runs/srcar_3hp_v7/reports/v7_portfolio.{html,json,csv}` +
`reports/provenance/v7_tier_plan.json`.

## 5. Claim discipline (Gate 6)

Every report goes through `io/_report_kit.page()` → ClaimGuard. Under `--strict`
(`EVOLIEZ_STRICT_CLAIMS`) any activity / kcat / "validated lead" / "catalytically superior" /
"reduced activation barrier" phrase is a hard failure. Allowed phrasing: *prioritized for
experimental testing*, *hypothesis-grade mechanism probe*, *screening-level evidence*,
*statistically supported evidence band*, *reaction-geometry access evidence*. The overall
claim ceiling is capped at `L1_screening` before wet-lab.

## 6. Results (run `srcar_3hp_v7`, 2026-07-09)

Full pipeline s01→s11 completed on the server (s01–s06b reused from v5; s07–s11 re-run).

- **Candidate universe:** 320 generated (chemistry 35 · msa 73 · ligandmpnn 12 · multipoint 200).
- **Multi-fidelity actually spent (subset-only, GPU-first):** 41 mutants got a real Boltz
  structure (s08b), 64 went to s09 docking, **38 to explicit-geometry MD (s10)** with Mg²⁺
  bridge + co-substrate retention restraint. The 48-variant panel spans the tiers it reached:
  30 cheap · 9 GPU-broad (Boltz) · 9 focused-MD.
- **Statistical bands:** **0 strong / 0 significant / 0 consensus.** This is the honest,
  claim-safe outcome — cheap evidence is uniform across these active-site variants, and the
  **dynamic O→P near-attack occupancy is 0 for WT and all 38 MD candidates** (valid,
  co-substrate-retained runs, but the in-line conformation is not sampled — consistent with
  the known rugged CAR O→P PMF / the v5 non-convergence). Reaction-geometry access therefore
  does **not** band; a short docking/MD distance is never promoted to a productive NAC on its
  own (§3 Axis 4). The reaction-geometry axis is present-but-uninformative for 19 MD
  candidates (subset-level) and deferred for the rest.
- **The panel (48 variants, ClaimGuard-clean, `claim_ceiling: L0_hypothesis`):** WT (1),
  deconvolution of 2 multipoint hypotheses (9), single-site + exploratory probes (32),
  uncertainty probes (4), design/scalar controls (2). Every variant has a lane + a claim-safe
  reason-to-test; the tier plan (`v7_tier_plan.json`) records why each expensive calculation
  ran.

**Interpretation (claim-safe):** V7 correctly refuses to manufacture a "strong lead" from
uniform/uninformative computational evidence. The deliverable is a mechanism-stratified,
calibration-ready panel: testing it lets the first wet-lab round estimate which evidence axes
enrich hits and whether the reaction-geometry signal needs a higher-fidelity tier (enhanced
sampling / QM-MM) to become discriminating for CAR. Artifacts:
`runs/srcar_3hp_v7/reports/v7_portfolio.{html,json,csv}` (pulled to `outputs/car_v7/`).
