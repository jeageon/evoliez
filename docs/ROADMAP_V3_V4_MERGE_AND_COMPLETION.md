# V3/V4 Merge Gate & Completion Plan (code-verified)

> **Meta-finding**: An external review of `feat/server-hardening` raised 8 blockers + 2 ML issues.
> After **independent code verification** (workflow, 12 agents, file:line evidence), the verdict is:
> the reviewer is largely right, but **almost every blocker is "built but not wired / not enabled /
> not curated," not "missing code."** The machinery exists; the gap is integration, config defaults,
> and curation. That makes the fixes far cheaper than the review implies.
>
> - Branch: `feat/server-hardening` (SHA `cccf087`), 210 commits ahead of `origin/main`.
> - Verification run: `verify-review-blockers` workflow (all verdicts backed by file:line).
> - Builds on: `ROADMAP_V3_MECHANISM_TRIAGE.md`, `ROADMAP_V4_GUIDED_ENSEMBLE.md`.

---

## STATUS — IMPLEMENTED 2026-07-02 (all 3 merge gates + 5 post-merge items)

All items below were implemented with tests. Full suite: **658 passed, 18 skipped, 10
failed (all PRE-EXISTING & unrelated** — rdkit missing, low disk, real-tool dry-run,
a parallel-session CIF-parser WIP; identical set with these changes stashed). 30 new
tests across 6 files, all green.

| Item | What shipped | Test |
|---|---|---|
| **Gate 1** B1+B2 | `s08_reranker.py` runs `select_multi_lane` UPSTREAM of the ml_score cut when lanes on; `selection_lanes.enabled: true` in prod + server yaml | `test_s08_multilane.py` |
| **Gate 2** B3 | `_report_kit.page()` lints every report via ClaimGuard — warn+banner by default, hard-raise in strict mode (`strict=`/`EVOLIEZ_STRICT_CLAIMS`), `claim_provenance`/`claim_allow` params | `test_claimguard_report_gate.py` |
| **Gate 3** B4 | `Config.mechanism` field (hard-gate validated at load); `s06` publishes `mechanism_spec` + runtime `geometry_terms` to ctx | `test_mechanism_wiring.py` |
| **B7** | `glycosidic_bond_cleavage` template now has real Koshland geometry (3rd mechanism with geometry) | `test_template_geometry.py` |
| **ML2** | `s08._load_labels` routes through `assay_label` provenance guard (long-format schema) + honours a `source` column in legacy CSVs | `test_ml2_label_provenance.py` |
| **B8** | `s11` emits `evidence_cards.json` + claim-clean `triage_v3.{json,md}` — EvidenceCard is now a canonical output | assertion in `test_pipeline_mock_e2e.py` |
| **B5** | `Config.reference_ensemble` flag; `Pipeline._effective_stages` inserts `s04x` after `s04` when enabled (default off) | `test_reference_ensemble_stage.py` |
| **B6** | `SeedManifest` FDH invariants gated on `target_id==fdh_nadp` (generic schema accepts non-FDH); `load_seed_manifest` path configurable (arg/`EVOLIEZ_SEED_MANIFEST`) | `test_reference_ensemble_stage.py` |

### Publication-readiness review (2026-07-02, 7-dimension + adversarial verify)

A final pre-publication review (16-agent workflow) found the pipeline **honest and
disciplined** (proxies labelled, real MD can't fabricate from mock, real backends
fail-loud) but caught **2 real regressions from the merge-gate work + a claim-scope
ceiling**. Fixed here:

- **FIXED — Gate 1 funnel collapse**: prod/server `top_for_redocking`=600/500 but my lane
  sizes (25+8+8=41) *replaced* the scalar cut → the whole downstream funnel (s08b/s09/
  focused library of 192/96) collapsed to ~41. Now the s08 ml_high lane = `top_for_redocking`
  so lanes **augment** (probes added on top), never shrink. Regression test added.
- **FIXED — Gate 2 gap**: `final_report.md` (CLI-primary deliverable) bypassed the HTML
  `page()` gate → now claim-linted in `io/report.py` (warn+banner / strict-raise).
- **FIXED — linter false-negatives**: catalytic synonyms ("more productive than WT",
  "beats WT reactivity", "more reaction-competent than WT") now caught; the offending
  captions in `md_report.py`/`paper_report.py` softened to honest NAC-screening language.

**Claim ceiling (unchanged, confirmed by review) — a paper may claim:**
- ✓ *mechanism-informed candidate triage / a prioritized wet-lab screening library with
  per-candidate evidence classes + honest attrition provenance*
- ✗ NOT activity/kcat/catalytic-improvement (NAC ΔNAC is noise-floor on a 0.0 WT baseline,
  template-seeded, no replicate/error; RBFE single-replica, no error bar; single-WT-reference
  fragility). ✗ V4 `v4_*` artifacts are SYNTHETIC hardcoded fixtures — must NOT enter a paper
  as computed data.

See "Go/no-go" verdict + full risk table in the session review.

**Deliberate follow-ups (NOT done — documented, out of local scope):**
- **ML1** multi-head prior as the *selection model* stays run/data-gated (needs a real
  anchored run's features to fit) — the scalar bottleneck is already removed by lanes.
- **B8** focused-library still ranks by `acquisition_score`; reading EvidenceCard axes
  directly is deferred to avoid regressing selection.
- **B6** the ensemble *builder* (`build_reference_ensemble_v0`) still emits hydride-specific
  distribution keys; full mechanism-term-driven distributions are a larger refactor.
- **mechanism_from_reactive_geometry** angle term is degenerate (a==b==donor) — pre-existing.

---

## 1. Verified verdict table

Legend — verdict is my independent finding, **not** the reviewer's assertion.

| ID | Reviewer claim | Verdict | Severity | What's actually true | Existing hook (why cheap) |
|---|---|---|---|---|---|
| **B1** | s08 uses scalar `ml_score` top-N; multi-lane only in s09 | **PARTIAL** | major | s08 cut CONFIRMED (`s08_reranker.py:228-229,253`). But multi-lane is ALSO at the **s08b fold queue** (`s08b:252`), not s09-only. **Real defect**: s08b/s09 both consume `redock_candidates` = the s08 scalar cut → anything cut at s08 never reaches any lane. | `select_multi_lane`/`LaneConfig` already imported & used at s08b/s09 — just move it **upstream** of the s08 cut |
| **B2** | `selection_lanes.enabled` default False; prod path scalar | **CONFIRMED** | major | `config.py:561` default `False`; **no** prod/server/default/example_fdh/smoke yaml sets it; only `example_metalloenzyme.yaml:93`. Nothing forces it on for `backend=real`. | Set `selection_lanes.enabled: true` in prod/server yaml (+ maybe flip default) |
| **B3** | ClaimGuard not enforced in report writers | **PARTIAL** | major | Overstated: `triage_report.py:119` **does** call `assert_report_clean` (text path). But **all 18 HTML report writers** (`io/`, `reports/`) have zero ClaimGuard — incl. `paper_report_v2.py:116`, `_report_kit.page()`. Gap real for the reports users actually read. | `assert_report_clean` (`claim_guard.py:280`) + working pattern in `triage_report.py:119`; lint body text before `kit.page()` |
| **B4** | MechanismSpec not integrated into Config/pipeline | **CONFIRMED** | major | No `Config.mechanism_spec` field. ctx `"mechanism"` = **legacy** `features.mechanism.annotate` (`s06:120`), NOT `mechanism.spec.MechanismSpec`. (This resolves my earlier doubt: the s06/s09 `mechanism` imports are the legacy annotator, so they are **not** real MechanismSpec wiring.) | `mechanism_from_reactive_geometry` (`spec.py:196`) lifts existing NAC config → MechanismSpec; add Config field + ctx.put |
| **B5** | `s04x_reference_ensemble` not in default pipeline | **CONFIRMED** | minor | Not in `ALL_STAGES` (`stages/__init__.py:18-32`); self-documented as optional; hard dep on frozen seed manifest. Intentional. | `ReferenceEnsembleStage` exists with run/load parity; add behind config flag |
| **B6** | V4 seed manifest / ensemble builder FDH-specific | **CONFIRMED** | minor | Worse than stated: `mut_00479`/`Q382R` enforced inside `SeedManifest._check_manifest` validator → generic schema **rejects** non-FDH manifests. Hydride keys KeyError for non-hydride. BUT a **generic** `build_reference_ensemble()` already exists in `reference_state.py` (reviewer missed it); the FDH `_v0` is legacy. | Generic builder exists; make manifest path configurable, drive term keys from MechanismSpec, drop FDH asserts from generic schema |
| **B7** | Most mechanism templates are stubs | **CONFIRMED** | minor | Exactly 6 templates; only `hydride_transfer` + `nucleophilic_acyl_substitution` have geometry. 4 (`glycosidic_bond_cleavage`, `phosphoryl_transfer`, `metal_cofactor_redox`, `proton_transfer_isomerization`) are `_stub()` with empty geometry but hard-gated reaction-state. | `_stub()` machinery works; populate `default_geometry_terms` (curation, not code) |
| **B8** | EvidenceCard not the canonical s11 output | **CONFIRMED** | minor (design) | s11 ranks by scalar `final_score`; evidence artifact = v2 `build_evidence_library` (gate-stack + Pareto). V3-D4 `EvidenceCard` lives only in the `triage_report` CLI. By-design layering. (Nuance: V4 `experimental/library_plan.py` **does** consume `EvidenceCardV4`.) | `build_evidence_card`/`_recommend` in `triage_report.py`; call from s11 evidence block |
| **ML1** | ML still single `ml_score`, not multi-head prior | **CONFIRMED** | major | `MultiHeadEvidencePrior` (`ml/evidence_prior.py`, heads: P(structural_viable)/P(reference_pose)/P(reaction_geometry_nonneg), score+confidence) genuinely exists — but **zero consumers** outside tests. s08 still single weighted-sum / single-target xgb `ml_score`. | `MultiHeadEvidencePrior.predict_row()` exists; wire heads into `cand.scores` + lane keys (fit is data/run-gated) |
| **ML2** | supervised-label guard is column-name only | **CONFIRMED** | minor | `assert_supervised_label_allowed` name-only; `is_boltz_derived('activity')=False` → a hand-added computed `activity` col passes. | Provenance guard **already exists** — `assay_label.assert_supervised_allowed` / `is_supervised_source` (source∈{wetlab,literature}); never wired. Route `_load_labels` through it |
| **S1** | real-backend hardening (guards, RMSD, OpenMM) | **CONFIRMED ✓** | none | All 5 real & correctly wired: `RealToolError`+`fail_unless_mock_allowed` (env-gated, called before mock fallback), full-atom guard, `lock_pose_to_reference`→RMSD None (no false-perfect), `_production_nsteps` (1000× fix), honest MD skips. Not superficial. | — merge-worthy as-is |
| **S2** | s09 redocking-consistency / failure-class logic | **CONFIRMED ✓** | none | Real, wired, reachable: None→0.5/0.5 (no free pass), 4 failure classes (GNINA-local/DiffDock-global/ipTM-Δ/catalytic), DiffDock-only escape = caution unless corroborated, own-frame RMSD. | — merge-worthy as-is |

**Bottom line**: hardening (S1/S2) is genuinely strong → the branch is a valid merge candidate.
The V3/V4 *framework* claim is not yet E2E: the gaps are **wiring + config + curation**, and the
required functions already exist for every major item.

---

## 2. Merge gate — 3 must-fix before `main` (all cheap; machinery exists)

### Gate 1 — kill the s08 scalar bottleneck  *(B1 + B2 + ML1 cluster)*
The single biggest scientific risk: an ML false-negative cut at s08 can never be recovered by any
downstream lane, because s08b/s09 only re-select from `redock_candidates` (the s08 cut).

- Move `select_multi_lane(candidates, LaneConfig(...))` **upstream** of the s08 cut in
  `s08_reranker.py:228-229`, so a `low_ml_control` / evolutionary / diversity / mechanism-seed lane
  survives to s08b. (Pre-validation lanes only — stability/geometry don't exist yet at s08.)
- Turn `selection_lanes.enabled: true` in `configs/prod_fdh_nadp.yaml` + `configs/server_fdh_nadp.yaml`;
  consider flipping the default for `backend=real`.
- (ML1, optional-but-aligned) write `MultiHeadEvidencePrior.predict_row()` heads into `cand.scores`
  alongside `ml_score` so lanes can key off heads, not one scalar.
- **Done when**: a low-ml but high-mechanism/evolution/diversity candidate provably survives s08 →
  reaches s08b fold queue (regression test).

### Gate 2 — enforce ClaimGuard on every report  *(B3)*
- Add an optional `claim_provenance` arg to `_report_kit.page()` (or a shared wrapper) that lints the
  assembled body text via `assert_report_clean` before returning; route all 18 HTML writers through it.
- **Done when**: injecting "activity improved" / "kcat" / "stable functional complex" into any report
  body without wet-lab provenance fails a test.

### Gate 3 — wire MechanismSpec into Config + pipeline  *(B4)*
- Add `mechanism: Optional[MechanismSpec] = None` to `Config`; in a preflight/s06 step build it
  (from config, else `mechanism_from_reactive_geometry` over the existing NAC config) and
  `ctx.put("mechanism_spec", ...)` + `ctx.put("geometry_terms", spec.to_geometry_terms())`.
- **Done when**: the FDH hydride config yields the same geometry terms as legacy NAC; a config missing
  a required reaction-state field fails config-load.

> **Do NOT merge with a "V3/V4 complete" claim.** Merge as *"server/real-backend hardening + V3
> schema modules; framework integration in progress."* (ClaimGuard should enforce this on our own docs too.)

---

## 3. Post-merge / V4 near-term (minor; wiring & curation)

- **B8**: call `build_evidence_card` per candidate in s11, emit `evidence_cards.json`, and have the
  focused-library planner read EvidenceCard axes (not `final_score`). Auto-hedge high-score/low-confidence.
- **B6**: isolate FDH seed to `examples/`/`data/`, make manifest path configurable, drive
  `ReferenceEnsemble` distribution keys off mechanism term labels; drop `mut_00479`/`Q382R` asserts
  from the generic schema.
- **B5**: add `ReferenceEnsembleStage` to the pipeline behind `reference_ensemble.enabled`.
- **B7**: populate `default_geometry_terms` for ≥1 more class (glycosidic **or** redox) so a non-FDH
  benchmark carries real geometry evidence.
- **ML2**: route `_load_labels` through `assay_label` provenance guard (require `source∈{wetlab,literature}`).

---

## 4. Strategy triage — 15 breakthrough ideas → near-term vs research

Several "strategies" **are** the verified blockers restated (the reviewer's own MVP list = §2/§3).
Deduped and mapped to existing code:

### Near-term (buildable now on existing hooks)
| Strategy | Lands on | Status |
|---|---|---|
| Off-pathway / negative-state ensemble (#2) | `ranking/negative_design.py` (exists) + `MechanismSpec.negative_states` (new field) | hook exists |
| Uncertainty typing: epistemic / mechanistic / physical (#4) | `evidence_card.py` uncertainty axis (exists, higher=worse) — subdivide `evidence` | cheap |
| Portfolio library design (#6) | `multi_lane.py` lanes → fractional portfolio + `experimental/library_plan.py` | hook exists |
| Counterfactual epistasis map (#10) | `library_plan.py` deconvolution lane — compute single/pair proxies | cheap |
| Benchmark = failure-mode coverage (#11) | `mechanism/benchmarks.py` (exists) — reframe metrics (retain-control, reject-broken, FN-rate, calibration) | cheap |
| ClaimGuard-driven claim-level targets (#13) | `claim_guard.py` already computes ceiling — add `desired_claim_level` + gap analysis | cheap |
| Post-hoc posterior reweighting (#1, G0) | `reference_state.build_reference_ensemble` + EvidenceCard reference axis | seed step |

### Research track (GPU / differentiable / new theory — separate from merge)
| Strategy | Note |
|---|---|
| Guided-Boltz phased G0→G1→G2 (#12) | already the plan in `ROADMAP_V4_GUIDED_ENSEMBLE.md`; start at G0 reweight |
| Reaction-graph DSL `ReactionGraphSpec` (#3) | extends MechanismSpec beyond distance/angle |
| Multi-fidelity Bayesian / value-of-information (#5) | `experimental/acquisition` + `library_plan` as seed |
| Reaction-state perturbation robustness (mean vs fragility) (#8) | `MechanismSpec.reaction_state_ensemble` |
| Minimum QM/MM as decision-trigger, tiered Q0–Q3 (#9) | server/GPU; oracle-for-ambiguity, not screen |
| PLM naturalness prior + mutation grammar (#7) | ties to ML plan's ESM-feature gap; **prior only, not activity** |
| Experiment rounds R0–R4 (#15) | ops/wet-lab; unlock kinetic claims only at R4 |

---

## 5. Recommended sequence

1. **Gate 1–3** (merge blockers) — all local-verifiable (`.venv-light`, regression tests, existing configs).
2. Open PR `feat/server-hardening → main` with the honest "hardening + schema, integration in progress" framing.
3. **Post-merge near-term** (B5–B8, ML2 + near-term strategies) — still mostly wiring/curation, local.
4. **Research track** — G0 posterior reweighting first (no backprop), then the rest, on the server.

The framing that unifies all of this: **EvoLiEZ's edge is not a better single score — it's a
mechanism-conditioned, claim-safe evidence engine whose modules already exist and now need to be
wired into one E2E path.** Nothing in the merge gate requires new research; it requires integration.
