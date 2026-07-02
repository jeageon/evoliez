# EvoLiEZ v3 — Mechanism-Configurable Variant Triage
**Concrete improvement plan: from a functional-state *contract* (v2) to a claim-disciplined, mechanism-configurable *triage framework* (v3)**

- **Date:** 2026-06-29 (rev. 2 — incorporates two expert review passes: §0.1 changelog)
- **Status:** PROPOSAL (this document only). Each phase is built only when explicitly requested.
- **Relationship to existing docs:**
  - **Builds directly on** [docs/ROADMAP_V2_FUNCTIONAL_STATE.md](ROADMAP_V2_FUNCTIONAL_STATE.md) — v2 is **implemented** (Phases A–E, G, H1 + multi-ligand + F-framework, 62 tests, deployed). v3 does **not** redo v2; it adds the layer v2 deliberately left for later.
  - **Extends** [docs/IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) (per-stage SOTA) and [docs/MECHANISM_AND_BENCHMARK.md](MECHANISM_AND_BENCHMARK.md) (mechanism/benchmark source).
  - **Inherits** the ML data policy ([docs/ML_DATA_POLICY.md](ML_DATA_POLICY.md)) unchanged.

---

## 0. 한국어 요약 (Executive summary, Korean)

외부 전문가 의견의 **방향은 전부 채택**한다. 두 가지를 분명히 한다.

1. **v2가 이미 구현한 것**(역할-태그 reference state, multi-lane + low-ML control, evidence-class
   라이브러리, anchored validation, `learnability.py`의 cheap/expensive leakage guard, MSA QC)은
   **다시 만들지 않는다.** v3는 *재작성*이 아니라 *delta*다.
2. v3가 실제로 **추가하는 것**은 10개 delta(§3, D1–D10)다. 대부분은 큰 연구 모듈이 아니라
   **schema / provenance / claim discipline 보강**으로, 구현 부담은 작지만 논문 방어력은 크게 올린다.

**핵심 원칙(문서 전체를 관통):** 계산 결과는 활성을 증명하지 않는다. 어떤 후보를 먼저 실험할지,
어떤 후보가 왜 위험한지, 어떤 mechanism 가정이 취약한지를 정량적으로 정리할 뿐이다. — 그리고 v3는
이 원칙을 **코드로 강제**한다(`ClaimGuard`, D3).

### 0.1 rev.2 changelog (두 차례 전문가 검토 반영)

- **Phase 순서 교정.** `ClaimGuard`를 둘로 쪼갠다 — **report linter(skeleton)는 먼저**(V3-1),
  **full claim engine은 MechanismSpec + ReferenceConfidenceCard 이후**(V3-4). 이유: full ClaimGuard와
  ReferenceConfidenceCard는 모두 `MechanismSpec.reaction_state`(substrate real/analog, redox,
  protonation, conformational state)를 **전제**로 하기 때문이다.
- **새 delta 추가:** D6 `ActiveStateReferenceEnsemble`(단일 reference가 아니라 ensemble 분포·disagreement·
  claim ceiling), D7 `SimulationSetupCard`(ligand/cofactor/metal parameterization → MD confidence),
  D8 `Lane allocator + ControlStrategy`(percent table → plate-budget quota), D9 `BenchmarkCard`,
  D10 `WetLabLabelSpec` + Active-Learning loop, 그리고 D1에 **선택적 `HostContext`**(in-vivo pH/대사체/
  발현 독성), `ReactionGeometry`의 **tolerance calibration tier(G1–G5)**, D3에 **schema-validated
  fail-safe**, MD **adaptive early-stopping**.
- **문서 수정:** V3-7 dependency 오타(`V3-1.V3-5` → `V3-1..V3-5`), V3-1 effort 분할(rename=S/low,
  ClaimGuard skeleton=S–M/medium), "no code changes" → **"no core engine changes"**.

---

## 1. Final positioning (locked)

**Retire** these phrasings everywhere (code comments, reports, paper draft):

> ML/MD-based catalytic improvement predictor · kcat/KM improvement predictor ·
> catalytic-ready mutation predictor

**Adopt** (the one defensible sentence):

> **EvoLiEZ v3 does not rewrite the v2 functional-state machinery; it adds `MechanismSpec/
> ReactionState`, `ActiveStateReferenceEnsemble`, `ReferenceConfidenceCard`, `EvidenceCard`,
> `ClaimGuard`, and an `ML Evidence Prior` so that each variant is evaluated for how *reliably*
> it can accommodate the enzyme's structural, evolutionary, and reaction-geometry conditions —
> and only claims commensurate with that reliability and its limits are permitted — making
> EvoLiEZ a mechanism-configurable enzyme-variant triage framework.**

The load-bearing word is **mechanism-configurable framework**, not **universal predictor**.
v3 supports a *set of mechanism templates* (hydride transfer, nucleophilic acyl substitution,
glycosidic-bond cleavage, phosphoryl transfer, metal/cofactor-assisted redox,
proton-transfer/isomerization) — not "one model that scores activity for any enzyme."

---

## 2. What v2 already delivers (do NOT rebuild)

Confirmed by reading the tree — the expert opinion overlaps heavily with shipped v2 code:

| Expert-opinion item | Already in v2 | Module |
|---|---|---|
| Role-tagged reference (design_ligand/cofactor/substrate/product/metal/context/TS-proxy) | ✅ | [src/evoliez/roles.py](../src/evoliez/roles.py) |
| Active-state reference + **hard gates** (curated PDB) | ✅ (binary gates) | [src/evoliez/reference_state.py](../src/evoliez/reference_state.py) |
| Generic, config-driven geometry terms (distance/angle, SMARTS) | ✅ | [src/evoliez/md/geometry_spec.py](../src/evoliez/md/geometry_spec.py) |
| Multi-lane selection incl. **low-ML control** | ✅ (opt-in) | [src/evoliez/ranking/multi_lane.py](../src/evoliez/ranking/multi_lane.py) |
| Evidence-class / Pareto library (not one scalar) | ✅ | [src/evoliez/ranking/evidence.py](../src/evoliez/ranking/evidence.py) |
| de novo pose ⟂ anchored validation kept separate | ✅ | [src/evoliez/md/anchored_build.py](../src/evoliez/md/anchored_build.py), `pose_gate.py`, `gate_stack.py` |
| MD/RBFE/GBSA as gate-stack evidence (not proof) | ✅ | `md/gate_stack.py`, s10 |
| ML cheap/expensive **leakage guard** (no training on the validator) | ✅ | [src/evoliez/ml/learnability.py](../src/evoliez/ml/learnability.py) |
| ML data policy (Boltz ≠ supervised label) | ✅ | [src/evoliez/ml/labels.py](../src/evoliez/ml/labels.py) |
| MSA QC (Neff / subfamily balance / real-vs-synthetic) | ✅ | `features/msa_qc`, s03 |

**Conclusion:** v3 is **not** a re-architecture. It is ten focused, mostly-schema additions
(§3) + a terminology pass + a benchmark restructure. Everything else is already standing.

---

## 3. The v3 deltas (D1–D10)

> Most are **schema / provenance / claim-discipline** additions — small to implement, large for
> defensibility. Heavy research is confined to D5 (ML retrain, run/data-gated). Each is
> local/mock-testable first, per the repo's local-first rule.

### D1 — `MechanismSpec` + `ReactionState` (+ optional `HostContext`)

**Gap.** `md/geometry_spec.py` is generic for *geometry* (distance/angle terms), but the
**reaction state** that makes a geometry meaningful is still implicit / FDH-instantiated
(`config.ReactiveGeometryConfig`, the NAC hardcoding). Same "donor–acceptor distance" term
means different things for an NADP+/closed/pH-7.5 enzyme vs. a resting-heme P450. Also
`geometry_spec.py:22` already flags the missing piece: **protein-atom selectors** (catalytic
His, coordinating residue) are a "planned extension" — the term schema carries the fields but
the resolver only does ligand SMARTS. v3 closes this so catalytic-residue contact terms are
first-class.

**Add.** A `MechanismSpec` wrapping the existing `GeometryTerm` list with a reaction-state
envelope, a template registry, and **geometry-tolerance calibration**. New
`src/evoliez/mechanism/{spec.py,templates/}`.

```yaml
mechanism_spec:
  schema_version: 1.0
  reaction:
    class: hydride_transfer
    biological_objective: { primary: "NADPH formation", secondary: [NADP_specificity, formate_use, stability] }
  reaction_state:                       # required block — abort at config-load if a template-required field is absent
    pH: 7.5
    cofactor_redox_state: "NADP+"       # NADP+ vs NADPH changes the productive geometry
    substrate_state: "formate"
    substrate_is_real: true             # real substrate vs analog/proxy -> feeds ReferenceConfidenceCard + ClaimGuard
    protonation_model: "propka+manual_active_site_review"
    metal_state: "none"                 # identity + coordination number when present
    tautomer_state: "default"
    covalent_intermediate: false
    conformational_state: "closed_ternary_complex"
    required_waters: []
  host_context:                         # OPTIONAL — only for in-vivo / metabolic-pathway targets (E. coli etc.)
    enabled: false
    physiological_pH_profile: [6.8, 7.4]   # a range/profile, not a single value
    metabolite_inhibition: []              # competing metabolites + approx concentrations
    expression_toxicity_risk: unknown      # informs a structural-viability sub-axis, never a catalytic claim
  catalytic_residues:
    - { residue: "ARG290", role: formate_carboxylate_stabilizer, required: true }
    - { residue: "HIS338", role: proton_relay_or_pocket_polarizer, required: false }
  geometry_terms:                       # same schema as md/geometry_spec.GeometryTerm, now mechanism-scoped
    - { kind: distance, label: C4_to_formate_C, a_smarts: "...", b_smarts: "..." }
    - { kind: angle,    label: hydride_axis,    atoms: [...] }
  geometry_calibration:                 # where the soft-kernel tolerances come from (reviewer-defensible)
    source_tier: G2                     # G1 WT+known-active empirical | G2 homolog ensemble | G3 literature default | G4 chemistry heuristic | G5 uncalibrated
    distance_kernel: { type: gaussian,        center_source: reference_ensemble_median, sigma_source: reference_ensemble_iqr }
    angle_kernel:    { type: von_mises_or_gaussian, center_source: reference_ensemble_median, sigma_source: mechanism_template_default }
    known_active_controls_used: true
    known_inactive_controls_used: false
    claim_limit: screening_only
```

**Contract.** `reaction.class` selects a **template** (default geometry terms + which
reaction-state fields are required); the per-target YAML overrides specifics. Template-required
`reaction_state` fields are **required, not optional** — a missing redox/protonation/
conformational state for a template that needs it **aborts at config-load**, mirroring the v2
reference hard-gate discipline. `host_context` is opt-in (`enabled: false` default) so a pure
in-vitro target is never forced to fill it; when enabled it adds a *structural-viability
sub-axis*, never a catalytic claim. `geometry_calibration.source_tier` (G1–G5) **caps the
claim strength** of any reaction-geometry evidence derived from those tolerances.

**Touch.** new `src/evoliez/mechanism/{spec.py,templates/}`; extend
`geometry_spec._resolve_global` with a protein-atom resolver (residue+atom-name → trajectory
index, using the s10 topology); `config.ReactiveGeometryConfig` becomes a view onto a
`MechanismSpec` (FDH config keeps working via the `hydride_transfer` template). Local-testable.

**Acceptance.** FDH NAC is reproduced *from the `hydride_transfer` template + config*, with
**zero FDH-specific code** in the geometry path; a second template
(`nucleophilic_acyl_substitution` for TEM-1) is declarable in config alone; a catalytic-residue
contact term resolves to a protein atom and scores occupancy; a tolerance drawn from a G4
heuristic is auto-capped to `hypothesis_grade` claim strength.

---

### D2 — `ReferenceConfidenceCard` (graded, claim-limiting reference quality)

**Gap.** `reference_state.validate_reference` returns a **binary** `ReferenceGates`
(pass/fail). A reference can *pass* the hard gates yet still be weak (homolog at 54% identity,
analog instead of real substrate, no known-inactive control) — and the downstream claim must
then be weaker regardless of how good the scores are.

**Add.** A `ReferenceConfidenceCard` produced alongside `ReferenceGates` (extends, does not
replace — hard gates still abort on failure; the card *grades* what survives):

```yaml
reference_confidence:
  tier: B                              # A experimental ternary | B homolog ternary | C pose-transfer+constrained dock | D de-novo only
  source: homolog_ternary_complex
  resolution_A: 1.8
  sequence_identity_to_target: 54.2
  ligand_state: NADP_plus_and_formate_analog
  substrate_is_real: false             # from MechanismSpec.reaction_state
  analog_identity: formate_proxy
  catalytic_residue_alignment_verified: true
  functional_atom_mapping_verified: true
  protonation_state_assigned: true
  redox_state_assigned: true
  active_closed_state_supported: partial
  known_active_controls_available: true
  known_inactive_controls_available: false
  claim_strength: strong_screening_only   # DERIVED -> consumed by ClaimGuard (D3)
```

**Contract.** `claim_strength ∈ {strong_screening, moderate_screening, hypothesis_grade,
uncalibrated}` is **derived** from the card (tier + analog + control availability) and is a
**primary input ClaimGuard reads** to cap report language.

**Touch.** extend `reference_state.py` (new dataclass + `grade_reference()` next to
`validate_reference()`); persist into run provenance.

**Acceptance.** A tier-C reference run produces identical *numbers* but a report whose
mechanistic claims are automatically downgraded to "hypothesis-grade" (verified by D3 tests).

---

### D3 — `ClaimGuard` (claim-category + template-whitelist engine, schema-validated) — **the keystone**

**Gap.** None of this exists. v2 has a *paper-grade gate* (`ranking/evidence.is_paper_grade`)
deciding whether a candidate may carry a paper-grade claim, but **no engine constrains the
claim *language* itself**.

**Add.** New module `src/evoliez/ranking/claim_guard.py`. Two reviewer-driven hardenings over a
naive token blacklist:

1. **Claim *category* + allowed-template whitelist, not token blacklist.** A bare token list
   both over-blocks ("This workflow does **not** predict kcat." is good, should pass) and
   under-blocks ("The variant is catalytically superior." over-claims with no banned token).
   So reports assemble sentences **only from an allowed-template library** per claim category:

   ```yaml
   claim_guard:
     activity_improvement:
       status: prohibited_without_wetlab
       prohibited_intents: ["variant improves activity", "catalytically superior", "enhanced catalysis"]
       allowed_templates:
         - "This variant is prioritized for experimental testing."
         - "This variant has screening-level evidence for structural and ligand/cofactor competence."
     kinetic_parameter_prediction:
       status: prohibited_without_wetlab_or_qmmm
       allowed_templates: ["The workflow does not directly predict kcat or kcat/KM."]
     short_md_interpretation:
       status: restricted
       allowed_templates: ["Short MD supports local reference-pose accommodation."]
       prohibited_templates: ["Short MD validates long-term ligand stability.", "Short MD confirms a functional catalytic state."]
   ```

2. **Schema-validated, fail-safe input (reviewer B's point — critical).** A table-driven rule
   engine **silently fails** if a provenance key is renamed or mistyped, leaking a prohibited
   claim. So validate the input provenance (`ReferenceConfidenceCard`, `EvidenceCard`,
   MD-tier, controls) with a strict schema (Pydantic) **before** the rules run; on any
   missing/mistyped field, **fail safe** — default to the most conservative
   (`hypothesis_grade` / `uncalibrated`) claim strength, never the permissive branch.

**Contract.** Reports **cannot** emit a prohibited token *or* a non-whitelisted sentence in a
restricted category; a violation is a **test failure**, not a warning. Missing provenance →
most-conservative claim, by construction.

**Touch.** new `ranking/claim_guard.py` + a Pydantic provenance schema; wire into the report
builder (paper_report_v2) and s11. Fully local-testable.

**Acceptance + required tests:**

| Test | Expectation |
|---|---|
| positive prohibited | "activity improved" in report → **fail** |
| negated-safe | "does not predict kcat" → **allow** |
| synonym | "catalytically superior" → **fail** |
| Korean | "활성 증가", "촉매 효율 향상" → **fail** |
| LaTeX/math | `$k_{cat}/K_M$ improved` → **fail** |
| template whitelist | restricted-category sentences come only from the allowed set |
| **fail-safe** | missing `reference_confidence.claim_strength` key → claim auto-downgraded to `hypothesis_grade`, not crash/leak |

Given a synthetic provenance with `reference_tier=C`, `only_short_md=true`,
`no_known_controls=true`: report contains "hypothesis-grade", "local accommodation",
"uncalibrated" and **none** of "kcat", "activity improved", "stable functional complex",
"inactive mutant".

---

### D4 — `EvidenceCard`: separate **score** from **confidence**, with derivation rules

**Gap.** `ranking/evidence.py` groups by gate-stack *verdict class* (good) but per-axis signal
is largely scalar, a few binary labels survive (`reference_like`/`displaced`), and there's no
**rule** for how "confidence" is assigned (human annotation isn't reproducible).

**Add.** A per-variant `EvidenceCard` (extends the verdict library — verdict = summary, card =
breakdown). Every axis carries `score`, `confidence`, `evidence[]`, **and a reproducible
`confidence_model`**:

```yaml
variant_evidence:
  variant_id: A123G
  reaction_geometry_accommodation:
    score: 0.82
    confidence: low                     # score HIGH yet confidence LOW is allowed and meaningful
    reason: "score derived from Tier-C reference with no known active/inactive controls"
    evidence: [soft_kernel_partial, constrained_min_violation_moderate]
  uncertainty: { score: 0.72, confidence: high, evidence: [de_novo_pose_disagreement, reference_tier_C, no_known_inactive_control] }
  # ... structural_viability / evolutionary_tolerance / ligand_cofactor_competence / substrate_positioning
  confidence_model:
    method: rule_based_v1
    components: { reference_confidence: 0.55, validator_agreement: 0.62, control_calibration: 0.30, data_quality: 0.70 }
    final_confidence: low_to_medium
```

**Confidence-derivation rules (standardized, per axis):**

| Axis | lowers confidence | raises confidence |
|---|---|---|
| structural_viability | stability-model disagreement, buried-polar increase | multi-model consensus, motif intact |
| evolutionary_tolerance | low Neff, subfamily imbalance, synthetic-seq excess | high Neff, family-balanced MSA |
| ligand_cofactor_competence | weak reference tier, ligand-param uncertainty | conserved anchor contacts, ensemble agreement |
| substrate_positioning | small-ligand ambiguity, analog-only reference | real-substrate reference, pose consensus |
| reaction_geometry_accommodation | WT geometry sparse, high ensemble variance | WT/known-active controls reproduce expected geometry |
| uncertainty | de-novo/docking disagreement, OOD mutation | validators agree, controls calibrated |

`uncertainty.score` is "higher = worse"; **score and confidence are always separate fields.**
The reaction-geometry kernel generalizes FDH NAC: `S_geom = K_d · K_θ · K_c · K_s` — soft
kernels over the **reference-ensemble distribution** (D6), never a binary cutoff (this is the
`spec_satisfaction` shape in `geometry_spec.py:141`, surfaced per-axis with a confidence).

**Touch.** extend `ranking/evidence.py`; map existing per-axis numbers + attach
provenance-derived confidence; rename surviving binaries per §4.

**Acceptance.** A candidate can read "high structural_viability, **score-high/confidence-low**
reaction_geometry_accommodation, high uncertainty" simultaneously; no axis is a bare boolean.

---

### D5 — ML = **Evidence Prior** (multi-head, per-head label provenance, split-validated)

**Gap.** `learnability.py` already enforces the cheap/expensive split and `assert_no_leakage`
(the hardest part — done). Remaining: naming/claim discipline; **multi-head outputs**;
**per-head label provenance**; **target/family/enzyme-class split** validation.

**Add.** Keep the contact-fingerprint model as a *demoted auxiliary feature*. Train a
multi-head Evidence Prior on the cheap block; **each head carries its label provenance**:

```yaml
heads:
  - name: P(structural_viable)
  - name: P(ligand_cofactor_competent)
  - name: P(reference_pose_accommodation)
    label_source: { type: computational_surrogate, md_used_as_label: true, experimental_label: false }
    allowed_claim: surrogate evidence prior
    prohibited_claim: experimental activity predictor
  - name: P(nonMD_validation_pass)      # learns a computational validator — useful, NOT an activity predictor
  - name: uncertainty/OOD
# NEVER a head: P(kcat improved) · P(kcat/KM) · transition-state barrier · wet-lab activity
generalization_claim_policy:
  random_split:            "internal interpolation only"
  target_split:            "target-level transfer estimate"
  family_split:            "family-level generalization estimate"
  enzyme_class_split:      "mechanism-template transfer estimate"
  fewer_than_3_classes:    PROHIBIT "broad enzyme-class generalization"
```

Report **top-k enrichment + false-negative rate from the low-ML control lane**, not a single
AUC. With Stage-1 = FDH + TEM-1 + glycosidase (3 classes), enzyme-class split has low power →
claim only "initial cross-mechanism stress test", not broad generalization.

**Touch.** extend `learnability.py` into the multi-head eval; enforce splits in
`ml/datasets.py`; rename in reports per §4. **Run/data-gated** (needs a real anchored run's
features — same gating as v2 Phase F).

**Acceptance.** No head predicts activity/kcat; the leakage assertion still blocks any
MD/pose-agreement feature from training; random-split results are never reported as broad
generalization.

---

### D6 — `ActiveStateReferenceEnsemble` (ensemble distribution + disagreement + claim ceiling)

**Gap.** D4's soft kernel is defined *against a reference-ensemble distribution*, but only a
single-reference `ReferenceConfidenceCard` exists. Open questions the card alone can't answer:
which value scores when 3 references disagree? a Tier-A single vs. Tier-C ×3 conflict? how is
`reaction_geometry_accommodation` confidence computed when formate pose varies per reference? a
mutant that passes one reference and fails another?

**Add.** A first-class ensemble object that owns the distribution the kernels read:

```yaml
active_state_reference_ensemble:
  ensemble_id: fdh_active_state_v1
  mechanism_spec_id: fdh_hydride_transfer_v1
  references:
    - { reference_id: ref_A1, tier: A, weight: 1.0, source: experimental_ternary_or_TS_analog, confidence_card: {...} }
    - { reference_id: ref_B1, tier: B, weight: 0.7, source: homolog_ternary_transfer,           confidence_card: {...} }
    - { reference_id: ref_C1, tier: C, weight: 0.4, source: constrained_docking,                 confidence_card: {...} }
  ensemble_geometry_distribution:
    distance_terms: { C4_to_formate_C: { median: ..., iqr: ..., source_references: [ref_A1, ref_B1, ref_C1] } }
    angle_terms:    { hydride_axis:    { median: ..., iqr: ... } }
  ensemble_disagreement: { geometry_variance: ..., ligand_pose_variance: ..., catalytic_contact_variance: ... }
  claim_ceiling: { mechanistic_claim_strength: moderate_screening }
```

**Contract.** The kernel centers/sigmas in `geometry_calibration` (D1) read from
`ensemble_geometry_distribution`; `ensemble_disagreement` lowers `reaction_geometry_
accommodation.confidence` (D4) and raises `uncertainty`; `claim_ceiling` is an upper bound
ClaimGuard (D3) cannot exceed regardless of per-candidate scores. A per-reference pass/fail
split is recorded in the EvidenceCard, not collapsed.

**Touch.** new dataclass beside `reference_state.py`; consumed by D4 kernels + D3 ceiling.
Local-testable on synthetic reference sets.

**Acceptance.** With 3 disagreeing references, a candidate's geometry score reads against the
ensemble median/IQR; high `geometry_variance` measurably lowers confidence and caps the claim.

---

### D7 — `SimulationSetupCard` (parameterization → MD-evidence confidence)

**Gap.** ReactionState is covered, but MD/docking confidence hinges on **ligand/cofactor/metal
parameterization** (NADP redox, formate protonation, heme, metal coordination, covalent
intermediate, water model, FF version, restraint policy). Without it, MD-evidence confidence
can't be assigned honestly.

**Add.** A provenance card emitted by s10 setup:

```yaml
simulation_setup_card:
  force_field: { protein: amber14sb, water: tip3p, ligand_param_source: openff|gaff|manual }
  ligand_parameters:
    NADP:    { redox_state: NADP+, charge_model: ..., parameter_validated: true }
    formate: { protonation_state: anion, parameter_validated: true }
  restraints: { distance_restraints: true, angle_restraints: false, note: "angle terms are read-only occupancy measures" }
  md_protocol: { solvent: explicit, length_ns: 2.0, temperature_K: 300 }
  confidence_impact: { ligand_param_confidence: medium, metal_param_confidence: not_applicable }
```

**Contract.** `confidence_impact.*` feeds the D4 `ligand_cofactor_competence` /
`reaction_geometry_accommodation` confidence; a failed/uncertain parameterization **downgrades
confidence** (it does not silently pass). Reinforces the v2 invariant *distance-only restraints,
never angle*.

**Touch.** emit from s10 setup into provenance; read by D4. Acceptance: an unvalidated ligand
parameter visibly lowers the MD-evidence confidence in the report.

---

### D8 — Lane allocator + ControlStrategy (plate budget, not percentages)

**Gap.** v2 `multi_lane.py` + the §6 percentage table can sum past 96. Need a **quota
allocator** that reserves controls first and never overflows; and "known controls mandatory"
is wrong for novel enzymes lacking literature.

**Add.** A budget allocator + a *control strategy* (not a hard known-control requirement):

```yaml
lane_allocator:
  budget: 96
  reserve_controls_first: true
  deduplicate_variants: true
  collision_policy: assign_primary_lane_by_priority_but_record_all_lanes   # record all_lanes, not just one
  diversity_metric: { position_distance: true, physicochemical_class_distance: true, sequence_distance: true }
  overflow_policy: { rank_within_lane_by: [confidence_adjusted_score, lower_uncertainty, diversity_gain] }
```

Recommended 96-well allocation: WT replicates 4–8 · known-active 2–6 · known-inactive 2–6 ·
negative-computational-controls 6–8 · uncertainty-probes 6–8 · consensus-high 18–24 ·
mechanism-geometry-high 12–16 · cofactor/ligand-specific 8–12 · structural/evolutionary-high
8–12 · diversity-fill = remainder.

**Control strategy (mandatory; known controls *when available*):**

| Control | Priority | Example |
|---|---|---|
| WT | required | per-plate replicate |
| known active | when available | literature active mutant |
| known inactive | when available | catalytic-residue knockout |
| mechanism-negative | strongly recommended | catalytic-residue / anchor-destroying mutant |
| stable low-score | required | ML false-negative probe |
| remote neutral | recommended | conservative mutation far from the site |

A target with no known active/inactive proceeds but is flagged **uncalibrated** (D3).

**Touch.** wrap `multi_lane.py` selection in the allocator; record `all_lanes` per candidate.
Acceptance: allocation always ≤ budget; lane purpose preserved; multi-lane candidates record
every lane that surfaced them.

---

### D9 — `BenchmarkCard` (label meaning + allowed claims per benchmark)

**Gap.** Benchmark labels differ in kind (antibiotic fitness ≠ kcat). Without per-benchmark
constraint, cross-benchmark claims drift.

**Add.** One card per benchmark target:

```yaml
benchmark_card:
  target_id: TEM1_beta_lactamase
  mechanism_template: nucleophilic_acyl_substitution
  label_type: { primary: antibiotic_fitness, secondary: known_activity_or_resistance, is_direct_kcat: false }
  label_quality: { replicate_available: true, quantitative: true, assay_context: "E_coli_growth_under_antibiotic" }
  reference: { available_tiers: [A, B], substrate_or_analog: beta_lactam_or_inhibitor }
  controls: { wt_available: true, known_active_available: true, known_inactive_available: true }
  allowed_claims: ["known-active retention", "known-inactive rejection", "top-k enrichment against benchmark label"]
  prohibited_claims: ["direct kcat prediction"]
```

**Contract.** ClaimGuard reads `prohibited_claims`/`allowed_claims` per benchmark; metrics stay
known-active retention, known-inactive rejection, top-k enrichment, FN rate, uncertainty
calibration — **not** a single AUC. Local-testable.

---

### D10 — `WetLabLabelSpec` + Active-Learning loop, and adaptive MD early-stopping

**Gap.** Wet-lab is the only real label (Tier 5), yet there's no import schema and no feedback
narrative; and Multi-lane sends more candidates into MD, wasting GPU on doomed runs.

**Add (a) — wet-lab schema + AL loop (the platform's long-term payoff):**

```yaml
wetlab_label_spec:
  assay_id: fdh_nadph_340nm
  endpoint: { primary: initial_rate, units: "umol_min_mg or s^-1" }
  substrates: { formate_mM: [10, 50, 100], NADP_mM: [0.1, 0.5, 1.0] }
  counter_screen: { NAD_plus: true }
  normalization: { expression_normalized: true, protein_concentration_method: Bradford_or_purified }
  replicates: { biological: 3, technical: 2 }
  controls: { wt_replicates: 6, blank: true, no_substrate: true, no_cofactor: true }
  derived_labels: [active_above_wt, NADP_specificity_ratio, stability_retained]
  claim_permission: { activity_claim_allowed_only_if_replicated: true }
```

When wet-lab data arrives, the AL loop (i) updates the multi-head prior weights, (ii)
recalibrates the target's geometry soft-kernel against measured actives, (iii) lets ClaimGuard
*unlock* enrichment/activity claims **only** for replicated endpoints. Round 0 computational →
Round 1 multi-lane assay → Round 2 calibrated evidence-to-activity → Round 3 AL-guided library.

**Add (b) — adaptive MD early-stopping (reviewer B):** in `md/gate_stack.py` (or the s10 MD
loop), if within the first ~100–200 ps the ligand fully leaves the pocket or the catalytic
contact network collapses irrecoverably, **stop the simulation** and pin
`structural_viability`/`uncertainty` to worst — don't spend the full 2 ns on a doomed
candidate. Pure efficiency; honest (a recorded early-failure verdict, not a silent skip).

**Touch.** `WetLabLabelSpec` + AL hook are schema/provenance (cheap, build early so imports are
ready); early-stopping is a small s10/gate_stack rule (low risk, lands anytime).

**Acceptance.** Wet-lab results store experimental labels **separately** from computational
evidence (never mixed); an early-leaving ligand terminates its MD early with a worst-case,
recorded verdict.

---

## 4. Terminology refactor (Phase V3-1 — pure rename + report pass)

Rename across comments, provenance keys (with back-compat **aliases**), and reports. Keep old
keys as aliases so the restored fdh_5track provenance still parses.

| Current | v3 |
|---|---|
| FDH predictor | mechanism-configurable enzyme-variant triage framework |
| functional / nonfunctional label | evidence label (per-axis card) |
| `reference_like` | `reference_pose_accommodation` |
| `displaced` | `pose_uncertainty` |
| NAC score / NAC fraction | `reaction_geometry_accommodation` |
| anchored MD validation | reference-pose accommodation under short local relaxation |
| de novo displacement | alternative-pose uncertainty signal |
| ML catalytic classifier / `ml_score` (as decider) | ML **evidence prior** |
| final prediction / final rank | triage recommendation |

---

## 5. MD / docking / Boltz interpretation tiers (locked vocabulary)

| Tier | Name | Compute | Interpretation |
|---|---|---|---|
| 0 | Static screen | MSA, PSSM, stability, clash, pocket distance | cheap prior |
| 1 | Reference accommodation | mutant local relaxation in the active-state reference | reference-pose acceptability |
| 2 | Pose uncertainty | docking / de-novo ensemble variance | alternative-pose risk |
| 3 | Short MD retention | short explicit-solvent MD (+ adaptive early-stop, D10) | immediate-failure check only |
| 4 | Enhanced sampling / QM-MM | metadynamics, umbrella, EVB, QM/MM | mechanistic support |
| 5 | Wet-lab assay | activity, specificity, stability | the only real label |

ThermoMPNN stays **structural-viability evidence only** — never promoted to a catalytic score.

---

## 6. Disposition policy (shared by ClaimGuard + EvidenceCard)

Not every problem aborts (kills usability); not every problem warns (leaks over-claims). The
shared table:

| Condition | Disposition | Why |
|---|---|---|
| required functional atom missing | **hard abort** | geometry uncomputable |
| catalytic-residue mapping fails | **hard abort** *or* axis disabled | mechanism score impossible |
| redox/protonation state missing | **hard abort** (redox enzymes) | geometry meaning changes |
| reference tier C/D | continue + **claim downgrade** | screening still valid |
| no known controls | continue + **uncalibrated flag** | novel enzyme OK |
| WT reaction geometry sparse | continue + **diagnostic-only flag** | reference/sampling issue likely |
| ligand parameterization uncertain | continue + **confidence downgrade** | MD evidence weakened |
| de-novo pose disagreement high | continue + **uncertainty high** | must not declare inactive |
| ML OOD high | continue + **ML confidence low** | don't discard the candidate |

This table is the single source both `claim_guard.py` and the `EvidenceCard` builder import.

---

## 7. Report output — triage recommendation card (not a scalar rank)

Each candidate is described by **why selected · what's risky · what to validate next** (also
better for active learning):

```yaml
triage_recommendation:
  variant_id: A123G
  recommendation: test_in_round1
  assigned_lanes: [consensus_high, cofactor_ligand_specific]
  primary_reason: ["high structural viability", "NADP anchor contacts retained"]
  primary_risk:   ["reaction-geometry confidence low (Tier-C reference)"]
  claim_level:    ["screening evidence only"]
  next_validation: ["wet-lab activity assay", "NADP/NAD specificity counter-screen"]
```

---

## 8. Benchmark strategy (staged, realistic)

**Stage 1: FDH + 2 non-FDH.**

| Benchmark | Mechanism | Why | Caveat |
|---|---|---|---|
| NADP-dependent FDH | hydride transfer | current target | reference state / NAC sensitive |
| TEM-1 β-lactamase | nucleophilic acyl substitution | rich DMS/fitness landscape, known active/inactive | antibiotic fitness ≠ pure kcat |
| Glycosidase **or** P450 BM3 | glycosidic cleavage **or** heme oxidation | different mechanism class | P450 redox/intermediate state hard → **ReactionState stress test only**, not a strong mechanistic claim |

Pick **FDH + TEM-1 + glycosidase** for paper speed; **FDH + TEM-1 + P450 BM3** for an
oxidoreductase/industrial image (slower, harder reaction-state). Each benchmark carries a
`BenchmarkCard` (D9).

**Realism note (corrects "no code changes").** Adding TEM-1/glycosidase needs the
`nucleophilic_acyl_substitution` template, covalent/acyl-enzyme proxy handling, Ser/Lys/Glu
acid-base residue selectors, analog-reference curation, DMS-vs-activity label separation, and
known active/inactive mapping. The honest target is: **no *core engine* changes after V3-2;
benchmark-specific templates, references, controls, and label adapters still required.**

**Stage 2 (revision): 4–6 classes.** **Per-mechanism metrics (NOT a single AUC):** known-active
retention, known-inactive rejection, top-k enrichment, FN rate, uncertainty calibration,
diversity coverage, reference-tier score degradation, (with wet-lab) hit-rate improvement.

---

## 9. Phased roadmap (v3, rev.2)

> Each phase: independently shippable, mock/local-testable first, user-gated. Most are cheap
> because the heavy contract (v2) is already standing.

| Phase | Scope | Deltas | Depends | Effort/Risk | Gated? |
|---|---|---|---|---|---|
| **V3-0 Schema contract freeze** | all card schemas + invalid-example tests | D1–D10 schemas | — | S / low | local |
| **V3-1a Terminology rename** | §4 rename + aliases | — | — | **S / low** | local |
| **V3-1b ClaimGuard skeleton (linter)** | report linter: forbidden-phrase assertions | D3 (partial) | V3-1a | **S–M / medium** | local |
| **V3-2 MechanismSpec + ReactionState** | template registry + protein-atom resolver + geometry calibration + opt. HostContext | **D1** | V3-0 | M / medium | local |
| **V3-3 Ensemble + ReferenceConfidenceCard** | ensemble distribution/disagreement + tier→claim_strength | **D2, D6** | V3-2 | M / medium | local |
| **V3-4 Full ClaimGuard** | claim-category + template whitelist + schema-validated fail-safe + disposition table | **D3** | V3-1b, V3-3 | M / medium | local |
| **V3-5 EvidenceCard** | score/confidence split + derivation rules + SimulationSetupCard | **D4, D7** | V3-3, V3-4 | M / medium | local |
| **V3-6 Lane allocator + ControlStrategy** | plate-budget quota + all_lanes + control strategy | **D8** | V3-2, V3-5 | S–M / low | local |
| **V3-7 ML Evidence Prior** | multi-head + per-head label provenance + split policy | **D5** | V3-2, V3-5 + a real anchored run | L / research | **run/data-gated** |
| **V3-8 Benchmark Stage 1** | TEM-1 + glycosidase configs + templates + BenchmarkCard | **D9** | V3-1..V3-6 | M / medium | **server-gated** |
| **V3-9 Wet-lab import + Active Learning + MD early-stop** | WetLabLabelSpec + AL recalibration hook + adaptive early-stop | **D10** | V3-5 (schema early); AL gated on wet-lab | M / medium | partly data-gated |

**Recommended first slice (smallest valuable, local-only, no server time):**
**V3-0 → V3-1a → V3-1b → V3-2.** The schema freeze + rename + ClaimGuard *linter* + MechanismSpec
make every report honest and remove the last FDH hardcoding without the busy server. Full
ClaimGuard (V3-4) follows once the reference card (V3-3) exists to feed it. (MD early-stopping,
part of D10, is low-risk and can land alongside anytime.)

```
V3-0 ──> V3-1a ──> V3-1b ──┐
              V3-2 ──> V3-3 ──> V3-4 ──> V3-5 ──┬─> V3-6 ──> V3-8
                                                 └─> V3-7 (ML, run-gated)
                                       V3-5 ──> V3-9 (schema early; AL wet-lab-gated)
```

---

## 10. Invariants carried from v2 (do not break)

- ML data policy: Boltz outputs are features/weights/weak signals — **never** supervised
  experimental labels (`ml/labels.py`, enforced).
- Restraints stay **distance-only, never angle**. D1/D4 *angle geometry terms* are **read-only
  occupancy measures**, not restraints — keep that line (D7 records it explicitly).
- Mock/real backend separation; honest-degrade (no silent mock fallback); purge-safety guard.
- GPU-first, CPU-bounded (≤16–24 cores, watchdog-safe); local-first verification before any
  server run.
- Pipeline/stage names unchanged; v3 changes the **contract + claims inside** each stage, plus
  new pure modules (`mechanism/spec.py`, `ranking/claim_guard.py`) and dataclass extensions.
- **ClaimGuard fail-safe (new invariant):** any missing/malformed claim-provenance defaults to
  the most conservative claim strength — a guard must never fail open.

---

## 11. Paper framing (matches ClaimGuard output)

**No wet-lab yet** → workflow/validation-framework paper:

> *A Mechanistically Informed Triage Framework for Enzyme-Variant Prioritization Reveals Label
> Instability in Catalytic-State Prediction.* — we do **not** claim activity prediction; we
> expose the collapse of computational functional labels, show anchored-MD vs de-novo-pose
> failure modes differ, propose `MechanismSpec` / active-state ensemble / evidence labels /
> ClaimGuard; validated on FDH + 2 benchmarks for triage consistency.

**With wet-lab** → enrichment paper (claims bounded to hit-rate / enrichment, **never** kcat):

> *Mechanism-Guided Variant Triage Enables Efficient Prioritization of NADP-Dependent FDH
> Mutants.* — allowed: "enriched experimentally active variants over random/stability-only
> baselines", "reduced the experimental search space while preserving mechanistic diversity".
> Prohibited (by ClaimGuard): "predicts kcat/KM", "catalytic-ready states from ML alone",
> "short MD validates improved catalysis."

---

## 12. The one design principle (governs every phase)

> **Computational results do not prove activity. They quantify which candidate to test first,
> which candidate is risky and why, and which mechanistic assumption is fragile.**

Holding this line is what lets EvoLiEZ outlive a single-enzyme predictor — and v3 makes it
**enforced in code** (ClaimGuard, D3, fail-safe by construction), not left to the author's
discipline.
