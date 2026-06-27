# EvoLiEZ v2 — Functional-State Engineering Platform
**Strategy & phased roadmap for a major architectural overhaul**

- **Version mark:** **v2.0 epoch** (target `__version__` `0.2.0`; current `0.1.0`)
- **Status:** **STRATEGY / DESIGN ONLY.** No code is changed by this document. Each
  phase is implemented only on explicit, per-phase user request.
- **Date:** 2026-06-28
- **Relationship to existing docs:**
  - **Supersedes the *internal contract*** described in [docs/ARCHITECTURE.md](ARCHITECTURE.md)
    (the `sequence + ligand → … → final rank` flow).
  - **Extends** [docs/IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md), which remains the
    per-stage SOTA/citation source (the *what model* layer). This file is the
    *what contract* layer.
  - **Builds on** the landed 6-principle anchored-validation redesign
    (see §3 / [memory anchored-validation-redesign]) — v2 does **not** redo it; it
    promotes it from an s10-only, opt-in capability to the pipeline's default contract.

---

## 0. Thesis — from "FDH funnel" to "functional-state platform"

EvoLiEZ today is a **working enzyme/ligand screening funnel** — it runs end-to-end,
degrades honestly, and produces a ranked library. But the design center of gravity is
**one primary ligand** and **one scalar rank**. If the goal is a *general* protein/
enzyme **engineering** platform (any reaction, any cofactor/substrate/metal, any
functional state) rather than an FDH-tuned screener, the central axis must change.

**v2 thesis:** the unit of truth is not *a ligand and a score* but **a functional
reference state and an evidence stack**. Every stage should ask *"is the engineered
variant's functional state preserved-and-improved relative to a trusted reference?"*
— not *"did a fresh per-mutant pose score well?"*

---

## 1. Root diagnosis — three structural problems (+ one verified corollary)

**P1 — Primary-ligand-centric structure.**
`Complex.ligand` is load-bearing; the core graph/ranking in s06/s07/s08 is built on
`cx.ligand.atoms` (see `src/evoliez/stages/s06_interaction_graph.py:33`). `extra_ligands`
reach s04/s05/s08b/s10 on *some* paths but are **not first-class functional
objectives**. In FDH this surfaced as NADP-centric candidates with formate relegated to
a downstream NAC check, not an equal functional partner.

**P2 — Trusting the Boltz mutant pose too early.**
s08b re-predicts the top-N mutant complexes de novo with Boltz. s09 then, when a real
mutant Boltz exists, scores redocking RMSD against the **candidate's own Boltz ligand
pose**, not the WT reference (`src/evoliez/stages/s09_nonmd_validation.py:203`). This is
a *reasonable* fix for cross-frame RMSD false-fails — but it means **"WT-like functional
pose preserved?" is never tested before the s10 `pose_gate`.** "Redock consistency" and
"functional-state preservation" are silently conflated.

**P3 — ML top-N hard funnel.**
s08 `ml_score` strongly prunes the candidate set; the expensive stages (s08b/s09/s10)
only ever see that top subset. ML enriches **MD-pass / binding-validity** (AUC ≈ 0.83)
but **not catalysis/NAC** (AUC ≈ 0.36). Used as a hard filter, it can discard good
catalytic candidates before they are ever validated.

**Corollary (verified 2026-06-28 against the restored provenance).** The features the
ranking and s09 rely on are themselves built on the suspect pose:
- ML rank-1 `N288T` → `invalid_cosubstrate_diffused` in s10; the only NAC-positive s10
  candidate was `S340G` (later **refuted** as a pose artifact by anchored MD).
- `catalytic_geometry_penalty` median: **all-467 = 0.0**, **ML-top40 = 2.99**,
  **s10-9 = 1.70** (only 9% of all candidates carry any penalty, yet the ML-top are
  enriched for it).
- `d_ligand_iptm` for the top40: **40/40 negative** (min −0.534, median −0.413).
- **Decisive:** for the top40, `catalytic_geometry_source`, `mechanism_source`, and
  `boltz_delta_source` are **100% `mutant_boltz`** — i.e. the geometry penalty *and* the
  binding delta are computed **on the per-mutant Boltz pose**, the exact structure the
  redesign flagged as noisy.

➡ The honest reading is **not** "ML rewards perturbation." It is: **the entire
selection + non-MD feature stack is computed on pose-search-contaminated structures**,
so it cannot measure functional-state preservation. ML is *downstream* of the
contamination. (The "all-negative `d_ligand_iptm`" claim additionally lacks a control —
only the ML-top40 were folded — so it needs the low-ML control lane to be conclusive.)

---

## 2. The v2 internal contract

**v1 (current — from ARCHITECTURE.md):**
```
sequence + ligand
 → homolog/MSA → Boltz complex → docking → ligand–residue graph
 → mutation gen → ML rerank → non-MD validation → MD → final multi-objective rank
```

**v2 (target):**
```
sequence + FUNCTIONAL REFERENCE STATE            (role-tagged ligands + curated/Boltz reference)
 → multi-source homolog/MSA (with QC)
 → REFERENCE COMPLEX + role-tagged ligands        (design_ligand/cofactor/substrate/product/metal/context/TS-proxy)
 → FUNCTIONAL-STATE GRAPH                          (all functional partners + catalytic residues + waters/metals, not 1 ligand)
 → MULTI-LANE candidate generation                (binding / catalytic-geometry / stability / diversity / control)
 → ML as PRIOR, never a hard filter
 → de novo pose HYPOTHESIS  ⟂  REFERENCE-ANCHORED validation   (kept separate, never conflated)
 → MD / NAC / RBFE / GBSA as GATE-STACK EVIDENCE  (role-specific objectives)
 → EVIDENCE-CLASS / Pareto library                (not a single scalar rank)
```

Pipeline stage **names stay the same**; the **contract inside each stage changes**.

---

## 3. Already landed — Phase 0 (6-principle anchored redesign)

These exist and are unit-tested; v2 promotes them from "available" to "default contract."

| Capability | Module | Current scope |
|---|---|---|
| WT-anchored mutant builder | `src/evoliez/md/anchored_build.py` | used by s10 only |
| Reference-like pose gate (role-aware) | `src/evoliez/md/pose_gate.py` | s10 post-MD |
| Generic functional-geometry spec | `src/evoliez/md/geometry_spec.py` | module; FDH uses `reactive_geometry`/NAC |
| Gate-stack verdict | `src/evoliez/md/gate_stack.py` | s10 end-of-run pass |
| Multi-lane selection | `src/evoliez/ranking/multi_lane.py` | **opt-in**, default OFF |
| Config flags | `src/evoliez/config.py:560` (`SelectionLanesConfig`), `MDConfig.anchored_validation`/`pose_gate_enabled` | defaults: anchored ON, lanes OFF |
| Purge-safety guard | `src/evoliez/context.py` | landed |

**Gap v2 closes:** anchoring is s10-only; the *graph, generation, ranking, and s09
features* still run on the primary ligand and the de novo Boltz pose.

---

## 4. Stage-by-stage: current → v2 target

| Stage | Current | v2 target |
|---|---|---|
| **s01_input** | role manifest exists | promote `design_ligand / cofactor / substrate / product / metal / context / transition_state_proxy` to **explicit functional objectives** with per-role goals |
| **s02/s03** | direction good | surface real-vs-synthetic/remote homologs in QC; require **Neff, subfamily balance, motif alignment** |
| **s04_complex** | WT Boltz = de facto reference | wire `input.target_structure` / curated PDB as a **first-class reference**; separate *computed* vs *experimental/curated* reference in provenance |
| **s05_docking** | WT-reference docking exists | label whether the docking reference is computed-Boltz or curated PDB |
| **s06_graph** | **biggest change** — primary-ligand proximity mask (`s06_interaction_graph.py:33`) | **functional-state graph**: cofactor–substrate–residue (–metal/–water) triad; design mask = proximity to the **functional state**, not one ligand |
| **s06b_interaction** | family geometry model useful | add **pose validity, ligand role, PLIP/contact type, context-ligand preservation** as explicit features |
| **s07_mutation_gen** | strong generation | split into **lanes**: ligand-binding / catalytic-geometry / stability / diversity-control |
| **s08_reranker** | ML over-gates | ML → **prior**; **multi-lane selection default-on** in paper/production profile; **low-ML control mandatory** |
| **s08b_mutant_boltz** | top-N de novo predicted | tag de novo pose as **"alternative hypothesis"**; keep separate from the reference-anchored structure |
| **s09_nonmd** | reinforced | **separate redock-consistency from WT-like preservation** in report + gates; compute features on **anchored** structures (not self-Boltz pose) |
| **s10_md** | newest code is right | **clean run must persist** `anchored_validation` + `pose_gate` + `gate_stack` |
| **s11_final** | scalar-score centric | **evidence-class / Pareto / gate-stack-verdict** library, not a single rank |

---

## 5. Phased roadmap (A–G)

> Each phase is independently shippable, mock-testable, and gated on user review.
> Template: **Goal · Changes · Contract · Deliverables · Acceptance · Depends · Effort/Risk.**

### Phase A — Functional-state reference contract `[s01, s04, s05, s10]`
- **Goal.** Make the *role-tagged reference complex* the single anchor for the whole run.
- **Changes.** Elevate ligand roles to first-class objectives (per-role objective table:
  binding→affinity/stability, substrate/TS→catalytic geometry, cofactor→orientation/
  retention, metal→coordination, context→conserved-contact). Wire `input.target_structure`/
  curated PDB as a real reference; separate "computed reference" vs "curated reference."
- **Contract.** `RoleSpec` per ligand in config; `ReferenceState` object {structure
  (curated|boltz), role-tagged ligand poses, catalytic residues, waters/metals}.
- **Deliverables.** config `RoleSpec`/`ReferenceState`; reference loader; provenance flag
  `reference_source ∈ {curated_pdb, computed_boltz}`; s04/s05/s10 consume `ReferenceState`.
- **Acceptance.** A target with a curated PDB + role-tagged ligands runs end-to-end; every
  downstream artifact records which reference it anchored on.
- **Depends.** Phase 0 (anchored_build). **Effort/Risk.** M / medium (touches s01/s04 contracts).

### Phase B — Functional-state graph `[s06, s06b]`
- **Goal.** Replace "near the primary ligand" with "near the functional state."
- **Changes.** Build the design mask + interaction graph from **all** functional partners
  (role-tagged ligands + catalytic residues + TS proxy + conserved waters/metals). Add a
  reaction-state (triad) graph. s06b features gain pose validity / role / contact-type /
  context preservation.
- **Contract.** `FunctionalStateGraph` keyed by the reference state, not `cx.ligand`.
- **Deliverables.** graph builder; multi-ligand design mask; role/contact/preservation features.
- **Acceptance.** FDH design mask covers formate + the catalytic core (R290/H338/…), not
  only NADP-proximal residues; a metal/water can be declared and enters the graph.
- **Depends.** Phase A. **Effort/Risk.** L / high (s06 is load-bearing — the deepest change).

### Phase C — Multi-lane generation + ML-as-prior `[s07, s08, s08b]`
- **Goal.** Stop ML from being a hard funnel; never lose a catalytic candidate to a binding prior.
- **Changes.** Lane-aware generation (binding / catalytic-geometry / stability / diversity /
  control). Turn `selection_lanes.enabled` **on by default** in the paper/production profile
  (the schema already exists, `config.py:560`). Tag s08b de novo poses as alternative
  hypotheses; keep them separate from anchored structures.
- **Contract.** MD shortlist = **union of lanes incl. low-ML control**, each candidate
  carrying `selection_lane`.
- **Deliverables.** lane-aware s07; production profile YAML with lanes on + tuned counts;
  pose-hypothesis tagging in s08b.
- **Acceptance.** MD set is provably a lane union (not a single ML cut); ≥1 low-ML control
  reaches MD; de novo poses are never labeled "improved mutant."
- **Depends.** Phase 0 (multi_lane). **Effort/Risk.** M / low-medium (largely wiring + a profile).

### Phase D — Anchored features upstream `[s09]`  ← directly fixes the §1 corollary
- **Goal.** Compute selection/validation features on **anchored** structures, not the
  per-mutant Boltz pose.
- **Changes.** In s09, build the WT-anchored mutant (Phase 0 builder) and compute
  `catalytic_geometry_penalty` / binding delta there; keep the Boltz-pose features as a
  separate "alternative-hypothesis" column. Split **redock-consistency** from **WT-like
  preservation** in report + gates.
- **Contract.** `catalytic_geometry_source = anchored` becomes the ranking input;
  `*_boltz` sources are kept but demoted to hypotheses.
- **Deliverables.** anchored feature pass in s09; report/gate separation; provenance source tags.
- **Acceptance.** For a re-run, `catalytic_geometry_source` is `anchored` (not
  `mutant_boltz`); a candidate can pass redock-consistency yet fail WT-like preservation
  (and the report shows both).
- **Depends.** Phases A + 0. **Effort/Risk.** M / medium (adds an anchored build into s09's loop).

### Phase E — Gate-stack evidence output `[s10, s11]`
- **Goal.** Final claim = an evidence stack, not a scalar.
- **Changes.** s10 *always* persists `validation_structure`, `pose_gate`, `gate_stack`
  (clean-run requirement). s11 emits an **evidence-class / Pareto** library
  (stable · reference-pose-preserved · functional-geometry-improved · energetically-
  confirmed · experimentally-pending) instead of one ranked score.
- **Contract.** Paper-grade candidate **requires** `validation_structure=wt_anchored` +
  `pose_gate.design_ligand.status=reference_like` + a `gate_stack.verdict`.
- **Deliverables.** evidence-class s11; paper-grade gate; report verdict columns.
- **Acceptance.** s11 output groups candidates by evidence class / Pareto front; any s10
  record lacking the anchored fields is excluded from paper-grade claims.
- **Depends.** Phases A, D. **Effort/Risk.** M / low-medium.

### Phase F — ML retrain & re-evaluation `[s06b/s08 model]`
- **Goal.** Re-align ML once it trains on clean (anchored) features.
- **Changes.** Retrain on anchored features/labels; evaluate against **functional-state
  preservation** (AUC→NAC / reference-like / gate-stack verdict), not just MD-pass; use the
  low-ML control lane to quantify the false-negative rate.
- **Contract.** ML label = "reference-like functional complex preserved **and** stability/
  interaction/geometry improved" — never "high predicted affinity / changed pose."
- **Deliverables.** anchored training set; re-eval report (functional metrics + FN rate).
- **Acceptance.** ML is scored on functional-state preservation; the control lane gives a
  measured false-negative rate; ML stays a prior regardless of the number.
- **Depends.** **Phases A–E first** (this is intentionally last — clean contract before retrain).
  **Effort/Risk.** L / research.

### Phase G — Generalization & QC `[s02, s03, cross-cutting]`
- **Goal.** Prove "not FDH-only."
- **Changes.** MSA QC (Neff, subfamily balance, motif alignment, real-vs-synthetic flags);
  drive a **second target** (different reaction/cofactor/metal) by role + geometry config alone.
- **Deliverables.** MSA QC metrics in report; a 2nd-target config + smoke.
- **Acceptance.** A non-FDH target runs end-to-end with **config only, no code changes**.
- **Depends.** A–E. **Effort/Risk.** M / medium.

---

## 6. Sequencing, dependencies, and the first slice

```
Phase A (reference contract) ──┬─> Phase B (functional-state graph)
                               ├─> Phase D (anchored features) ──> Phase F (ML retrain)
Phase C (multi-lane, mostly built) ─┘                          │
Phase A,D ─> Phase E (evidence output)                          │
A–E ─> Phase G (generalization) <──────────────────────────────┘
```

- **Critical path:** **A → D → F.** Per the diagnosis, *fix the reference/functional-state
  contract and the feature anchoring first; retrain ML only after.* Retraining ML on the
  current contaminated features would relearn the same bias.
- **Recommended first slice (smallest valuable PR):**
  1. **Phase A core** — `RoleSpec` + curated-reference wiring (the contract everything else needs).
  2. **Phase C switch** — flip `selection_lanes.enabled` on in a production profile (already
     built; immediate false-negative protection at near-zero risk).
  - These two are low/medium risk and unblock B/D/E.
- **Lowest-risk quick win independent of A:** Phase E's "clean-run requires anchored fields"
  gate (pure provenance/report rule).

---

## 7. Invariants / non-goals (do **not** break)

- Mock/real backend separation; real-backend honest-degrade (no silent mock fallback).
- ML data policy: Boltz outputs stay features/weights/weak signals — **never supervised
  experimental labels** (see [docs/ML_DATA_POLICY.md](ML_DATA_POLICY.md)).
- Restraints stay **distance-only, never angle** (an angle restraint would manufacture NAC).
- Pipeline/stage names unchanged; changes are to the **contract inside** each stage.
- Shared-server discipline + purge-safety guard remain in force.
- v2 is **not** "add every new model" — it is "fix the contract, then re-evaluate models."

---

## 8. "v2 done" definition (acceptance for the epoch → bump `__version__` 0.2.0)

1. A target is defined by a **functional reference state** (role-tagged ligands + curated|
   computed reference), not a single ligand.
2. The design graph is built from the **functional state** (≥2 functional partners + catalytic
   residues enter the FDH mask).
3. The MD shortlist is a **multi-lane union with a low-ML control**, by default in the
   production profile.
4. Selection/validation features are computed on **anchored** structures
   (`catalytic_geometry_source = anchored`); redock-consistency and WT-like preservation are
   **separate** signals.
5. Every paper-grade candidate carries `validation_structure=wt_anchored` +
   `pose_gate=reference_like` + a `gate_stack.verdict`; s11 outputs **evidence classes /
   Pareto**, not one scalar rank.
6. ML is retrained on anchored features and **evaluated on functional-state preservation**,
   with a measured false-negative rate from the control lane — and remains a prior.
7. A **second, non-FDH target** runs end-to-end with config only.

---

*Implementation is staged and user-gated: this document defines the plan; each phase is
built only when explicitly requested. The current in-flight clean anchored production run
(runs/fdh_anchored) is the Phase 0 → Phase E baseline that this v2 plan extends upstream.*
