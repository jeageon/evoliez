# V6 Amber-GPU Platform — Sign-off

**One document to read for what V6 completed, what it did not, and the claim boundary.**
Consolidates the per-phase docs; the live evidence is in
[`server_verification_report.md`](server_verification_report.md).

- **Branch:** `feat/v6-amber-gpu-platform` (11 commits over `feat/car-v5-reviewer2-refinements`)
- **Server:** evo, 4× RTX A6000, CUDA 12.4 — real runs, not mock
- **Status:** platform foundation complete + server-verified. **Not** a commercial/product release.

---

## 1. Phase-by-phase status

| Phase | What it delivers | Status | Live evidence |
|-------|------------------|--------|---------------|
| V6-0 Amber capability audit | `tleap→pmemd.cuda→cpptraj` round-trip + runtime profile | ✅ | implicit GB (0.51 Å) + explicit PME (0.22 Å) GPU round-trip; no CPU fallback |
| V6-1 Amber system builder | mechanism-generic build, reactive-atom map, fingerprint | ✅ | CAR A-domain (ATP+3-HP+Mg) built; ambiguous inputs fail loud |
| V6-2 Amber GPU MD executor | 1 pmemd/GPU, restart-safe, cpptraj geometry | ✅ | WT+G430R+lead explicit trajectories; **WT reproducible to 0.02 Å** |
| V6-3 Amber PMF tier | umbrella + WHAM/QC + convergence gate | ✅ (diagnostic) | 13-window CAR PMF ran; **non-converged → `diagnostic_only`, ranking-prohibited** |
| V6-4 CAR portfolio | claim-safe mechanism-probe panel + EvidenceCards | ✅ | 19-variant panel, all cards L0; ClaimGuard-clean |
| V6-5 QM/MM-lite | reaction-core plausibility on carved subsystem | ✅ (smoke) | WT+candidates; claim ceiling "plausibility, NOT activity" |
| V6-6 non-CAR proof | real backend on a non-redox target | ✅ | **TEM-1 β-lactamase** Amber+GPU MD; Ser68 Oγ→β-lactam geometry resolved |
| V6-7 production hardening | reproducibility, artifact/license policy, tests | ✅ | full suite green; server code sha1-synced to committed |

All five completion gates (Amber works · CAR decision actionable · non-CAR proof ·
no overclaim possible · safe server ops) are demonstrated.

## 2. CAR — the honest decision

CAR yields a **hypothesis-grade mechanism-probe panel**, not a validated-lead set.

- **Panel:** [`outputs/car_v6/mechanism_probe_plate.csv`](../../outputs/car_v6/mechanism_probe_plate.csv)
  — 19 variants: the multipoint hypothesis **G430R;S433F;G407K**, its full
  single/pairwise **deconvolution** series, **P438** clean single-site probes, and
  epistasis/objective-conflict **controls**. Every well has a test-purpose rationale.
- **Strongest signal (screening-level, claim-safe):** in live 0.2 ns MD the multipoint
  variant reaches a near-linear in-line attack angle (167°/157° across two independent
  seeds) vs WT ~90°; **the qualitative ordering reproduces, magnitudes are noisy**, and
  **near-attack occupancy is 0.0 for every variant**. Distance-access and in-line-angle
  disagree → the PMF/QM-MM tier is the correct escalation, not a ranking.
- **PMF verdict:** non-converged (rugged 1-D coordinate) → `diagnostic_only`,
  **prohibited from ranking candidates**. This is honest surfacing of coordinate
  insufficiency, not a failure to produce candidates.

## 3. TEM-1 — generality proof (kept separate from CAR)

A non-redox, non-metal, **protein-nucleophile** target runs end-to-end on real
backends: Boltz Michaelis complex → Amber build → GPU MD → Ser68 Oγ→β-lactam
carbonyl geometry resolved from topology (mean 8.51 Å, drift 0.72 %). Report:
[`nonredox_real_backend_report.md`](nonredox_real_backend_report.md) — ClaimGuard-clean.
This proves the framework is **not CAR-specific**; it claims capability, not activity.

## 4. Claim boundary (ClaimGuard-enforced)

**Allowed:** screening-level reaction-geometry evidence · near-attack access cost ·
mechanism-probe panel · hypothesis-grade candidate · higher-cost validation recommended.

**Forbidden before wet-lab data** (all 6 ROADMAP §13 phrases now enforced after a
live-found Gate-4 hole was closed): activity/kcat improvement, catalytic validation,
validated-lead labeling, activation-barrier lowering, experimental-activity claims.

**G430R;S433F;G407K** is a *hypothesis-grade mechanism-probe candidate supported by
screening-level reaction-geometry evidence* — **not** a validated activity-improving lead.

## 5. Known limitations (not overclaims — the ceilings say so)

- Portfolio MD is screening-grade: 0.2 ns, template-mutagenesis side chains, few replicas.
- CAR 1-D PMF is non-converged (diagnostic-only).
- QM/MM-lite is smoke-grade (single-frame, thin `qm_atoms` metadata).
- Not commercial-ready; this is a production **foundation**.

## 6. Next research tiers (NOT V6 completion gates)

1. **✅ Done — with a cautionary result.** QM/MM-lite frame-stability check (3 frames
   × WT/P438N/lead, at two geometries). At a **non-reactive** distance (O→P ~6–9 Å)
   the variants separate cleanly (WT 91°, P438N 123°, lead 167°). At the
   **near-attack** reaction geometry (~3.2 Å) the separation **largely dissolves**:
   WT 142.7±8.1°, lead 156.0±2.3° (overlapping), P438N 109.1±6.5° (*below* WT) — and
   the earlier single-frame WT ≈ 69° does not reproduce (WT is not robustly bent). So
   the apparent in-line-angle advantage was substantially a frame/geometry artifact. The multi-frame check
   **weakened** the computational case for differentiating the CAR variants; it
   reinforces that ranking needs wet-lab data. See
   [`../car_v6/qmmm_frame_stability.md`](../car_v6/qmmm_frame_stability.md).
2. Converged access-cost: 2-D or better-coordinate PMF before any candidate ranking.
3. Only then: wet-lab assay of the deconvolution panel (with controls) — the only
   thing that can unlock activity-level claims.

## 7. Live-verification fixes folded in (2026-07-08)

Three real bugs, stop-and-fix: (1) stale server `amber_engine.py` energy-drift
(100.3 %→0.6 %); (2) candidate-input reproducibility gap → `prep_car_adomain_mutant.py`;
(3) Gate-4 ClaimGuard hole (4 of 6 §13 phrases leaked) → closed + regression-locked.
