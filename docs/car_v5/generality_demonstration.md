# Generality demonstration — one framework, three mechanism classes (commercial-platform criterion)

The commercial-platform criterion: *"FDH, CAR/adenylation, and at least one non-redox enzyme template
run under the same evidence/ClaimGuard framework."* Demonstrated:

| enzyme | reaction class | redox? | mode | verification |
|--------|----------------|--------|------|--------------|
| **FDH** (formate dehydrogenase) | `hydride_transfer` | redox | `mechanism_spec` | mock e2e s01→s11 ✓, final report ClaimGuard-clean ✓ |
| **metalloenzyme** (probe) | `nucleophilic_acyl_substitution` | **non-redox** | `mechanism_spec` | mock e2e s01→s11 ✓, final report ClaimGuard-clean ✓ |
| **CAR** (SrCAR → 3-HP) | `adenylation_phosphoryl_transfer` | non-redox, metal-bridged | `mechanism_spec` | **server-verified** (full V5 investigation) ✓ |

## What makes it generality (not three special cases)
- **Config-only.** Each enzyme declares only a `mechanism:` block (`reaction.class` + the template's
  `required_reaction_state`). No per-enzyme core code — the same `mechanism_mode` / template registry
  resolves all three (`test_v5_generality` proves the spec contract; `test_v5_generality_run_configs`
  proves the shipped run configs resolve it).
- **Three distinct chemistries.** Redox hydride transfer, non-redox acyl substitution, and
  metal-bridged phosphoryl transfer — the framework selects each template and emits its geometry
  terms without hardcoding a reaction.
- **Same evidence + claim discipline.** All three flow through the identical s01–s11 pipeline, produce
  the same report set, and pass ClaimGuard (no `validated catalytic lead` / `activity improved` /
  `catalytic lead` / `strong candidate` in any final report).

## Reproduce (mock, any machine)
```
PYTHONPATH=src python -m evoliez.cli run -c configs/gen_fdh_hydride.yaml      # hydride
PYTHONPATH=src python -m evoliez.cli run -c configs/gen_metallo_acyl.yaml # non-redox acyl
```
Both complete end-to-end and write `runs/<name>/reports/final_report.md`. CAR is the real-backend,
server-verified case (see `CAR_V5_EVIDENCE_PACKAGE.md`).

## Scope note
The FDH and metalloenzyme runs use the **mock backend** with placeholder sequences — they prove the
framework is mechanism-configurable and claim-safe across chemistries, not a real activity result for
those enzymes (the mock top candidate is illustrative only). A real-backend FDH run exists separately
(the NADP-switch campaign); a real non-redox run is the next data-gated step. This satisfies the
platform-generality criterion: the pipeline is enzyme-agnostic and the evidence/ClaimGuard layer is
uniform across mechanism classes.
