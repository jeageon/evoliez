# CAR V5 Server Runbook (Day 0–7)

> Server-smoke orchestration for the CAR adenylation V5 acceptance test. **Core code lives in the
> merged PR #1 (`feat/server-hardening`); this branch (`feat/car-v5-smoke-runbook`) adds ONLY the
> smoke config + launcher + acceptance judge — no core-code changes.** CAR is the **acceptance test
> for a mechanism-configurable triage platform**, not a CAR-specific tool: success = valid,
> claim-safe mechanism evidence, NOT "`G430R;S433F;G407K` ranks #1".

## Artifacts on this branch
- `configs/car_srcar_3hp_v5.yaml` — production V5 config (in PR #1; A-domain primary, adenylation mechanism, Mg²⁺, O→P via legacy path).
- `configs/car_srcar_3hp_v5_smoke.yaml` — smoke override (level 1 / 0.05 ns / 1 replica / top 3 / no RBFE/GBSA).
- `configs/car_v5_focused_candidates.csv` — the locked 16–24 focused candidate manifest.
- `scripts/run_car_v5.sh` — one self-contained launcher: `preflight | smoke | focused`.
- `scripts/check_car_v5_provenance.py` — the acceptance judge (PASS / CONDITIONAL / FAIL), unit-tested.

---

## Day 0 — merge-readiness (DONE)
PR #1 merged into `feat/server-hardening` (merge commit `34309b0`). V5 tests green; pre-existing 9 failures documented; no report writer bypasses ClaimGuard. **No server run before merge.**

## Day 1 — merge + server preflight
```bash
# on the server:
git checkout feat/server-hardening && git pull && git log --oneline -5
test -f configs/car_srcar_3hp_v5.yaml && test -f src/evoliez/md/metal_placement.py
bash scripts/run_car_v5.sh preflight        # lightweight V5 tests + config load + OpenMM platforms
```
**PASS if:** V5 lightweight tests pass · CAR V5 + legacy-FDH configs load · OpenMM imports and lists a platform. Record `server_preflight_log.md` (git commit, env, OpenMM platforms, test result). Fail → stop, log as a follow-up issue.

## Day 2 — Mg/OpenMM smoke (WT + 2 variants)
```bash
bash scripts/run_car_v5.sh smoke            # config=car_srcar_3hp_v5_smoke.yaml, --resume, then the verdict
```
Check `runs/srcar_3hp_v5_smoke/reports/provenance/md_candidates.json`:

| field | PASS requires |
|---|---|
| `metal_setup.inserted_in_pdb` | `true` |
| `metal_setup.openff_parameterized` | `false` (Mg is a structure-level MG ion, never OpenFF) |
| `metal_setup.amber_standard_ion` | recognised by Amber (flip from `server_verified_pending`) |
| `geometry_source` | `legacy_reactive_geometry_from_mechanism_spec` |
| `nac.angle_mean` (WT + candidates) | **finite, not NaN** |
| MG in PDB/topology | present |

Failure classes: MG stripped by PDBFixer/Modeller → server-integration bug; Amber doesn't recognise MG → ion naming/FF; angle NaN → geometry_implementation_failure; MG absent but NAC 0 → invalid-interpretation bug. `check_car_v5_provenance.py` emits these automatically.

## Day 3 — smoke analysis + go/no-go
`scripts/check_car_v5_provenance.py runs/srcar_3hp_v5_smoke` → `go_no_go_car_focused_rerun.md`.
- **PASS/CONDITIONAL** → proceed to Day 4.
- **FAIL** → open **PR #3 (server integration fixes)**; fix; do NOT run the focused rerun.

## Day 4 — focused rerun manifest lock
Lock `configs/car_v5_focused_candidates.csv` (16–24; WT + lead + full deconvolution + P438 series + controls). Verify WT/mutant numbering against the A-domain FASTA, fix the Mg reference frame, record the run fingerprint.

## Day 5 — focused rerun
```bash
bash scripts/run_car_v5.sh focused          # production tier; RBFE OFF first pass, GBSA ON
```
1st pass: `protocol_level:3, production_ns:2.0, replicas:3, rbfe.enabled:false, binding_dg.enabled:true`. RBFE is a **2nd confirmatory tier** (`rbfe.enabled:true, top_n:3`) only after dynamic geometry is shown valid. **Abort if:** angle NaN reappears · MG missing in topology · >30% candidates fail assembly · OpenMM silently on CPU despite `fail_loud` · ClaimGuard strict failure.

## Day 6 — acceptance report (evidence validity, not lead-hunting)
`check_car_v5_provenance.py runs/srcar_3hp_v5` → verdict; write `car_v5_acceptance_report.md`.

| Verdict | Criteria |
|---|---|
| **PASS** | angle finite · Mg tracked · WT baseline finite · geometry terms computed · invalid cases classified · reports ClaimGuard-clean · (candidate-level differences observed) |
| **CONDITIONAL PASS** | all valid but no candidate discrimination → a **sampling limitation**, not a failure; recommend longer MD / explicit solvent / QM-MM / reference-ensemble refinement |
| **FAIL** | angle NaN · Mg setup broken · NAC 0 used despite missing Mg · Mg via OpenFF · ClaimGuard permits an activity/kcat/superiority claim |

## Day 7 — split follow-up PRs
- **PR #3 — server integration fixes** (only issues found Day 2–6: MG residue naming, PDBFixer `removeHeterogens`, OpenMM ion FF mapping, topology atom-index mapping, `metal_setup` propagation). No core-behavior expansion.
- **PR #4 — EvidenceCard ranking A/B** (only after the focused run): `legacy_vs_evidence_rank_ablation.csv`, lead/control retained, `rank_shift_report.html`; decide the default flip. `ranking.use_evidence_card` stays `false` until then.
- **PR #5 — generality proof**: FDH hydride regression + TEM-1 (nucleophilic acyl) + glycosidase config smokes + per-mechanism ClaimGuard policy + assay schema.

---

## Operating principles (always on)
1. **Evidence validity > new features.** CAR success = valid Mg-inclusive dynamic adenylation geometry recorded as ClaimGuard-passing evidence.
2. **"No discrimination" may be CONDITIONAL PASS**, not FAIL — if Mg + O→P are valid, escalate sampling; don't call it a pipeline failure.
3. **ClaimGuard always on.** Pre-wet-lab, forbid `activity improved` / `kcat improved` / `confirmed productive` / `catalytically superior`; allow `hypothesis-grade lead` / `screening-level catalytic-geometry evidence` / `prioritized for experimental testing`.
4. **A-domain primary**; full-length CAR is context validation only.
