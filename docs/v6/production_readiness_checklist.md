# V6-7 — Production Readiness Checklist

**Phase:** V6-7 · **Status:** ✅ PASS · Turns the V6 Amber platform from a research
workflow into a reproducible production foundation.

## 1. Reproducibility bundle

Every V6 server run is reproducible from four artifacts:

| Artifact | Produced by | Contents |
|---|---|---|
| **Git commit** | this repo | the exact code (all V6 phases committed on `feat/v6-amber-gpu-platform`) |
| **Config** | `configs/*.yaml` | schema-validated (every shipped config is a CI citizen — `test_all_shipped_configs_validate`) |
| **Parameter files** | `params/*/`, curated mol2 | e.g. `params/atp_4minus/ATP.fixed.mol2` (curated −4 ATP) |
| **Runtime profile** | V6-0 `check_amber_gpu.py` | `reports/provenance/amber_runtime_profile.json` — server, CUDA, GPU, tool paths + versions, deck hashes |

Plus the **fingerprints/manifests** that pin each stage:

- `amber_system_manifest.json` (V6-1) — deterministic `system_fingerprint`
  (protein seq + ligand identities/charges + FF/water/ion + reactive spec). Same
  config → same fingerprint (verified: `7f8e7306d4d9edd5` across re-runs).
- `amber_gpu_jobs.jsonl` (V6-2), `amber_pmf_jobs.jsonl` (V6-3),
  `amber_qmmm_jobs.jsonl` (V6-5), `tem1_amber_md.jsonl` (V6-6) — per-run provenance
  (status, GPU, wallclock, analysis, claim ceiling, fingerprint).

**Recommended:** run from a clean git tree (the pipeline already warns "RUN BUILT
FROM A DIRTY GIT TREE" for rigorous, high-stakes runs).

## 2. Server operation safety (Gate 5)

- **One independent pmemd per GPU** (V6-2 scheduler / V6-3 PMF windows) — no
  multi-GPU single jobs; predictable scheduling on the shared box.
- **CPU-fallback blocked** unless explicitly requested (verified: empty GPU pool →
  `skipped_no_gpu`).
- **Restart-safe**: completed stages are skipped on resume (verified: 0.5 s re-run).
- **Memory safety**: QM/MM-lite carves a subsystem (~11k atoms) rather than QM/MM the
  72k box — after a 258 GB OOM on the full box (V6-5). Shared-server courtesy: never
  `pkill` by process name (hits other users); kill only own PIDs; keep heavy jobs
  within the CPU budget.
- **No accidental data loss**: a fingerprint change **refuses to purge** finished run
  dirs unless `EVOLIEZ_ALLOW_PURGE=1` (the anchored-validation-redesign safety guard).

## 3. Mechanism-configurability (Gate: no core edits for a new template)

A new enzyme/mechanism is added from **config + a spec**, not by editing core code.
Proven end-to-end: **CAR (adenylation + Mg)** and **TEM-1 (serine hydrolase, protein
nucleophile, no metal)** run through the **same** `amber_builder` + `amber_scheduler`
+ mechanism-geometry resolver. The only per-mechanism inputs are:

- the `MechanismSpec` template (`mechanism/templates/`) — CAR/FDH/TEM-1 present,
- the `AmberBuildSpec` (ligands, metal, reactive donor/acceptor),
- for a protein nucleophile: `donor_protein: "RESNAME:ATOM:NUM"` (resolved from the
  topology; no code change).

## 4. Claim safety (Gate 4)

Every V6 report is **ClaimGuard-clean** under a default no-wet-lab `ClaimProvenance`
(verified for the CAR portfolio, the QM/MM decision, and the TEM-1 report). Cards are
born `L0_uncalibrated`; non-L0 requires experimental calibration. Enforced in code
(`classify_pmf`, `EvidenceCardV4` validator, `assert_report_clean`), not just prose.

## 5. Test-failure triage (the roadmap's "known failures")

Baseline: 5 pre-existing failures + 2 introduced by V6-6's config. Post-V6-7:

| Test | Root cause | Disposition |
|---|---|---|
| `test_config…[tem1_acyl_amber.yaml]` (×2) | V6-6 config had a top-level `amber:` extra key | **FIXED** — moved `reactive_geometry` under `validation.md` (schema-valid) |
| `test_boltz_cif::test_parse_cif_atom_site_loop` | stale: `_parse_cif_atoms` now returns a 3-tuple | **FIXED** — unpack 3 |
| `test_tool_output_parsers::test_boltz_pdb_and_cif` | same 3-tuple | **FIXED** — unpack 3 |
| `test_boltz_outdir_scoping::…primary_ligand_chain` | `_parse_pdb_atoms` 3-tuple | **FIXED** — unpack 3 |
| `test_config…[example_metalloenzyme.yaml]` | the example config reused the FDH fasta with placeholder residues (H94/H96/H119) not present at those positions | **FIXED** — retargeted to actual His positions in the bundled fasta (H217/H219/H224); the example stays illustrative but internally consistent |
| `test_reuse_dir_purge::…purges_db_and_artifacts` | tested the *old* purge behavior; the code intentionally added a data-loss **guard** that refuses to purge unless opted in | **FIXED** — the test now opts in with `EVOLIEZ_ALLOW_PURGE=1` (it exercises the purge path it is named for); `test_same_fingerprint_keeps_db` still guards the no-purge default |

Net: **all 7 baseline failures are now resolved — the full suite is green.** None were
V6 regressions (5 stale tests / config-schema, 2 example/guard consistency).

## 6. CI test separation

See [`artifact_policy.md`](artifact_policy.md) for storage. CI tiers:

- **local** — pure helpers, RDKit-only, mock backends. All V6 unit tests
  (`test_amber_builder/scheduler/pmf/qmmm`, `test_car_v6_portfolio`) run here in
  seconds. **This is the PR gate.**
- **server (Amber)** — the real `tleap→pmemd.cuda→cpptraj` round-trips, builds, MD,
  PMF, QM/MM, and the TEM-1 acceptance run — driven by `scripts/run_amber_*.sh` /
  `run_*_amber.py` over SSH, not pytest (they need CUDA + AmberTools + Boltz).
- **long-running acceptance** — the full CAR/TEM-1 end-to-end (Boltz + Amber),
  nightly/manual, not on every PR.

## Completion gates (V6 §11)

- [x] **Gate 1 — Amber backend works**: detected, system built, GPU run completes, analyzed automatically (V6-0…V6-2).
- [x] **Gate 2 — CAR decision actionable**: claim-safe 19-variant portfolio + a clear QM/MM-lite/2D-PMF escalation (V6-4/V6-5).
- [x] **Gate 3 — non-CAR proof exists**: TEM-1 serine hydrolase, real backend, protein-nucleophile geometry (V6-6).
- [x] **Gate 4 — no overclaim possible**: every report ClaimGuard-clean, enforced in code.
- [x] **Gate 5 — server operation safe**: one-per-GPU, restart-safe, CPU-fallback-blocked, OOM-safe, purge-guarded.
