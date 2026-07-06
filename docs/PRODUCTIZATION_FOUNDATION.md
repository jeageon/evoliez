# Productization foundation — what "commercial-platform" does and does NOT yet mean

EvoLiEZ has reached **commercial-platform FOUNDATION**: mechanism-configurable, claim-safe, and
reproducible-from-commit, demonstrated across three mechanism classes (CAR real; FDH + metalloenzyme
mock). It is **not** a shippable product. This file is the honest gap list, so "foundation" is never
misread as "commercial-ready."

## Status by criterion
| criterion | state |
|-----------|-------|
| System (MechanismSpec · ClaimGuard · explicit-solvent MD · Mg · evidence package) | **works** |
| CAR acceptance (Mg-consistent geometry, meaningful WT baseline, honest mechanism limitation) | **met** |
| Generality (3 mechanism classes, config-only, ClaimGuard-clean) | **foundation** (2 of 3 mock) |
| Candidate experiment recommendation | **withheld** (correct — no activity evidence) |

## Open items before "commercial-ready"
1. **Real non-redox backend run** (TEM-1 / glycosidase) — move generality from mock to proof.
2. **E4a PMF** (built, server run pending) — the correct reaction-geometry escalation before QM/MM.
3. **License / packaging / install** audit — deployable artifact + dependency + license review.
4. **Artifact-storage policy** — large run outputs off GitHub (Google Drive / object store), only
   small provenance/summaries committed (already the practice; needs to be a written policy + .gitignore).
5. **CI stabilization** — the 9 pre-existing test failures (triaged below) resolved or explicitly xfail'd.
6. **≥2 real benchmarks** — beyond CAR, at least two real targets with claim-safe evidence output.
7. **User-facing report template** — a clean, uniform deliverable per run.

## The 9 pre-existing test failures — triage
| test | class | disposition |
|------|-------|-------------|
| `test_boltz_cif::test_parse_cif_atom_site_loop` | **stale test** — parser returns a 3-tuple `(residues, lig, extra)`; test unpacks 2 | **fix** (update test to the multi-ligand API) — real, small debt |
| `test_boltz_outdir_scoping::test_pdb_parser_keeps_only_primary_ligand_chain` | same 2-vs-3-tuple stale test | **fix** |
| `test_tool_output_parsers::test_boltz_pdb_and_cif` | same 2-vs-3-tuple stale test | **fix** |
| `test_geometry_spec::test_real_smarts_resolution` | **local-env** — rdkit not in `.venv-light`; **passes on the server** | CI: install rdkit, or mark `requires_rdkit` |
| `test_dryrun_real_default_config::…without_database` | **env** — real-backend dry-run needs a homolog DB path | CI: provide a DB stub or mark `requires_real_env` |
| `test_mutant_boltz::test_dry_run_real_labels_source_as_dry_run` | **env** — real-backend dry-run | same |
| `test_pipeline_mock_e2e::test_dry_run_real_backend_builds_commands` | **env** — real-backend dry-run | same |
| `test_reuse_dir_purge::test_changed_fingerprint_purges_db_and_artifacts` | **env** — reuse/purge fixture | CI: fixture/env fix |
| `test_config::…[example_metalloenzyme.yaml]` | **illustrative placeholder** — the metallo example's His-triad tokens don't match its borrowed FASTA (by design) | accept, or ship a real metallo FASTA (the `gen_metallo_acyl` demo config already aligns its tokens) |

**Summary:** 3 are genuine (small) stale-test debt (the boltz parser now returns a 3-tuple); 5 are
**environmental** (rdkit / real-backend DB / fixtures — they pass with the right CI env, not code bugs);
1 is an accepted illustrative-placeholder mismatch. None is a defect in the shipped mechanism/ClaimGuard
framework. Fixing the 3 stale tests + wiring a CI env for the 5 environmental ones clears the suite for
productization; that work is deliberately out of scope for the science tracks above.

## Positioning language
Use **"commercial-platform foundation + one real hard case (CAR) + mock generality proof."** Avoid
"commercial-ready", "production product", or any candidate-activity claim until items 1–7 close.
