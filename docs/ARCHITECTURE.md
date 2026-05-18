# Architecture & spec cross-reference

`evoliez` implements the architecture in
`EvoLigand_Enzyme_Engineer_Complete_MD_Architecture_Plan.md`. Each spec module
maps to one pipeline stage; each external tool to one adapter.

## Stage map

| Stage | Spec | Module | Adapters used |
|---|---|---|---|
| `s01_input` | §6 | `stages/s01_input_preprocess.py` | `features/ligand` (RDKit) |
| `s02_homolog` | §7 | `stages/s02_homolog.py` | `adapters/msa_tools` (mmseqs2/jackhmmer/blastp) |
| `s03_msa` | §8 | `stages/s03_msa.py` | `adapters/msa_tools` (mafft), `adapters/remote_msa` |
| `s04_complex` | §9 | `stages/s04_complex.py` | `adapters/boltz` (Boltz-2) |
| `s05_docking` | §10 | `stages/s05_docking.py` | `adapters/{vina,gnina,diffdock}` |
| `s06_graph` | §11 | `stages/s06_interaction_graph.py` | `features/graph` |
| `s07_mutation_gen` | §12 | `stages/s07_mutation_gen.py` | `adapters/ligandmpnn` |
| `s08_reranker` | §13 | `stages/s08_reranker.py` | xgboost (optional) |
| `s09_nonmd` | §14 | `stages/s09_nonmd_validation.py` | `adapters/{foldx,rosetta}`, dockers |
| `s10_md` | §15 | `stages/s10_md.py` | `adapters/openmm_engine`, `md/analysis` |
| `s11_final` | §16 | `stages/s11_final_ranking.py` | `ranking/score`, `io/report` |

Persistence: `db/schema.py` = spec §17.1 tables; `io/paths.py` = spec §17.2
directory layout. Config: `config.py` = spec §19. Orchestrator: `pipeline.py`
= spec §18.1.

## Backends

`Config.backend` (or per-stage `Config.backends`) selects `mock` or `real` for
every adapter. `mock` produces deterministic synthetic structures/poses/
sequences (seeded in `utils/seeds.py`) so the full data flow, schema, scoring
and reporting run identically on a laptop. `real` shells out (via
`utils/subprocess_utils.py`) to the actual tool and parses its output; missing
tools raise an actionable error or degrade to mock per the spec §23 risk
mitigations.

## Phase coverage (spec §21)

- Phase 0 scaffold/schema/config/example/report ✓
- Phase 1 rule-based full pipeline (s01–s06, s09 stability+geometry, s10 L0, s11) ✓
- Phase 2 LigandMPNN (`adapters/ligandmpnn`) ✓
- Phase 3 reranker + weak-supervision heuristic / XGBoost (s08) ✓
- Phase 4 MD-lite restrained relaxation (`md/restraints`, openmm L1) ✓
- Phase 5 short / explicit-solvent MD + replicas (openmm L2–L3, config) ✓
- Phase 6 experimental-label supervised reranker (s08 `use_experimental_labels`) ✓

## Data flow (artifacts on `ctx`)

```
target_sequence, ligand            (s01)
 -> homologs                       (s02)
 -> msa, position_features         (s03)
 -> wt_complex                     (s04)
 -> reference_atoms                (s05)
 -> interaction_graph, designable_positions  (s06)
 -> candidates                     (s07)
 -> redock_candidates              (s08)
 -> validated_candidates, md_candidates       (s09)
 -> md scores on md_candidates     (s10)
 -> ranked_candidates + reports    (s11)
```
