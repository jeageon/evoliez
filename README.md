# EvoLigand-Enzyme Engineer (`evoliez`)

Ligand-aware, enzyme-family-specific protein engineering pipeline.

Input: a target enzyme sequence + a ligand. Output: ranked, evidence-weighted
mutation candidates and a focused experimental library, built by combining
evolutionary constraints, ligand atom-level geometry, structural confidence,
redocking consistency, mutation-induced stability, and short molecular dynamics.

This is a faithful implementation of the architecture in
`EvoLigand_Enzyme_Engineer_Complete_MD_Architecture_Plan.md`.

```
sequence + ligand
 -> homolog retrieval -> MSA + evolutionary features
 -> protein-ligand complex prediction (Boltz-2)
 -> docking / redocking ensemble
 -> ligand atom-residue interaction graph
 -> mutation generation (LigandMPNN / MSA / chemistry rules)
 -> family-specific reranking
 -> non-MD validation (stability, geometry, redocking)
 -> OpenMM MD validation (minimization -> MD-lite -> short MD)
 -> final multi-objective ranking + focused library
```

## Two execution modes

| Backend | Where | What it does |
|---|---|---|
| `mock`  | laptop / CI (no GPU) | deterministic synthetic structures/poses/sequences so the **whole pipeline runs end-to-end** for development and testing |
| `real`  | the CUDA server | shells out to the actual tools (Boltz-2, DiffDock, GNINA, LigandMPNN, FoldX, OpenMM-CUDA) |

The pipeline, data schema, scoring, reporting and resume logic are identical in
both modes — only the heavy tool adapters change.

## Quickstart (laptop, mock)

```bash
make venv          # local virtualenv with light deps
make test          # unit tests + full mock end-to-end pipeline
make demo          # run the FDH/NADP example, see runs/demo_fdh_nadp/reports/
```

## Server deployment

The real pipeline runs on a GPU server. **Do not install on root `/`** — see
[`docs/SERVER_RUNBOOK.md`](docs/SERVER_RUNBOOK.md). In short:

```bash
git clone <your-fork> /mnt/data2/<you>/EvoLiEZ && cd /mnt/data2/<you>/EvoLiEZ
bash scripts/setup_server_env.sh        # conda env at /mnt/data2/envs/evoliez
bash scripts/fetch_weights.sh           # Boltz-2 / LigandMPNN weights -> /mnt/data2
bash scripts/run_pipeline.sh configs/example_fdh_nadp.yaml   # auto-picks a free GPU
```

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module map, spec section cross-reference
- [`docs/SERVER_RUNBOOK.md`](docs/SERVER_RUNBOOK.md) — server install + run, GPU/disk constraints
- [`docs/DATA_AND_WEIGHTS.md`](docs/DATA_AND_WEIGHTS.md) — where to get models, weights, databases

## Status / scope

Implements Phases 0–6 of the spec. External GPU tools are integrated via
subprocess adapters (the spec treats them as external references, not code to
re-implement). Out of scope: activity guarantees without labels, routine FEP,
de novo design, wet-lab automation, foundation-model fine-tuning.
