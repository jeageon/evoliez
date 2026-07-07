# Server runbook (admin1-SYS-740GP-TNRT)

Real GPU runs happen on the lab server. The dev box (Mac) only runs the mock
pipeline. Read the constraints before doing anything.

## Hard constraints

1. **Root `/` is ~99% full (≈30 GB free).** `~/anaconda3` is on root.
   - The conda env, model weights and sequence DBs **must** live on
     `/mnt/data2` (14 TB NVMe, ≈4.8 TB free). Fallback: `/mnt/data` (≈1.9 TB).
   - All `scripts/*.sh` refuse paths outside `/mnt/data2` / `/mnt/data`.
   - Set `project.output_dir` in the config to a `/mnt/data2` path.
2. **Shared GPU server (4× RTX A6000).** Other users run jobs (placer,
   rxnmapper, enzrep). Never assume a free GPU.
   - `scripts/run_pipeline.sh` and `utils/gpu.py` pick the least-busy GPU with
     enough free VRAM and pin `CUDA_VISIBLE_DEVICES`. One GPU per stage.
   - Do not hand-set `CUDA_VISIBLE_DEVICES=0,1,2,3`.

## First-time setup

One env var, `EVOLIEZ_ROOT`, drives the clone / conda env / weights / DB /
runs location. Put it in `~/.bashrc` so every script agrees. Must be under
`/mnt/data2` (NVMe, more free) or `/mnt/data` — never `/` or `$HOME`.

```bash
# pick your root (this example: /mnt/data/jglee)
export EVOLIEZ_ROOT=/mnt/data/jglee
echo 'export EVOLIEZ_ROOT=/mnt/data/jglee' >> ~/.bashrc

mkdir -p "$EVOLIEZ_ROOT" && cd "$EVOLIEZ_ROOT"
git clone -b feat/server-hardening <your-fork-url> EvoLiEZ
cd EvoLiEZ

# creates the conda env at $EVOLIEZ_ROOT/envs/evoliez
bash scripts/setup_server_env.sh
conda activate "$EVOLIEZ_ROOT/envs/evoliez"
pip install -e ".[science,md,gnn,dev]"      # dev = pytest for the unit step

bash scripts/fetch_weights.sh             # LigandMPNN/DiffDock code -> $EVOLIEZ_ROOT
# optional, large: local homolog DB. Otherwise set msa.remote_server: true
bash scripts/fetch_databases.sh uniref30

# Boltz in its OWN isolated env (NEVER `pip install boltz` into evoliez:
# Boltz pins numpy<2 / old click and will break rdkit/openmm/typer).
bash scripts/setup_boltz_env.sh           # -> $EVOLIEZ_ROOT/envs/boltz
```

> WARNING: do not `pip install` Boltz / DiffDock into the evoliez env. They
> are subprocess tools; isolated envs only. `server_smoke.sh` auto-prepends
> `$EVOLIEZ_ROOT/envs/boltz/bin` to PATH so the `boltz` call resolves there.

For a real full run, also point the config at your root:
`project.output_dir: /mnt/data/jglee/runs/fdh_nadp` and
`homologs.database: /mnt/data/jglee/evoliez_db/...` in
`configs/server_fdh_nadp.yaml` (the smoke steps already pass `--output-dir`).

## Configure a real run

Copy `configs/example_fdh_nadp.yaml`, then:

```yaml
project:
  output_dir: /mnt/data2/<you>/runs/fdh_nadp   # NOT under / or ~
backend: real
homologs:
  database: /mnt/data2/<you>/evoliez_db/uniref30_2302/uniref30_2302_db
# or: msa: { remote_server: true }  to skip the local DB entirely
```

Per-stage backend mixing is supported, e.g. keep MD mocked while testing Boltz:

```yaml
backend: real
backends: { s10_md: mock }
```

## Preflight (run first on the server)

```bash
evoliez doctor -c configs/server_fdh_nadp.yaml
```

Reports tools / Python deps / CUDA / GPU free VRAM / disk / `/mnt/data2`
writability / weights env vars / config sanity. Fix any `[BLOCK]` before a
real run; `[MISS]` on an optional tool just means that stage falls back to
mock.

## Staged validation ladder (do not big-bang)

| Step | Command | Validates |
|---|---|---|
| 0 | `evoliez run -c configs/example_fdh_nadp.yaml` (mock) | install integrity |
| 1 | `... --backend real --dry-run` | exact tool commands, no execution |
| 2 | `backends: {s04_complex: real}`, `--to s06b_interaction` | Boltz-2 + parser, 1 GPU |
| 3 | `--from s05_docking` + docking/FoldX real | redock / stability parsers |
| 4 | `backends: {s10_md: real}` | OpenMM-CUDA MD-lite |
| 5 | `configs/server_fdh_nadp.yaml` full real | end-to-end + datasets |
| 6 | `evoliez train-gnn` → `torchrun --nproc_per_node=4` | GNN 1-GPU → DDP |
| 7 | `evoliez bench --benchmark <known mutations>` | quantitative metrics |

`--resume` skips completed stages, so iterate per-stage cheaply.

## Run

```bash
# preview the exact tool commands without executing anything
evoliez run -c my_config.yaml --backend real --dry-run

# real run, GPU auto-pinned
bash scripts/run_pipeline.sh my_config.yaml

# resume after interruption (completed stages are skipped)
bash scripts/run_pipeline.sh my_config.yaml --resume
```

## Smoke test (recommended): one-shot script

Don't hand-run the ladder above — `scripts/server_test.sh` does it for you. It
activates the conda env, makes the editable install current with the pulled
source, runs the unit suite in the server env, then the STAGED real-backend
smoke (doctor → dryrun → boltz → dock → md → gnn), stopping on the first
failure. Idempotent; no sudo; nothing on root `/`.

```bash
cd "$EVOLIEZ_ROOT/EvoLiEZ"
git fetch origin && git checkout feat/server-hardening && git pull --ff-only
export EVOLIEZ_ROOT                       # already in ~/.bashrc

bash scripts/server_test.sh               # unit suite + full staged smoke
bash scripts/server_test.sh quick         # just the unit suite (no GPU)
bash scripts/server_test.sh boltz         # one stage: doctor|dryrun|boltz|dock|md|gnn
```

The `md` rung asserts real MD actually ran for ≥1 candidate (a degraded stage
must not pass as OK). Steps 2–4 auto-capture real tool outputs as fixtures —
see [`SERVER_SMOKE.md`](SERVER_SMOKE.md) for what each step proves and how to
send the fixtures back. Only after all steps pass green do a bounded full real
run via `scripts/run_pipeline.sh`, then scale.

## Server-grade GNN + Snakemake

`configs/server_fdh_nadp.yaml` enables the EvoLigand-GNN and `/mnt/data2`
storage. After the pipeline builds the graph dataset:

```bash
evoliez train-gnn -c configs/server_fdh_nadp.yaml          # 1 GPU
# DDP across 4 A6000:
torchrun --nproc_per_node=4 -m evoliez.ml.train_gnn ...    # gnn.ddp: true
evoliez run -c configs/server_fdh_nadp.yaml --from s08_reranker --resume
# or end-to-end:
snakemake -s workflow/Snakefile --config cfg=configs/server_fdh_nadp.yaml \
          --cores 32 --resources gpu=4
```

Install the GNN deps in the server env: `pip install -e ".[gnn]"`.
Full design: [`SERVER_GRADE.md`](SERVER_GRADE.md).

## Outputs

Everything under `project.output_dir`:
`reports/final_report.md`, `reports/final_candidates.csv`,
`reports/focused_library.csv`, `evoliez.sqlite`, plus per-stage artifacts
(spec §17.2 layout). `_state.json` tracks completed stages for `--resume`.
