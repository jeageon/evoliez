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

```bash
# clone INTO /mnt/data2 (not home/root)
mkdir -p /mnt/data2/$USER && cd /mnt/data2/$USER
git clone <your-fork-url> EvoLiEZ && cd EvoLiEZ

bash scripts/setup_server_env.sh          # conda env at /mnt/data2/$USER/envs/evoliez
conda activate /mnt/data2/$USER/envs/evoliez
pip install -e .

bash scripts/fetch_weights.sh             # Boltz-2 / LigandMPNN -> /mnt/data2
# optional, large: local homolog DB. Otherwise set msa.remote_server: true
bash scripts/fetch_databases.sh uniref30
```

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

## Run

```bash
# preview the exact tool commands without executing anything
evoliez run -c my_config.yaml --backend real --dry-run

# real run, GPU auto-pinned
bash scripts/run_pipeline.sh my_config.yaml

# resume after interruption (completed stages are skipped)
bash scripts/run_pipeline.sh my_config.yaml --resume
```

## Smoke test (recommended order)

1. `evoliez run -c configs/example_fdh_nadp.yaml` (backend mock) — proves the
   install end-to-end, ~seconds.
2. Same config, `backend: real`, `backends: { s10_md: mock }`, `--to s04_complex`
   — exercises Boltz-2 on one A6000 only.
3. Add `s10_md: real`, `--from s09_nonmd` — exercises OpenMM-CUDA MD-lite on a
   handful of candidates.
4. Full real run.

## Outputs

Everything under `project.output_dir`:
`reports/final_report.md`, `reports/final_candidates.csv`,
`reports/focused_library.csv`, `evoliez.sqlite`, plus per-stage artifacts
(spec §17.2 layout). `_state.json` tracks completed stages for `--resume`.
