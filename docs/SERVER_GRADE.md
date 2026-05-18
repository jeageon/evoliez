# Server-grade architecture (4× RTX A6000)

The lab server runs the full research platform, not a Colab MVP. This doc
captures the server-grade design; see `configs/server_fdh_nadp.yaml` and
`workflow/Snakefile`.

## Resource model

- **GPU**: 4× RTX A6000 (~49 GB each), CUDA 12.4. Strategy = **independent job
  per GPU** (Boltz / DiffDock / LigandMPNN+GNN-train / MD), *not*
  model-parallel across GPUs. Training uses DDP (`gnn.ddp: true` +
  `torchrun --nproc_per_node=4`). `utils.gpu.GpuPool` / `free_gpu_indices`
  hand out free GPUs; `workflow/Snakefile` declares `resources: gpu=1` per
  rule so Snakemake parallelises across the 4 cards.
- **CPU**: 48 logical cores → MMseqs2/HMMER/MAFFT/RDKit/FoldX batched
  multi-process.
- **Storage**: root `/` is full → everything under `/mnt/data2`
  (`output_dir: /mnt/data2/evoligand/runs/...`), archive on `/mnt/data`.
  `context._check_disk` refuses near-full filesystems.

```
/mnt/data2/evoligand/
  db/                 sequence DBs (UniRef30, …)
  runs/<project>/     boltz/ docking/ ligandmpnn/ foldx/ md/
                      interaction_graphs/ ml_datasets/ datasets/graph_pt/
                      checkpoints/ reports/ logs/
/mnt/data/evoligand_archive/   compressed trajectories/datasets
```

## Model stack

```
Boltz/Boltz-2 diffusion ensemble  (features, NOT labels - see ML_DATA_POLICY)
 + representative-homolog complexes
 + ligand-atom→protein relative-vector graph dataset
 + MSA evolutionary features
        ↓
EvoLigand-GNN  (ml/egnn.py) — E(3)-invariant heterogeneous EGNN
  nodes: ligand atoms + nearby residues
  edges: r_ij + |r_ij| + unit vector + RBF
  multitask: contact prob · interaction type · permissiveness ·
             native-residue recovery · graph mutation score
        ↓
family-specific reranker (xgboost/heuristic) keeps as baseline
        ↓
final ranking = GNN + family-interaction + Boltz confidence +
                redocking + FoldX ΔΔG + MD-lite − catalytic penalty
```

XGBoost/heuristic stays as the always-on baseline (the GNN must beat it).
The GNN is **optional**: no torch / no checkpoint → ranking falls back to the
family interaction model automatically (`gnn` weight 0 by default).

## Data-generation scale (`data_scale`)

- homolog representatives: 50–300 clustered, 20–100 Boltz complexes
- diffusion samples: target 20–50, homolog 5–20, mutant 5–20
- candidate funnel: LigandMPNN 1k–10k → redock 500 → stability 100 →
  MD-lite 20–50 → experimental 10–50

## Run order

```bash
bash scripts/setup_server_env.sh        # conda env on /mnt/data2 (+ gnn extra)
bash scripts/run_pipeline.sh configs/server_fdh_nadp.yaml   # build + datasets
evoliez train-gnn -c configs/server_fdh_nadp.yaml           # 1-GPU, or:
torchrun --nproc_per_node=4 -m evoliez.ml.train_gnn ...     # DDP (gnn.ddp:true)
evoliez run -c configs/server_fdh_nadp.yaml --from s08_reranker --resume
# or: snakemake -s workflow/Snakefile --config cfg=configs/server_fdh_nadp.yaml \
#               --cores 32 --resources gpu=4
```

## Roadmap (server-grade)

| Phase | Deliverable |
|---|---|
| 1 | server baseline pipeline (done) |
| 2 | Boltz diffusion-ensemble dataset + contact frequency (done) |
| 3 | tabular reranker baseline (done — keep) |
| 4 | EvoLigand-GNN relative-vector model (this change) |
| 5 | MD-lite final validation (done) |
| 6 | experimental-label supervised fine-tuning: frozen GNN + family adapter + task head |
