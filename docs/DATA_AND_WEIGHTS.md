# Data, weights & external tools

None of these are bundled (license + size). Acquire on the server, store on
`/mnt/data2`. The pipeline detects missing tools and either errors with a
clear message or degrades to the mock backend (spec §23 risk mitigations).

## Model weights / code

| Tool | Source | Notes |
|---|---|---|
| Boltz / Boltz-2 | https://github.com/jwohlwend/boltz | `pip install boltz`; weights auto-download — set `BOLTZ_CACHE` to a `/mnt/data2` path so they do not land on root |
| LigandMPNN | https://github.com/dauparas/LigandMPNN | clone + `get_model_params.sh`; set `EVOLIEZ_LIGANDMPNN` |
| DiffDock | https://github.com/gcorso/DiffDock | optional docking backend |
| GNINA | https://github.com/gnina/gnina | CUDA build; put binary on PATH |
| AutoDock Vina | https://vina.scripps.edu/ | `conda install -c conda-forge vina`; CPU |

`scripts/fetch_weights.sh` automates the first three onto `/mnt/data2`.

## Licensed tools

| Tool | License |
|---|---|
| FoldX | Academic license required — https://foldxsuite.crg.eu/ . Put `foldx` on PATH; selected by `validation.stability.method: foldx` |
| Rosetta | License required — https://rosettacommons.org/ . `cartesian_ddg` binary on PATH; `validation.stability.method: rosetta` |

If neither is available, set `validation.stability.method: ml` (or run that
stage with `mock`) to use the chemistry-informed ΔΔG proxy.

## Sequence databases (optional)

Local homolog search needs a DB (UniRef30 ≈120 GB unpacked). On a near-full
disk, prefer the hosted MSA service instead:

```yaml
msa: { remote_server: true }     # uses api.colabfold.com, no local DB
```

Otherwise: `scripts/fetch_databases.sh uniref30` (downloads to
`$EVOLIEZ_DB_DIR` on `/mnt/data2`), then set
`homologs.database: /mnt/data2/<you>/evoliez_db/uniref30_2302/uniref30_2302_db`.

## Environment variables

| Var | Meaning |
|---|---|
| `EVOLIEZ_ENV_PREFIX` | conda env location (must be `/mnt/data2/...`) |
| `EVOLIEZ_DATA_DIR` | weights/assets dir on `/mnt/data2` |
| `EVOLIEZ_DB_DIR` | sequence DB dir on `/mnt/data2` |
| `BOLTZ_CACHE` | Boltz weight cache (point at `/mnt/data2`) |
| `CUDA_VISIBLE_DEVICES` | leave unset — the pipeline auto-pins a free GPU |
