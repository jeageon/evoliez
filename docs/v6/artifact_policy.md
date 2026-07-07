# V6-7 — Artifact Policy

Where every V6 artifact lives, and why. The rule: **small, human-meaningful
provenance is version-controlled; large regenerable simulation data is not.**

## Version-controlled (in this Git repo)

| Class | Path | Rationale |
|---|---|---|
| Source | `src/evoliez/**` | the code |
| Configs | `configs/*.yaml` | schema-validated, part of the reproducibility bundle |
| Curated parameters | `params/*/` (small mol2/sdf) | e.g. `params/atp_4minus/ATP.fixed.mol2` — inputs, not regenerable cheaply |
| **Provenance records** | `reports/provenance/*.json`, `*.jsonl` | runtime profile, system manifests, per-run job records — the reproducibility spine |
| Docs | `docs/**`, `outputs/car_v6/*` | reports, portfolios, evidence cards (small, claim-gated) |
| Tests | `tests/**` | CI |

The `.gitignore` carve-out makes this explicit: `/reports/*` is ignored (regenerable
run reports) **except** `!/reports/provenance/` — provenance is not regenerable and
documents a specific server run.

## NOT version-controlled (the approved artifact location)

Large, regenerable simulation data **must not** be committed:

| Class | Example | Size | Lives on |
|---|---|---|---|
| Trajectories | `prod_*.nc`, umbrella windows | GBs (v6_md_work ≈ 7.8 GB) | server `EVOLIEZ_ROOT` work dirs |
| Topologies/restarts | `complex.prmtop` (14 MB), `*.rst` | 10s of MB each | server work dirs |
| Model weights | Boltz `.ckpt` (2 GB each) | GBs | server `$BOLTZ_CACHE` (`.boltz/`) |
| Full run dirs | `runs/*` | 16 GB (a CAR run) | server + **Google Drive** archive (`EvoLiEZ_runs/…`) |

`.gitignore` already excludes `runs/`, `reports/`, `reports_v3/`, `fdh/`, and the
large data classes. **Deliverable runs are archived to Google Drive** (e.g. the
SrCAR→3-HP run → `GDrive EvoLiEZ_runs/car`); the repo keeps only the provenance JSON
that points at them.

## Regenerating an artifact

Any large artifact is reproducible from the version-controlled bundle:
`git commit + config + params + runtime profile` → re-run the relevant
`scripts/run_amber_*.sh` / `run_*_amber.py` on the server. The manifests'
fingerprints confirm an identical rebuild.

## Server hygiene (shared box)

- Work dirs under `/mnt/data/jglee/v6_*` are scratch; prune after archiving.
- Never monopolize CPU/GPU/disk; keep heavy jobs within the CPU budget; one pmemd
  per GPU.
- Archive → prune → keep the provenance JSON in Git.
