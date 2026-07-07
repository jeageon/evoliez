# Server real-backend smoke runbook

Do NOT run a full real pipeline first — it becomes a debugging hell. Run the
staged smoke (`scripts/server_smoke.sh <step>`), one step at a time, smallest
real surface first. Server execution is a **real-backend validation-data
collection** step: each tool step auto-captures real outputs as fixtures.

Prereqs (once): clone the branch onto `/mnt/data2`, create the env, fetch
weights — see `SERVER_RUNBOOK.md`.

| Step | Command | What it proves / collect |
|---|---|---|
| 0 | `bash scripts/server_smoke.sh doctor` | tools/CUDA/`/mnt/data2`/config sane — STRICT (exit 1 on any `[BLOCK]`); fix every `[BLOCK]` before a real run |
| 1 | `bash scripts/server_smoke.sh dryrun` | every real tool command previews (no GPU). Already works on the default config (fixed in `119f65b`) |
| 2 | `bash scripts/server_smoke.sh boltz` | homolog+MSA+**Boltz** real, stop at s04. Checks Boltz out path, mmCIF/PDB parse, confidence/affinity JSON, **atom-index lock** |
| 3 | `bash scripts/server_smoke.sh dock` | s05+s09 real on few candidates: Vina ligand pdbqt prep, GNINA/DiffDock pose parse, FoldX/Rosetta input |
| 4 | `bash scripts/server_smoke.sh md` | OpenMM real (set `validation.md.protocol_level: 0` or `1` in config first). Ligand parameterization is the riskiest part |
| 5 | `bash scripts/server_smoke.sh gnn` | graph dataset + `train-gnn --epochs 1` single-GPU (DDP later) |

`--stage-backend STAGE=real` flips just one stage to real (everything else
mock) so each step has a minimal real surface; `--to/--from` bound the run.

> Note: the `all` ladder (and `scripts/server_test.sh`) runs step 0 doctor in
> ADVISORY mode — it surfaces every check but does NOT abort plumbing
> validation on the expected placeholder-target `[BLOCK]`, so a fresh clone can
> exercise the tool integration. The standalone `server_smoke.sh doctor` stays
> strict (exit 1) — run it once you've set a real target, before a real run.

## After steps 2–4: send the captured fixtures back

`scripts/capture_fixtures.sh` copies the first real output of each format to
`tests/fixtures/tool_outputs/captured/` (Boltz cif/json/npz, Vina pdbqt,
GNINA/DiffDock sdf, homolog m8/sto, FoldX fxout, Rosetta ddg) + `MANIFEST.txt`.

Commit + share that directory (or paste a step's stderr/stdout on failure).
Then the real parsers get hardened against the actual formats:
`tests/test_tool_output_parsers.py` gains `*_real.*` cases, and the
quantitative tool-validation metrics are wired:

- `ml.pose_validity.pose_rmsd` — ligand RMSD vs reference (success ≤ 2 Å)
- `ml.pose_validity.plif_recovery` — interaction-fingerprint recovery
- `ml.pose_validity.pose_sanity` — clash / in-pocket (PoseBusters stand-in)

## Iteration loop

```
run step -> fails? paste log  ->  fix adapter/parser + add *_real fixture test
run step -> ok?    capture_fixtures.sh -> commit -> next step
```

Only after steps 0–5 pass green do a bounded full real run, then scale
(diffusion samples, candidates, DDP).
