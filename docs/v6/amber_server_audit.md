# V6-0 — Amber Capability Audit (server-verified)

**Phase:** V6-0 · **Status:** ✅ PASS · **Server:** `admin1-SYS-740GP-TNRT` (evo, 143.248.33.117:8866)
**Runtime profile:** [`reports/provenance/amber_runtime_profile.json`](../../reports/provenance/amber_runtime_profile.json)
**Probe:** [`scripts/check_amber_gpu.py`](../../scripts/check_amber_gpu.py) · **Runner:** [`scripts/run_amber_audit.sh`](../../scripts/run_amber_audit.sh)

## Goal

Establish whether the shared GPU server can reliably support Amber-native
production runs: verify the tool chain, prove a real
`tleap → prmtop/inpcrd → pmemd.cuda → cpptraj` round-trip on the GPU with **no
silent CPU fallback**, and write a reproducible runtime profile for every run.

## Verdict

> **PASS: GPU round-trip OK on NVIDIA RTX A6000 (implicit + explicit-PME).**

Both the implicit-GB and the explicit-solvent PME paths build, run on the GPU,
and analyze cleanly. This unblocks V6-1 (Amber system builder) and V6-2 (Amber
GPU MD executor).

## Hardware / driver

| Item | Value |
|---|---|
| GPUs | 4 × NVIDIA RTX A6000 (48 GB each), all idle at audit time |
| CUDA driver | 550.163.01 |
| CUDA toolkits present | `/usr/local/cuda-12.4` (+ `cuda`, `cuda-12` symlinks) |
| CPU / RAM | 48 cores / 251 GB |

`pmemd.cuda_SPFP` was built against CUDA 12.4 and matches this driver — GPU MD
runs natively (unlike the OpenMM tier, which hits a PTX mismatch on this driver
and falls back to OpenCL; see `[[s10-openmm-torch-cuda-ptx]]`). This is a key
reason V6 makes Amber the production MD path.

## Tool inventory (all present)

AmberTools live in the `evoliez` prefix env; `pmemd.cuda` in the separate
`pmemd24` build:

| Tool | Location |
|---|---|
| `tleap`, `parmed`, `antechamber`, `parmchk2`, `pdb4amber` | `/mnt/data/jglee/envs/evoliez/bin` |
| `cpptraj`, `sander`, `sqm`, `MMPBSA.py`, `MCPB.py` | `/mnt/data/jglee/envs/evoliez/bin` |
| `quick` (QM/MM / semiempirical) | `/mnt/data/jglee/envs/evoliez/bin` |
| `pmemd.cuda_SPFP` (+ `_DPFP`, `.MPI`) | `/mnt/data/jglee/pmemd24/bin` |

Notes:
- **`MCPB.py` present** → metal-site parameterization (V6-1 Mg) has a real backend.
- **`quick` present** → V6-5 QM/MM-lite has a real electronic-structure backend
  on the server (no need to mock).

## Canonical runtime recipe (reproducible)

`scripts/run_amber_audit.sh` encodes the exact env; the essence:

```bash
# 1) AmberTools from the evoliez prefix (RPATH-linked; provides leaprc)
source /mnt/data/jglee/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/data/jglee/envs/evoliez
# 2) pmemd.cuda runtime libs
source /mnt/data/jglee/pmemd24/amber.sh
# 3) point the pipeline at the right binaries
export AMBERHOME=/mnt/data/jglee/envs/evoliez
export EVOLIEZ_AMBERTOOLS_BIN=$AMBERHOME/bin
export EVOLIEZ_PMEMD_CUDA=/mnt/data/jglee/pmemd24/bin/pmemd.cuda_SPFP
export CUDA_VISIBLE_DEVICES=<one idle gpu>
```

`amber_engine.py` reads `EVOLIEZ_AMBERTOOLS_BIN` / `EVOLIEZ_PMEMD_CUDA`, so the
production engine and this audit resolve the same tools.

## Round-trip results

Both systems are built from scratch by `tleap` (no external inputs), minimized +
short-MD'd on `pmemd.cuda_SPFP`, then analyzed by `cpptraj`. GPU execution is
confirmed by parsing pmemd's `GPU DEVICE INFO` block (device id + name + mem) —
`pmemd.cuda` cannot run on CPU, so a clean run *with* that block is a positive
no-fallback proof.

| Mode | System | Atoms | Wall | GPU | cpptraj final RMSD |
|---|---|---|---|---|---|
| implicit GB (igb=8) | ACE-(ALA)₃-NME | 42 | 0.89 s | [0] RTX A6000 | 0.506 Å |
| explicit PME (NVT) | ACE-(ALA)₃-NME + TIP3P (925 waters) | 9 924 | 1.86 s | [0] RTX A6000 | 0.225 Å |

Input decks are content-hashed into the profile (`input_sha1`) so an identical
re-run is verifiable.

## Failure taxonomy (validated live)

The probe classifies any failure as `environment` / `license` /
`parameterization` / `gpu` / `input-preparation`. This was exercised for real
during the audit:

- **First explicit-PME run FAILED**, initially mis-tagged `gpu`. Root cause was
  **not** the GPU: `pmemd.cuda`'s `gpu_neighbor_list_setup` rejects "small boxes"
  (≤ 2 neighbor-list hash cells per dimension). The tiny audit peptide with a
  10 Å solvent buffer produced a 30 Å-thin box.
- **Fix (this phase):** (1) the audit's explicit test now uses an 18 Å buffer so
  the box clears the guard on the *real* GPU PME path (not the `-AllowSmallBox`
  workaround); production systems — a ~500-residue A-domain — are far above the
  threshold anyway. (2) the classifier now recognizes the `Small box detected`
  signature as `input-preparation`, so the audit never falsely condemns the GPU.

This is exactly the diagnostic discipline V6 requires: a signal that
pattern-matched "GPU error" had a different, benign, system-prep cause.

## Success conditions — checklist

- [x] `tleap → prmtop/inpcrd → pmemd.cuda → cpptraj` round-trip succeeds on a small system (implicit **and** explicit-PME).
- [x] GPU execution confirmed without silent CPU fallback (GPU DEVICE INFO block parsed; `.cuda` binary).
- [x] Failures classified into the five-class taxonomy (validated on the real small-box failure).
- [x] A reproducible Amber runtime profile is written for the run (`amber_runtime_profile.json`, with deck hashes).

## Known follow-ups (carried forward)

- `git_commit` was `null` in the server run (the `EvoLiEZ_car` checkout's HEAD
  did not resolve from the scripts dir). Captured at commit time here; V6-7 will
  ensure every server run stamps the driving commit.
- The audit uses a single GPU. Multi-GPU independent-job scheduling is proven in
  V6-2 (`amber_scheduler.py`).
