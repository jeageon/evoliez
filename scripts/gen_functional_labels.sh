#!/usr/bin/env bash
# One-shot functional-state LABEL generator (server).
#
# Anchored-build + cheap md_lite (NAC/RBFE/binding-ΔG OFF) + pose_gate over s07 candidates
# of a finished run, to produce reference_like / alternative / displaced labels for the
# MD-free triage ML — including the displaced candidates that never reach s10 (where the
# learnability probe found ZERO negatives).
#
# Env preamble mirrors scripts/run_s10_paper.sh (the proven s10 MD launcher): conda env
# evoliez-cu124 + AmberTools25 (antechamber/sqm for the ligand FF step) + per-tool
# interpreters. Uses GUARD_PY because the PATH prepend shadows `python`.
#
# SAFETY (touches a FINISHED run): preflight scripts/check_run_fingerprint.py (the same
# purge-guard run_s10_paper.sh uses) ABORTS on any config mismatch; the driver itself also
# aborts if ctx.invalidated; MD overrides are in-memory; output is a NEW provenance file;
# all MD scratch under <run>/labels_md/. EVOLIEZ_ALLOW_PURGE is never set -> no deletion.
#
# Usage:
#   bash scripts/gen_functional_labels.sh                          # smoke: 3 cand, 1 GPU
#   LIMIT=150 GPUS=0,2,3 CUDA_VISIBLE_DEVICES=0,2,3 bash scripts/gen_functional_labels.sh
#   LIMIT=0   GPUS=0,2,3 CUDA_VISIBLE_DEVICES=0,2,3 bash scripts/gen_functional_labels.sh  # ALL
set -euo pipefail

# OpenMM + spawn-based GPU fan-out opens many file handles per MD; the default soft
# limit (1024) is exhausted after ~2 chunks. Raise it (hard limit is ~1M here).
ulimit -n 65536 || true

source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
GUARD_PY="$CONDA_PREFIX/bin/python"          # capture BEFORE PATH prepend shadows `python`

# AmberTools25 (antechamber/sqm/tleap) for the ligand force-field / charge step
AMBERTOOLS_ENV=/mnt/data/jglee/anaconda3/envs/AmberTools25
export AMBERHOME=$AMBERTOOLS_ENV
export EVOLIEZ_AMBERTOOLS_BIN=$AMBERTOOLS_ENV/bin
DD=/mnt/data/jglee/envs/diffdock
export PATH=/mnt/data/jglee/bin:$DD/bin:/mnt/data/jglee/envs/boltz/bin:$AMBERTOOLS_ENV/bin:$PATH
export LD_LIBRARY_PATH=$DD/lib:$AMBERTOOLS_ENV/lib:${LD_LIBRARY_PATH:-}
# per-tool interpreters (only needed if a load() re-touches a tool; harmless otherwise)
export EVOLIEZ_LIGANDMPNN_PYTHON=/mnt/data/jglee/envs/ligandmpnn/bin/python
export EVOLIEZ_THERMOMPNN_PYTHON=/mnt/data/jglee/miniforge3/envs/thermompnn/bin/python
export WANDB_MODE=disabled
export EVOLIEZ_ROOT=/mnt/data/jglee

# Shared-server etiquette (memory shared-server-constraints): bound threads so the
# 48-core/10-min auto-throttle never trips; leave GPU1 for others.
THREADS="${THREADS:-4}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENMM_CPU_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"

CONFIG="${CONFIG:-configs/target.local.v2.yaml}"
RUN_DIR="${RUN_DIR:-/mnt/data/jglee/runs/fdh_v2}"
GPUS="${GPUS:-}"                                       # smoke: empty -> serial, 1 GPU
LIMIT="${LIMIT:-3}"                                    # smoke: 3 candidates
NS="${NS:-0.2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd /mnt/data/jglee/EvoLiEZ
echo "[env] py=$GUARD_PY gpus='${GPUS:-serial}' cuda=$CUDA_VISIBLE_DEVICES limit=$LIMIT ns=$NS threads=$THREADS"
for t in antechamber sqm tleap; do
  command -v "$t" >/dev/null 2>&1 && echo "  $t: OK" || { echo "  !! $t MISSING -> ligand FF will fail"; exit 3; }
done

echo "=== preflight: purge-safe fingerprint guard ==="
if ! "$GUARD_PY" scripts/check_run_fingerprint.py "$CONFIG" "$RUN_DIR"; then
  echo "   !!! fingerprint MISMATCH -- aborting (config no longer matches $RUN_DIR)."
  echo "   patch with: $GUARD_PY scripts/check_run_fingerprint.py $CONFIG $RUN_DIR --patch"
  exit 1
fi
echo "=== fingerprint OK -- generating labels (no purge; new file only) ==="

EXTRA=()
[ -n "${CANDIDATE_IDS:-}" ] && EXTRA+=(--candidate-ids "$CANDIDATE_IDS")
[ -n "${OUT:-}" ] && EXTRA+=(--out "$OUT")
"$GUARD_PY" scripts/gen_functional_labels.py \
    --config "$CONFIG" --run-dir "$RUN_DIR" \
    --gpus "$GPUS" --limit "$LIMIT" --production-ns "$NS" "${EXTRA[@]}"

echo "[ok] labels -> $RUN_DIR/reports/provenance/functional_labels.json"
