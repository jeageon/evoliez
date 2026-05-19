#!/usr/bin/env bash
# Run the pipeline on the server, auto-pinning to the least-busy free GPU.
#
# Usage:  bash scripts/run_pipeline.sh configs/example_fdh_nadp.yaml [extra evoliez args]
set -euo pipefail

# Shared 48-core server (SERVER_RUNBOOK #2): bound BLAS / numexpr / xgboost
# thread pools so we don't oversubscribe the box and thrash other users'
# jobs. Raise with EVOLIEZ_NUM_THREADS=N if you own the machine.
export EVOLIEZ_NUM_THREADS="${EVOLIEZ_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$EVOLIEZ_NUM_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$EVOLIEZ_NUM_THREADS}"

CONFIG="${1:?usage: run_pipeline.sh <config.yaml> [args...]}"
shift || true
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pick the GPU with the most free VRAM and lowest utilisation (shared server).
if command -v nvidia-smi >/dev/null 2>&1 && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | sort -t, -k3 -n -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo ">> pinned to GPU $GPU"
fi
echo ">> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<none>}"

# Default real outputs to /mnt/data2 unless the config already does.
EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/$USER}"
export EVOLIEZ_DATA_DIR="${EVOLIEZ_DATA_DIR:-$EVOLIEZ_ROOT/evoliez_assets}"

cd "$REPO_DIR"
exec evoliez run -c "$CONFIG" "$@"
