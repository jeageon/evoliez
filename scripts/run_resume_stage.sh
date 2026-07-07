#!/usr/bin/env bash
# Resume a REAL GPU pipeline run on the SHARED lab server, bounded to <=50% of
# the box (user constraint). Boltz runs from its isolated env; pins exactly ONE
# GPU (1 of 4 = 25%); caps CPU threads (well under the 48-core watchdog). Safe to
# re-run: evoliez --resume skips completed stages.
#
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_resume_stage.sh configs/target.local.5track.yaml s06b_interaction
#
# Arg1 = config YAML, Arg2 = optional --to stage (omit to run to the end).
# Caller MUST preset CUDA_VISIBLE_DEVICES to a free GPU index (nvidia-smi is
# broken here; pick via torch.cuda.mem_get_info).
set -euo pipefail

EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data/jglee}"
CONFIG="${1:?usage: run_resume_stage.sh <config.yaml> [to_stage]}"
TO_STAGE="${2:-}"
CONDA_BASE="${CONDA_BASE:-$EVOLIEZ_ROOT/miniforge3}"
BOLTZ_ENV="${BOLTZ_ENV:-$EVOLIEZ_ROOT/envs/boltz}"

: "${CUDA_VISIBLE_DEVICES:?preset CUDA_VISIBLE_DEVICES to a free GPU index}"

# --- shared-server CPU caps: <=8 threads (of 48) so we stay far under 50% and
#     never trip the 48-core/10-min auto-throttle watchdog. -----------------
N="${EVOLIEZ_NUM_THREADS:-8}"
export EVOLIEZ_NUM_THREADS="$N" OMP_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N" \
       MKL_NUM_THREADS="$N" NUMEXPR_NUM_THREADS="$N" NUMEXPR_MAX_THREADS="$N" \
       VECLIB_MAXIMUM_THREADS="$N"

# --- orchestrator env (the pipeline itself) ---
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate evoliez-cu124
# --- isolated Boltz on PATH (bin first so `boltz` resolves to the cu124 env) ---
export PATH="$BOLTZ_ENV/bin:$PATH"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$EVOLIEZ_ROOT/evoliez_assets/boltz_cache}"

cd "$EVOLIEZ_ROOT/EvoLiEZ"
echo ">> $(date '+%F %T') start: GPU=$CUDA_VISIBLE_DEVICES threads=$N to='${TO_STAGE:-<end>}'"
echo ">> evoliez=$(command -v evoliez)  boltz=$(command -v boltz)  BOLTZ_CACHE=$BOLTZ_CACHE"
ARGS=(run -c "$CONFIG" --resume)
[ -n "$TO_STAGE" ] && ARGS+=(--to "$TO_STAGE")
exec evoliez "${ARGS[@]}"
