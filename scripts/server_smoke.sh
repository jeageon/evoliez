#!/usr/bin/env bash
# Staged real-backend smoke test (expert plan). Run ONE step at a time on the
# GPU server; each step is small and stops on first failure so debugging is
# localised, not a full-run debugging hell. Real tool outputs are captured as
# fixtures (scripts/capture_fixtures.sh) for parser regression tests.
#
#   bash scripts/server_smoke.sh <step> [config]
#     step = doctor | dryrun | boltz | dock | md | gnn | all
#     config defaults to configs/server_fdh_nadp.yaml
set -euo pipefail

# Shared 48-core server (SERVER_RUNBOOK #2): bound BLAS / numexpr / xgboost
# thread pools for the whole process tree so the per-pose/per-candidate loops
# in s06b/s08 don't oversubscribe and thrash (was 5-6 min of pure context
# switching). Raise with EVOLIEZ_NUM_THREADS=N if you own the box.
export EVOLIEZ_NUM_THREADS="${EVOLIEZ_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$EVOLIEZ_NUM_THREADS}"
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-$EVOLIEZ_NUM_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$EVOLIEZ_NUM_THREADS}"

STEP="${1:?usage: server_smoke.sh <doctor|dryrun|boltz|dock|md|gnn|all> [smoke_config]}"
# Two INDEPENDENT configs (do not collapse them):
#  - REAL_CFG : the real-run config `doctor` validates. Env-overridable ONLY,
#    NEVER from arg2 - otherwise `all` would pass the smoke config in and make
#    doctor validate the wrong thing.
#  - SMOKE_CFG: the tiny config the run steps use (arg2 overrides it).
REAL_CFG="${REAL_CFG:-configs/server_fdh_nadp.yaml}"
SMOKE_CFG="${2:-configs/smoke.yaml}"
export REAL_CFG SMOKE_CFG   # so an env override survives the `all` sub-calls
EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/${USER}}"
RUN="$EVOLIEZ_ROOT/runs/smoke"

# Boltz lives in its OWN env (pins numpy<2 etc; must NOT pollute evoliez).
# Prepend it so the `boltz` subprocess resolves there while `evoliez` runs
# from the evoliez env.
BOLTZ_ENV="${BOLTZ_ENV:-$EVOLIEZ_ROOT/envs/boltz}"
if [ -x "$BOLTZ_ENV/bin/boltz" ]; then
  export PATH="$BOLTZ_ENV/bin:$PATH"
  export BOLTZ_CACHE="${BOLTZ_CACHE:-$EVOLIEZ_ROOT/evoliez_assets/boltz_cache}"
  echo ">> using isolated Boltz: $BOLTZ_ENV/bin/boltz"
fi

# Pick a GPU that is ACTUALLY free, not merely idle-but-VRAM-allocated.
# Delegates to evoliez.utils.gpu.select_gpu (filters by free VRAM, then
# least-busy / most-free, with logging) instead of the old util-only
# nvidia-smi sort that could land on a 0%%-util GPU another user has
# fully VRAM-allocated -> Boltz-2 OOM. Threshold is Boltz-sized and
# overridable: EVOLIEZ_GPU_MIN_FREE_MIB (default 20000 = ~20 GiB).
pin_gpu() {
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && {
    echo ">> CUDA_VISIBLE_DEVICES preset to $CUDA_VISIBLE_DEVICES; honoring it"
    return 0; }
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local minf="${EVOLIEZ_GPU_MIN_FREE_MIB:-20000}"
  local idx
  idx="$(python -c "from evoliez.utils.gpu import select_gpu; i=select_gpu($minf); print('' if i is None else i)" 2>/dev/null || true)"
  if [ -n "$idx" ]; then
    export CUDA_VISIBLE_DEVICES="$idx"
    echo ">> pinned GPU $idx (selected for >= ${minf} MiB free; see gpu log)"
  else
    echo ">> WARNING: GPU auto-select found none with >= ${minf} MiB free;" \
         "not pinning - a real stage may OOM or contend. Set" \
         "CUDA_VISIBLE_DEVICES manually or lower EVOLIEZ_GPU_MIN_FREE_MIB."
  fi
}

case "$STEP" in
doctor)
  evoliez doctor -c "$REAL_CFG"     # validate the real-run config
  ;;
dryrun)   # step 2: command preview, no execution, no GPU
  evoliez run -c "$SMOKE_CFG" --backend real --dry-run --output-dir "$RUN/dry"
  ;;
boltz)    # step 3: ONLY Boltz real (homolog/MSA stay mock -> synthetic feed,
          # no homolog DB needed). Minimal real surface, stop at s04.
  pin_gpu
  export BOLTZ_CACHE="${BOLTZ_CACHE:-$EVOLIEZ_ROOT/evoliez_assets/boltz_cache}"
  mkdir -p "$BOLTZ_CACHE"
  echo ">> BOLTZ_CACHE=$BOLTZ_CACHE"
  evoliez run -c "$SMOKE_CFG" --backend mock --output-dir "$RUN/boltz" \
    --stage-backend s04_complex=real --to s04_complex
  bash scripts/capture_fixtures.sh "$RUN/boltz"
  ;;
dock)     # step 4: ONLY docking real (Vina). FoldX is academic/optional so
          # s09 stays mock here; isolate the docking parser/prep.
  pin_gpu
  evoliez run -c "$SMOKE_CFG" --backend mock --output-dir "$RUN/dock" \
    --stage-backend s05_docking=real --to s05_docking
  bash scripts/capture_fixtures.sh "$RUN/dock"
  ;;
md)       # step 5: OpenMM minimise/MD-lite real (protocol level via config)
  pin_gpu
  evoliez run -c "$SMOKE_CFG" --backend mock --output-dir "$RUN/md" \
    --stage-backend s10_md=real
  bash scripts/capture_fixtures.sh "$RUN/md"
  ;;
gnn)      # step 6: 1-epoch single-GPU GNN train (DDP comes later)
  pin_gpu
  evoliez run -c "$SMOKE_CFG" --backend mock --output-dir "$RUN/gnn" >/dev/null
  evoliez train-gnn -c "$SMOKE_CFG" --epochs 1 \
    --dataset "$RUN/gnn/datasets/graph_pt"
  ;;
all)
  for s in doctor dryrun boltz dock md gnn; do
    echo "==== smoke step: $s ===="
    bash "$0" "$s" "$SMOKE_CFG"
  done
  ;;
*) echo "unknown step '$STEP'"; exit 1 ;;
esac
echo ">> smoke step '$STEP' OK"
