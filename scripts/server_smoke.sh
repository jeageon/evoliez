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

# Pick a GPU that is ACTUALLY free (enough free VRAM), not merely idle.
# PURE shell + nvidia-smi - NO python/evoliez import: this function runs
# after `$BOLTZ_ENV/bin` is prepended to PATH, so bare `python` resolves to
# the isolated Boltz env (no evoliez) and the old `python -c "import
# evoliez..."` silently ModuleNotFound'd -> false "no free GPU". Filter by
# free VRAM >= EVOLIEZ_GPU_MIN_FREE_MIB (default 20000), then least-busy /
# most-free; fall back to the most-free GPU with a warning.
pin_gpu() {
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && {
    echo ">> CUDA_VISIBLE_DEVICES preset to $CUDA_VISIBLE_DEVICES; honoring it"
    return 0; }
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local minf="${EVOLIEZ_GPU_MIN_FREE_MIB:-20000}"
  local q idx
  q="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
       --format=csv,noheader,nounits 2>/dev/null)"
  # candidates with free >= minf, ordered by util asc then free desc
  idx="$(echo "$q" | awk -F', *' -v m="$minf" \
       '($2+0)>=m {print ($3+0), -($2+0), $1}' \
       | sort -k1,1n -k2,2n | head -1 | awk '{print $3}')"
  if [ -n "$idx" ]; then
    export CUDA_VISIBLE_DEVICES="$idx"
    echo ">> pinned GPU $idx (>= ${minf} MiB free)"
    return 0
  fi
  # nothing meets the floor: take the absolute most-free, but warn
  idx="$(echo "$q" | awk -F', *' '{print ($2+0), $1}' \
       | sort -k1,1nr | head -1 | awk '{print $2}')"
  if [ -n "$idx" ]; then
    export CUDA_VISIBLE_DEVICES="$idx"
    echo ">> WARNING: no GPU with >= ${minf} MiB free; using most-free GPU" \
         "$idx (may contend). Lower EVOLIEZ_GPU_MIN_FREE_MIB or wait."
  else
    echo ">> WARNING: nvidia-smi returned no GPUs; not pinning."
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
md)       # step 5: real OpenMM MD-lite. MUST also run s04 real: OpenMM
          # needs a FULL-ATOM structure and mock s04 only yields a CA trace
          # (-> "wrong set of atoms" / skipped_no_full_atom_structure). So
          # this rung includes a real Boltz-2 run (~20 min, GPU) feeding a
          # real full-atom PDB into MD. protocol level via smoke.yaml.
  pin_gpu
  export BOLTZ_CACHE="${BOLTZ_CACHE:-$EVOLIEZ_ROOT/evoliez_assets/boltz_cache}"
  mkdir -p "$BOLTZ_CACHE"
  evoliez run -c "$SMOKE_CFG" --backend mock --output-dir "$RUN/md" \
    --stage-backend s04_complex=real --stage-backend s10_md=real
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
