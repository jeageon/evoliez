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

STEP="${1:?usage: server_smoke.sh <doctor|dryrun|boltz|dock|md|gnn|all> [config]}"
CFG="${2:-configs/server_fdh_nadp.yaml}"
EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/${USER}}"
RUN="$EVOLIEZ_ROOT/runs/smoke"

pin_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | sort -t, -k3 -n -k2 -nr \
      | head -1 | cut -d, -f1 | tr -d ' ')"
    echo ">> pinned GPU $CUDA_VISIBLE_DEVICES"
  fi
}

case "$STEP" in
doctor)
  evoliez doctor -c "$CFG"
  ;;
dryrun)   # step 2: command preview, no execution, no GPU
  evoliez run -c "$CFG" --backend real --dry-run --output-dir "$RUN/dry"
  ;;
boltz)    # step 3: homolog+MSA+Boltz real, stop at s04
  pin_gpu
  evoliez run -c "$CFG" --backend mock --output-dir "$RUN/boltz" \
    --stage-backend s02_homolog=real --stage-backend s03_msa=real \
    --stage-backend s04_complex=real --to s04_complex
  bash scripts/capture_fixtures.sh "$RUN/boltz"
  ;;
dock)     # step 4: docking + stability real on a few candidates
  pin_gpu
  evoliez run -c "$CFG" --backend mock --output-dir "$RUN/dock" \
    --stage-backend s05_docking=real --stage-backend s09_nonmd=real \
    --to s09_nonmd
  bash scripts/capture_fixtures.sh "$RUN/dock"
  ;;
md)       # step 5: OpenMM minimise/MD-lite real (protocol level via config)
  pin_gpu
  evoliez run -c "$CFG" --backend mock --output-dir "$RUN/md" \
    --stage-backend s10_md=real
  bash scripts/capture_fixtures.sh "$RUN/md"
  ;;
gnn)      # step 6: 1-epoch single-GPU GNN train (DDP comes later)
  pin_gpu
  evoliez run -c "$CFG" --backend mock --output-dir "$RUN/gnn" >/dev/null
  evoliez train-gnn -c "$CFG" --epochs 1 \
    --dataset "$RUN/gnn/datasets/graph_pt"
  ;;
all)
  for s in doctor dryrun boltz dock md gnn; do
    echo "==== smoke step: $s ===="
    bash "$0" "$s" "$CFG"
  done
  ;;
*) echo "unknown step '$STEP'"; exit 1 ;;
esac
echo ">> smoke step '$STEP' OK"
