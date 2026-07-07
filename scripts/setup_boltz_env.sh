#!/usr/bin/env bash
# Boltz-2 in its OWN isolated conda env. Boltz pins numpy<2 / old click /
# etc., so `pip install boltz` into the evoliez env BREAKS it. Our adapter
# only ever calls `boltz` as a subprocess, so an isolated env on PATH is all
# that's needed. Same idea applies to DiffDock.
#
#   export EVOLIEZ_ROOT=/mnt/data/jglee
#   bash scripts/setup_boltz_env.sh
set -euo pipefail

EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/$USER}"
BOLTZ_ENV="${BOLTZ_ENV:-$EVOLIEZ_ROOT/envs/boltz}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$EVOLIEZ_ROOT/evoliez_assets/boltz_cache}"

case "$BOLTZ_ENV" in
  /mnt/data2/*|/mnt/data/*) : ;;
  *) echo "REFUSING: BOLTZ_ENV '$BOLTZ_ENV' must be under /mnt/data2|/mnt/data"
     exit 1 ;;
esac

CREATE=(conda create); command -v mamba >/dev/null 2>&1 && CREATE=(mamba create)
# Idempotent: `conda/mamba create -p` ERRORS on an existing prefix (and -y does
# not clobber), so under `set -e` a re-run would abort here before refreshing
# boltz/torch. Only create when the env isn't there; always run the refresh
# steps below so re-running picks up a newer boltz / re-pins torch.
if [ ! -x "$BOLTZ_ENV/bin/python" ]; then
  echo ">> creating isolated Boltz env at $BOLTZ_ENV"
  "${CREATE[@]}" -p "$BOLTZ_ENV" -y -c conda-forge python=3.10 pip
else
  echo ">> reusing existing Boltz env at $BOLTZ_ENV"
fi
conda run -p "$BOLTZ_ENV" pip install -U boltz

# PyPI's default `torch` now ships CUDA-13 wheels. The lab server driver is
# CUDA 12.4, which CANNOT run cu13 binaries -> torch.cuda.is_available()==False
# -> Boltz-2 silently falls back to CPU (unusably slow). Pin torch to a
# CUDA-12.4 build. Override with BOLTZ_TORCH_CUDA=cu126 / =skip if needed.
BOLTZ_TORCH_CUDA="${BOLTZ_TORCH_CUDA:-cu124}"
if [ "$BOLTZ_TORCH_CUDA" != "skip" ]; then
  echo ">> pinning torch to a $BOLTZ_TORCH_CUDA build (server driver = CUDA 12.4)"
  conda run -p "$BOLTZ_ENV" pip install --force-reinstall --no-cache-dir \
    torch --index-url "https://download.pytorch.org/whl/$BOLTZ_TORCH_CUDA"
fi
echo ">> boltz-env CUDA self-check:"
conda run -p "$BOLTZ_ENV" python -c \
  'import torch;print("  torch",torch.__version__,"cuda_build",torch.version.cuda,"avail",torch.cuda.is_available())' \
  || echo "  (torch import failed - inspect above)"

mkdir -p "$BOLTZ_CACHE"
ver="$(conda run -p "$BOLTZ_ENV" python -c \
  'import importlib.metadata as m; print(m.version("boltz"))')"
echo ">> boltz $ver  (binary: $BOLTZ_ENV/bin/boltz)"
echo ">> BOLTZ_CACHE=$BOLTZ_CACHE"
cat <<EOF

Done. The smoke script auto-prepends $BOLTZ_ENV/bin to PATH, so:

  bash scripts/server_smoke.sh boltz

runs evoliez (evoliez env) while 'boltz' resolves to the isolated env.
Do NOT 'pip install boltz' into the evoliez env.
EOF
