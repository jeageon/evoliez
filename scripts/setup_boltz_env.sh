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
echo ">> creating isolated Boltz env at $BOLTZ_ENV"
"${CREATE[@]}" -p "$BOLTZ_ENV" -y -c conda-forge python=3.10 pip
conda run -p "$BOLTZ_ENV" pip install -U boltz

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
