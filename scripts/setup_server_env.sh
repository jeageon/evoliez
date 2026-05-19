#!/usr/bin/env bash
# Create the EvoLiEZ conda env on the GPU server.
#
# CRITICAL: the server's root "/" is ~99% full and ~/anaconda3 lives there.
# Env, weights and DBs MUST live under EVOLIEZ_ROOT on /mnt/data2 or /mnt/data.
#
#   export EVOLIEZ_ROOT=/mnt/data/jglee     # one var drives everything
#   bash scripts/setup_server_env.sh
#
# (/mnt/data2 is NVMe + more free space; /mnt/data also works.)
set -euo pipefail

EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/$USER}"
ENV_PREFIX="${EVOLIEZ_ENV_PREFIX:-$EVOLIEZ_ROOT/envs/evoliez}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$ENV_PREFIX" in
  /mnt/data2/*|/mnt/data/*) : ;;
  *) echo "REFUSING: env prefix '$ENV_PREFIX' is not under /mnt/data2 or /mnt/data."
     echo "Root '/' is full - set EVOLIEZ_ROOT=/mnt/data/<you> (or /mnt/data2/<you>)."; exit 1 ;;
esac

avail_gb=$(df -BG --output=avail "$(dirname "$ENV_PREFIX" 2>/dev/null || echo /mnt/data2)" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [ "${avail_gb:-0}" -lt 30 ]; then
  echo "WARNING: only ${avail_gb}G free where the env will be created."
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH (expected the server 'base' env)."; exit 1
fi

echo ">> creating env at $ENV_PREFIX"
conda env create -f "$REPO_DIR/environment-gpu.yml" -p "$ENV_PREFIX"

echo ">> verifying CUDA / OpenMM"
conda run -p "$ENV_PREFIX" python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "n_gpu", torch.cuda.device_count())
try:
    import openmm
    from openmm import Platform
    print("openmm", openmm.__version__,
          "platforms", [Platform.getPlatform(i).getName()
                        for i in range(Platform.getNumPlatforms())])
except Exception as e:
    print("openmm check failed:", e)
PY

cat <<EOF

Done. Activate with:
  conda activate $ENV_PREFIX

Next (same EVOLIEZ_ROOT=$EVOLIEZ_ROOT):
  bash scripts/fetch_weights.sh        # Boltz-2 / LigandMPNN weights
  bash scripts/server_smoke.sh doctor  # preflight
EOF
