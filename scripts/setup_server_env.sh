#!/usr/bin/env bash
# Create the EvoLiEZ conda env on the GPU server.
#
# CRITICAL: the server's root "/" is ~99% full and ~/anaconda3 lives there.
# The env, weights and DBs MUST go on /mnt/data2 (NVMe, ~4.8 TB free).
#
# Usage:  bash scripts/setup_server_env.sh
set -euo pipefail

ENV_PREFIX="${EVOLIEZ_ENV_PREFIX:-/mnt/data2/$USER/envs/evoliez}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$ENV_PREFIX" in
  /mnt/data2/*|/mnt/data/*) : ;;
  *) echo "REFUSING: env prefix '$ENV_PREFIX' is not under /mnt/data2 or /mnt/data."
     echo "Root '/' is full - set EVOLIEZ_ENV_PREFIX to a /mnt/data2 path."; exit 1 ;;
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

Next:
  bash scripts/fetch_weights.sh        # Boltz-2 / LigandMPNN weights -> /mnt/data2
  bash scripts/run_pipeline.sh configs/example_fdh_nadp.yaml
EOF
