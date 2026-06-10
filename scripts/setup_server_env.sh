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

# df the NEAREST EXISTING parent (env dir doesn't exist yet -> avoids the
# bogus "0G free" warning).
probe="$ENV_PREFIX"
while [ ! -d "$probe" ] && [ "$probe" != "/" ]; do probe="$(dirname "$probe")"; done
avail_gb=$(df -BG --output=avail "$probe" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [ "${avail_gb:-0}" -lt 30 ]; then
  echo "WARNING: only ${avail_gb}G free at $probe."
else
  echo ">> ${avail_gb}G free at $probe"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH (expected the server 'base' env)."; exit 1
fi

# The GPU env (pytorch+cuda+ambertools+openmm+rdkit+blast+mmseqs2+foldseek)
# is heavy; the classic conda solver can take 30+ min. Use mamba or the
# libmamba solver (seconds-to-minutes) automatically.
CREATE=(conda env create)
if command -v mamba >/dev/null 2>&1; then
  CREATE=(mamba env create)
  echo ">> using mamba (fast solver)"
elif conda list -n base 2>/dev/null | grep -q conda-libmamba-solver; then
  # Gate ONLY on the plugin actually being installed. `conda config --show
  # solver` exits 0 regardless (it just prints the configured setting), so it
  # would select --solver=libmamba even when the plugin is absent -> env
  # creation then fails under `set -e`.
  CREATE=(conda env create --solver=libmamba)
  echo ">> using conda libmamba solver"
else
  echo ">> NOTE: no mamba/libmamba; the classic solver is slow. To speed up:"
  echo "   conda install -n base -y conda-libmamba-solver  (then re-run)"
fi

echo ">> creating env at $ENV_PREFIX"
"${CREATE[@]}" -f "$REPO_DIR/environment-gpu.yml" -p "$ENV_PREFIX"

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
