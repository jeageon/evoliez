#!/usr/bin/env bash
# One-shot server validation for the feat/server-hardening changes.
#
# Self-contained: finds + ACTIVATES the conda env on /mnt/data2 (the other
# server scripts assume it's already active), makes the editable install current
# with the freshly-pulled source, runs the unit suite, then the STAGED real-
# backend smoke (doctor -> dryrun -> boltz -> dock -> md -> gnn). Idempotent;
# stops on the first failure so debugging stays localised. No sudo, nothing on
# root "/".
#
#   # on the server, in the cloned repo (e.g. /mnt/data2/$USER/EvoLiEZ):
#   git fetch origin && git checkout feat/server-hardening && git pull --ff-only
#   export EVOLIEZ_ROOT=/mnt/data2/$USER          # where env/weights/runs live
#   bash scripts/server_test.sh                   # = quick + staged real smoke
#
#   bash scripts/server_test.sh quick             # just the 120 unit tests (no GPU)
#   bash scripts/server_test.sh doctor|dryrun|boltz|dock|md|gnn   # one stage
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/$USER}"
export EVOLIEZ_ROOT
ENV_PREFIX="${EVOLIEZ_ENV_PREFIX:-$EVOLIEZ_ROOT/envs/evoliez}"
STEP="${1:-all}"

# --- locate + activate the conda env (conda may live on root; the ENV is on
#     /mnt/data2). Source conda.sh so `conda activate` works in this shell. ----
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" /opt/conda; do
    if [ -f "$c/etc/profile.d/conda.sh" ]; then source "$c/etc/profile.d/conda.sh"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda not found. Run scripts/setup_server_env.sh first."; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
[ -d "$ENV_PREFIX" ] || {
  echo "ERROR: env $ENV_PREFIX missing."
  echo "  EVOLIEZ_ROOT=$EVOLIEZ_ROOT bash scripts/setup_server_env.sh"; exit 1; }
conda activate "$ENV_PREFIX"
echo ">> python: $(command -v python)  ($(python -V 2>&1))"
echo ">> EVOLIEZ_ROOT=$EVOLIEZ_ROOT  ENV=$ENV_PREFIX"

# --- make the pulled source live (idempotent; picks up the new adapters/MD) ---
pip install -e . -q
echo ">> evoliez: $(python -c 'import evoliez,inspect,os;print(os.path.dirname(inspect.getfile(evoliez)))')"

# --- step 0: fast unit sanity in the SERVER env (the 120 dev-box tests; no GPU)
if [ "$STEP" = "quick" ] || [ "$STEP" = "all" ]; then
  echo "==== unit suite (server env) ===="
  if python -c "import pytest" 2>/dev/null; then
    python -m pytest -q
  else
    echo ">> pytest not installed in env; skipping (pip install -e '.[dev]' to enable)"
  fi
  [ "$STEP" = "quick" ] && { echo ">> server_test 'quick' OK"; exit 0; }
fi

# --- staged real-backend smoke (reuses the existing per-stage script) ---------
echo "==== real-backend smoke: $STEP ===="
bash scripts/server_smoke.sh "$STEP"
echo ">> server_test '$STEP' OK"
