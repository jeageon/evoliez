#!/usr/bin/env bash
# Preview what `interaction_model.representative_homologs: auto` resolves to on a
# run's REAL homolog set — read-only, no Boltz, no GPU. Safe to run any time.
#
#   bash scripts/preview_representatives.sh configs/server_fdh_nadp.yaml
#
# Needs s02_homolog already done for that config (reads its evoliez.sqlite). Prints
# the subfamily-cluster coverage curve and the auto-resolved representative count
# under the config's coverage/min/max, so you can tune the bounds before launching
# the (GPU-heavy) s06b family ensemble.
set -euo pipefail

CONFIG="${1:?usage: preview_representatives.sh CONFIG.yaml}"
ENV_NAME="${EVOLIEZ_ENV:-evoliez-cu124}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME" 2>/dev/null || echo "WARN: conda env '$ENV_NAME' not found; using current python" >&2
fi
cd "$HERE"
exec python scripts/preview_representatives.py "$CONFIG"
