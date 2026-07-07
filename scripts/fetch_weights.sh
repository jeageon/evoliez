#!/usr/bin/env bash
# Fetch model weights / install GPU tools onto /mnt/data2 (never onto root).
#
# Usage:  bash scripts/fetch_weights.sh
set -euo pipefail

EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/$USER}"
DEST="${EVOLIEZ_DATA_DIR:-$EVOLIEZ_ROOT/evoliez_assets}"
case "$DEST" in
  /mnt/data2/*|/mnt/data/*) : ;;
  *) echo "REFUSING: '$DEST' must be under /mnt/data2 or /mnt/data "
     echo "(set EVOLIEZ_ROOT=/mnt/data/<you>)."; exit 1 ;;
esac
mkdir -p "$DEST"
cd "$DEST"

echo ">> assets dir: $DEST"

# --- Boltz / Boltz-2 -------------------------------------------------------
if [ ! -d boltz ]; then
  echo ">> installing Boltz"
  pip install boltz                       # weights auto-download to ~/.boltz on first run;
  export BOLTZ_CACHE="$DEST/boltz_cache"  # redirect cache off root
  mkdir -p "$BOLTZ_CACHE"
fi

# --- LigandMPNN ------------------------------------------------------------
if [ ! -d LigandMPNN ]; then
  echo ">> cloning LigandMPNN"
  git clone https://github.com/dauparas/LigandMPNN.git
  ( cd LigandMPNN && bash get_model_params.sh "./model_params" ) || \
    echo "   (run get_model_params.sh manually if it failed)"
fi

# --- DiffDock (optional) ---------------------------------------------------
if [ ! -d DiffDock ]; then
  echo ">> cloning DiffDock (optional docking backend)"
  git clone https://github.com/gcorso/DiffDock.git || true
fi

cat <<EOF

Done. Add to your shell / job script:
  export BOLTZ_CACHE=$DEST/boltz_cache
  export EVOLIEZ_LIGANDMPNN=$DEST/LigandMPNN
  export EVOLIEZ_DIFFDOCK=$DEST/DiffDock

GNINA: install from https://github.com/gnina/gnina (CUDA build) and put on PATH.
FoldX: obtain an academic license (https://foldxsuite.crg.eu/) and put on PATH.
See docs/DATA_AND_WEIGHTS.md.
EOF
