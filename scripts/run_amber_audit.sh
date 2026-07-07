#!/usr/bin/env bash
# V6-0 Amber Capability Audit — one-shot server runner.
#
# Self-contained + idempotent: sets up the canonical Amber runtime (AmberTools
# from the evoliez prefix env + pmemd.cuda_SPFP from the pmemd24 build with its
# runtime libs), then runs scripts/check_amber_gpu.py which does the real
# tleap -> pmemd.cuda -> cpptraj GPU round-trip and writes the runtime profile.
#
# Override any path via env, e.g.:  AUDIT_GPU=2 bash scripts/run_amber_audit.sh
set -euo pipefail

# ---- canonical server locations (override via env) ------------------------- #
EVOLIEZ_ENV="${EVOLIEZ_ENV:-/mnt/data/jglee/envs/evoliez}"          # AmberTools
PMEMD_ROOT="${PMEMD_ROOT:-/mnt/data/jglee/pmemd24}"                 # pmemd build
PMEMD_BIN="${PMEMD_BIN:-$PMEMD_ROOT/bin/pmemd.cuda_SPFP}"
CONDA_SH="${CONDA_SH:-/mnt/data/jglee/anaconda3/etc/profile.d/conda.sh}"
AUDIT_GPU="${AUDIT_GPU:-0}"                                         # single idle GPU
AUDIT_WORK="${AUDIT_WORK:-/mnt/data/jglee/v6_audit_work}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
AUDIT_OUT="${AUDIT_OUT:-$REPO_DIR/reports/provenance/amber_runtime_profile.json}"

echo "== V6-0 Amber audit =="
echo "repo       : $REPO_DIR"
echo "ambertools : $EVOLIEZ_ENV"
echo "pmemd.cuda : $PMEMD_BIN"
echo "gpu        : $AUDIT_GPU"
echo "work       : $AUDIT_WORK"
echo "out        : $AUDIT_OUT"

# ---- env setup ------------------------------------------------------------- #
# 1) AmberTools from the evoliez prefix (RPATH-linked binaries; sets leaprc path)
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$EVOLIEZ_ENV"
else
  export PATH="$EVOLIEZ_ENV/bin:$PATH"
fi
# 2) pmemd.cuda runtime libs (prepends pmemd24 libs to LD_LIBRARY_PATH)
if [ -f "$PMEMD_ROOT/amber.sh" ]; then
  # shellcheck disable=SC1090
  source "$PMEMD_ROOT/amber.sh"
fi
# 3) restore AMBERHOME to the AmberTools install (amber.sh above repointed it)
export AMBERHOME="$EVOLIEZ_ENV"
export PATH="$EVOLIEZ_ENV/bin:$PATH"
export EVOLIEZ_AMBERTOOLS_BIN="$EVOLIEZ_ENV/bin"
export EVOLIEZ_PMEMD_CUDA="$PMEMD_BIN"
export CUDA_VISIBLE_DEVICES="$AUDIT_GPU"

echo "-- resolved tools --"
echo "tleap    : $(command -v tleap || echo MISSING)"
echo "cpptraj  : $(command -v cpptraj || echo MISSING)"
echo "pmemd    : $EVOLIEZ_PMEMD_CUDA ($([ -x "$EVOLIEZ_PMEMD_CUDA" ] && echo exists || echo MISSING))"
echo "python   : $(command -v python)"

mkdir -p "$AUDIT_WORK" "$(dirname "$AUDIT_OUT")"

python "$REPO_DIR/scripts/check_amber_gpu.py" \
  --out "$AUDIT_OUT" \
  --work "$AUDIT_WORK"
