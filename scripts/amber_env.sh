#!/usr/bin/env bash
# Canonical Amber runtime env for EvoLiEZ V6 on the evo server. SOURCE this
# (do not execute) before any tleap/antechamber/pmemd.cuda/cpptraj/parmed work:
#
#   source scripts/amber_env.sh
#
# Sets: AmberTools from the evoliez prefix env (RPATH-linked; provides leaprc),
# pmemd.cuda_SPFP from the pmemd24 build (runtime libs sourced), and the
# EVOLIEZ_* pointers the engine reads. CUDA_VISIBLE_DEVICES is left untouched
# unless AMBER_GPU is set. Idempotent.
EVOLIEZ_ENV="${EVOLIEZ_ENV:-/mnt/data/jglee/envs/evoliez}"
PMEMD_ROOT="${PMEMD_ROOT:-/mnt/data/jglee/pmemd24}"
PMEMD_BIN="${PMEMD_BIN:-$PMEMD_ROOT/bin/pmemd.cuda_SPFP}"
CONDA_SH="${CONDA_SH:-/mnt/data/jglee/anaconda3/etc/profile.d/conda.sh}"

if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$EVOLIEZ_ENV"
else
  export PATH="$EVOLIEZ_ENV/bin:$PATH"
fi
if [ -f "$PMEMD_ROOT/amber.sh" ]; then
  # shellcheck disable=SC1090
  source "$PMEMD_ROOT/amber.sh"
fi
export AMBERHOME="$EVOLIEZ_ENV"
export PATH="$EVOLIEZ_ENV/bin:$PATH"
export EVOLIEZ_AMBERTOOLS_BIN="$EVOLIEZ_ENV/bin"
export EVOLIEZ_PMEMD_CUDA="$PMEMD_BIN"
if [ -n "${AMBER_GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$AMBER_GPU"
fi
