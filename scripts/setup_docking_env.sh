#!/usr/bin/env bash
# Docking tools for s05/s09: gnina (binary) + DiffDock (isolated conda env).
#
#   export EVOLIEZ_ROOT=/mnt/data/jglee
#   bash scripts/setup_docking_env.sh
#
# gnina: the newer prebuilt binaries need glibc >= 2.35; the lab server is
#   Ubuntu 20.04 (glibc 2.31), so use the v1.1 binary (self-contained, runs on
#   2.31) at $EVOLIEZ_ROOT/bin/gnina. Driver CUDA 12.4 is backward-compatible.
# DiffDock: its own env (the repo pins torch 1.13+cu117 — the server's CUDA-12.4
#   driver runs cu117 via backward-compat). openfold/esmfold are NOT installed:
#   DiffDock never imports openfold and we provide the receptor, so the ESMFold
#   structure path is unused. The adapter calls `python -m inference`, so s05
#   must run with PYTHONPATH=$EVOLIEZ_DIFFDOCK and this env's python on PATH.
set -euo pipefail

EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data2/$USER}"
DD_ENV="${DD_ENV:-$EVOLIEZ_ROOT/envs/diffdock}"
DD_SRC="${EVOLIEZ_DIFFDOCK:-$EVOLIEZ_ROOT/DiffDock}"
GNINA="$EVOLIEZ_ROOT/bin/gnina"
case "$DD_ENV" in /mnt/data2/*|/mnt/data/*) : ;; *)
  echo "REFUSING: DD_ENV '$DD_ENV' must be under /mnt/data2|/mnt/data"; exit 1 ;; esac

# --- gnina ---------------------------------------------------------------- #
if [ -x "$GNINA" ]; then
  echo ">> gnina present: $GNINA ($("$GNINA" --version 2>/dev/null | head -1))"
else
  echo ">> downloading gnina v1.1 (glibc-2.31 compatible) -> $GNINA"
  mkdir -p "$(dirname "$GNINA")"
  wget -q "https://github.com/gnina/gnina/releases/download/v1.1/gnina" -O "$GNINA"
  chmod +x "$GNINA"
fi

# --- DiffDock repo + source patch ----------------------------------------- #
[ -d "$DD_SRC" ] || git clone https://github.com/gcorso/DiffDock.git "$DD_SRC"
# torch.from_numpy(RDKit GetConformer().GetPositions()) raises "expected
# np.ndarray (got numpy.ndarray)" when RDKit and torch were built against
# different numpy ABIs. Wrap the RDKit arrays in np.array() (a copy owned by the
# runtime numpy). Idempotent — the already-patched form no longer matches.
PM="$DD_SRC/datasets/process_mols.py"
if [ -f "$PM" ]; then
  sed -i 's/torch.from_numpy(mol.GetConformer().GetPositions())/torch.from_numpy(np.array(mol.GetConformer().GetPositions()))/' "$PM"
  sed -i 's/torch.from_numpy(mol_rdkit.GetConformer().GetPositions())/torch.from_numpy(np.array(mol_rdkit.GetConformer().GetPositions()))/' "$PM"
fi

# --- DiffDock env (idempotent) -------------------------------------------- #
CREATE=(conda create); command -v mamba >/dev/null 2>&1 && CREATE=(mamba create)
if [ ! -x "$DD_ENV/bin/python" ]; then
  echo ">> creating DiffDock env at $DD_ENV (python 3.9)"
  "${CREATE[@]}" -p "$DD_ENV" -y -c conda-forge python=3.9 pip
else
  echo ">> reusing DiffDock env at $DD_ENV"
fi
RUN=(conda run --no-capture-output -p "$DD_ENV")

echo ">> torch 1.13.1+cu117 (driver CUDA 12.4 runs it backward-compat)"
"${RUN[@]}" pip install -q --extra-index-url https://download.pytorch.org/whl/cu117 \
  "torch==1.13.1+cu117"
echo ">> torch-geometric CUDA extensions — EXACT pt113cu117 wheels (bare names let"
echo "   pip grab ABI-mismatched PyPI builds -> 'undefined symbol' at import)"
"${RUN[@]}" pip install -q --force-reinstall --no-deps \
  -f https://data.pyg.org/whl/torch-1.13.1+cu117.html \
  "torch-scatter==2.1.0+pt113cu117" "torch-sparse==0.6.16+pt113cu117" \
  "torch-cluster==1.6.0+pt113cu117" "torch-spline-conv==1.2.1+pt113cu117"
echo ">> prody via conda (its pinned pip wheel fails to compile on modern gcc; the DiffDock environment.yml itself notes ProDy must come from conda)"
conda install -p "$DD_ENV" -y -c conda-forge prody >/dev/null
echo ">> DiffDock python deps (numpy<2 + scipy pinned for torch-1.13; NO openfold)"
"${RUN[@]}" pip install -q \
  "numpy==1.23.5" "scipy==1.12.0" \
  "torch-geometric==2.2.0" "e3nn==0.5.1" "fair-esm==2.0.0" rdkit \
  "networkx==2.8.4" "pandas==1.5.1" "scikit-learn==1.1.0" \
  "pytorch-lightning==1.9.5" "torchmetrics==0.11.0" pyyaml "pybind11==2.11.1"

# Run the env's python with its OWN libstdc++ (LD_LIBRARY_PATH): a bare
# $DD_ENV/bin/python picks the old system libstdc++ (Ubuntu 20.04) and scipy /
# torch C-extensions fail on `GLIBCXX_3.4.29 not found`.
echo ">> self-check (conda libstdc++ via LD_LIBRARY_PATH):"
LD_LIBRARY_PATH="$DD_ENV/lib" "$DD_ENV/bin/python" -c 'import torch,torch_geometric,torch_sparse,scipy,e3nn,esm,rdkit,prody;print("  torch",torch.__version__,"cuda",torch.cuda.is_available(),"| pyg/e3nn/esm/rdkit/prody OK")'
cat <<EOF

Done. For s05/s09 docking, launch evoliez (in the pipeline env) with:
  export PATH=$EVOLIEZ_ROOT/bin:$DD_ENV/bin:\$PATH   # gnina + the diffdock python
  export LD_LIBRARY_PATH=$DD_ENV/lib                 # conda libstdc++ for the diffdock subprocess
  export EVOLIEZ_DIFFDOCK=$DD_SRC                    # the diffdock adapter runs 'python -m inference' from here (cwd)
DiffDock downloads its score/confidence weights to $DD_SRC/workdir on first run.
EOF
