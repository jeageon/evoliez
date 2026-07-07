#!/usr/bin/env bash
# LigandMPNN env for s07 (ligand-aware sequence design). The repo + model weights
# are already on the server (LigandMPNN/model_params/*.pt); this just creates the
# isolated conda env with its deps. torch 2.2.1 ships cu121 wheels — fine on the
# lab CUDA-12.4 driver (minor backward-compat, same idea as the cu117 DiffDock
# env). Idempotent: re-running reuses the env and refreshes deps.
#
#   bash scripts/setup_ligandmpnn_env.sh
set -euo pipefail

EVOLIEZ_ROOT="${EVOLIEZ_ROOT:-/mnt/data/jglee}"
LM_SRC="${EVOLIEZ_LIGANDMPNN:-$EVOLIEZ_ROOT/LigandMPNN}"
LM_ENV="${LM_ENV:-$EVOLIEZ_ROOT/envs/ligandmpnn}"
case "$LM_ENV" in /mnt/data2/*|/mnt/data/*) : ;; *)
  echo "REFUSING: LM_ENV '$LM_ENV' must live under /mnt/data|/mnt/data2"; exit 1 ;; esac
[ -f "$LM_SRC/run.py" ] || { echo "LigandMPNN repo has no run.py at $LM_SRC"; exit 1; }

CREATE=(conda create); command -v mamba >/dev/null 2>&1 && CREATE=(mamba create)
if [ ! -x "$LM_ENV/bin/python" ]; then
  echo ">> creating LigandMPNN env at $LM_ENV (python 3.10)"
  "${CREATE[@]}" -p "$LM_ENV" -y -c conda-forge python=3.10 pip
else
  echo ">> reusing existing env at $LM_ENV"
fi
RUN=(conda run --no-capture-output -p "$LM_ENV")

echo ">> installing requirements (torch 2.2.1 cu121 runs on the CUDA-12.4 driver)"
if ! "${RUN[@]}" pip install -q -r "$LM_SRC/requirements.txt"; then
  echo ">> requirements.txt failed (usually the pinned ProDy build) — install the"
  echo "   core deps via pip and ProDy via conda (a wheel that builds cleanly)"
  "${RUN[@]}" pip install -q "torch==2.2.1" "numpy==1.23.5" "scipy==1.12.0" \
    ml-collections dm-tree biopython filelock networkx
  conda install -p "$LM_ENV" -y -c conda-forge "prody=2.4" >/dev/null
fi

# ProDy 2.4 still imports pkg_resources, which setuptools>=81 dropped -> pin it
# back so `import prody` (and thus run.py) works.
"${RUN[@]}" pip install -q "setuptools<81"

echo ">> weights: $(ls "$LM_SRC"/model_params/*.pt 2>/dev/null | wc -l) checkpoints in model_params/"
echo ">> self-check:"
"${RUN[@]}" python -c \
  'import torch,prody,ml_collections;print("  torch",torch.__version__,"cuda",torch.cuda.is_available(),"| prody",prody.__version__)' \
  || echo "  (import failed — inspect above)"

cat <<EOF

Done. For s07 (the ligandmpnn method), launch evoliez with:
  export PATH=$LM_ENV/bin:\$PATH        # the ligandmpnn python ('run.py' runs here)
  export EVOLIEZ_LIGANDMPNN=$LM_SRC     # the adapter runs 'python run.py' from this dir (cwd)
Weights already in $LM_SRC/model_params (run.py's default checkpoint resolves relative to the repo).
EOF
