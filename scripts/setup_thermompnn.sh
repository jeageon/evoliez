#!/usr/bin/env bash
# One-shot ThermoMPNN setup on the server: open-source FoldX/Rosetta replacement for
# the s09 stability ΔΔG (Kuhlman-Lab/ThermoMPNN, MIT, PNAS 2024). Weights (~66 MB) are
# bundled in the repo (plain git clone, no LFS). CPU inference (model is tiny; the SSM
# is one embedding). Idempotent. Pins are the ones a clean install needs:
#   setuptools<80  -> keeps pkg_resources (the PL-2.0.2 checkpoint's loader needs it)
#   numpy<2        -> torch-2.0.1's from_numpy ABI ("Numpy is not available" otherwise)
#   wandb, joblib  -> import-time deps of train_thermompnn (pulled in even for inference)
# custom_inference.py is CPU-broken (no map_location); we verify via OUR runner instead.
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
ROOT=/mnt/data/jglee
REPO="$ROOT/ThermoMPNN"
ENV=thermompnn

echo "=== 1. clone (weights bundled) ==="
[ -d "$REPO/.git" ] || git clone --depth 1 https://github.com/Kuhlman-Lab/ThermoMPNN.git "$REPO"
ls -la "$REPO/models/thermoMPNN_default.pt" 2>/dev/null

echo "=== 2. CPU env + pinned deps ==="
conda env list | grep -qE "^thermompnn[[:space:]]" || conda create -y -n "$ENV" python=3.10
conda activate "$ENV"
pip install -q "torch==2.0.1" --index-url https://download.pytorch.org/whl/cpu
pip install -q "pytorch-lightning==2.0.2" "setuptools<80" "numpy<2" \
  omegaconf biopython pandas tqdm wandb joblib

echo "=== 3. verify via OUR runner (forces map_location=cpu; overrides authors' paths) ==="
export WANDB_MODE=disabled
python /mnt/data/jglee/EvoLiEZ/scripts/thermompnn_ssm.py \
  "$REPO/examples/2OCJ.pdb" A /tmp/tmpnn_verify.json "$REPO"
python - <<'PY'
import json, statistics
t = json.load(open("/tmp/tmpnn_verify.json"))["by_resnum"]
v = [x for p in t.values() for x in p["ddg"].values()]
print(f"VERIFY OK: {len(t)} positions, ddG n={len(v)} "
      f"min={min(v):.2f} max={max(v):.2f} mean={statistics.mean(v):.2f} std={statistics.pstdev(v):.2f}")
PY
echo ">> EVOLIEZ_THERMOMPNN=$REPO"
echo ">> EVOLIEZ_THERMOMPNN_PYTHON=$CONDA_PREFIX/bin/python"
echo "DONE"