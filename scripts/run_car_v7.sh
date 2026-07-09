#!/usr/bin/env bash
# =====================================================================
# CAR V7 launcher — ONE self-contained, idempotent script (ROADMAP_V7).
#   bash scripts/run_car_v7.sh preflight   # V7 portfolio unit tests + config load (no GPU)
#   bash scripts/run_car_v7.sh prep        # seed runs/srcar_3hp_v7 from srcar_3hp_v5 cached s01-s06b
#   bash scripts/run_car_v7.sh gen         # s07 ONLY (full universe) + inspect — the halt-on-bug gate
#   bash scripts/run_car_v7.sh run         # full s07->s11 (GPU; subset Boltz/MD) + portfolio
#   bash scripts/run_car_v7.sh portfolio   # build the V7 mechanism-ranked portfolio from the run
#
# Mirrors scripts/run_car_v5.sh env. The V7 run REUSES the candidate-independent s01-s06b
# artifacts (Boltz WT structure, MSA, family interaction model) from the finished
# srcar_3hp_v5 run and re-runs only s07-s11 over the FULL candidate universe (no seed CSV),
# so bands are computed over hundreds of variants while expensive Boltz/MD stay subset-only.
# Shared-box safe (EVOLIEZ_NUM_THREADS<=8; GPU pool from CUDA_VISIBLE_DEVICES).
# =====================================================================
set -euo pipefail
ulimit -n 65536 2>/dev/null || ulimit -n 8192 2>/dev/null || true

MODE="${1:?usage: run_car_v7.sh <preflight|prep|gen|run|portfolio> [extra cli args...]}"; shift || true

CAR_ROOT="${CAR_ROOT:-/mnt/data/jglee/EvoLiEZ_car}"
EVO_PY="${EVO_PY:-/mnt/data/jglee/envs/evoliez/bin/python}"
export PYTHONPATH="$CAR_ROOT/src"
CONFIG="${CAR_CONFIG:-configs/car_srcar_3hp_v7.yaml}"
SRC_RUN="/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_v5"
OUT="/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_v7"

# --- storage / weights / external tool envs (adapters resolve these) ---
export EVOLIEZ_ROOT=/mnt/data/jglee
export EVOLIEZ_DATA_DIR=/mnt/data/jglee/evoliez_assets
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export EVOLIEZ_LIGANDMPNN=/mnt/data/jglee/LigandMPNN
export EVOLIEZ_LIGANDMPNN_PYTHON=/mnt/data/jglee/envs/ligandmpnn/bin/python
export EVOLIEZ_DIFFDOCK=/mnt/data/jglee/DiffDock
export EVOLIEZ_DIFFDOCK_PYTHON=/mnt/data/jglee/envs/diffdock/bin/python
export EVOLIEZ_THERMOMPNN=/mnt/data/jglee/ThermoMPNN
export EVOLIEZ_THERMOMPNN_PYTHON=/mnt/data/jglee/miniforge3/envs/thermompnn/bin/python

# --- AmberTools (AM1-BCC) — MG ion parameters live in the Amber ion FF ---
AMBERTOOLS_ENV=/mnt/data/jglee/anaconda3/envs/AmberTools25
if [ -x "$AMBERTOOLS_ENV/bin/antechamber" ]; then
  export AMBERHOME="$AMBERTOOLS_ENV"
  export EVOLIEZ_AMBERTOOLS_BIN="$AMBERTOOLS_ENV/bin"
  export LD_LIBRARY_PATH="$AMBERTOOLS_ENV/lib:${LD_LIBRARY_PATH:-}"
  AMBER_PATH="$AMBERTOOLS_ENV/bin:"
else
  AMBER_PATH=""
fi
export PATH="/mnt/data/jglee/envs/evoliez/bin:/mnt/data/jglee/bin:/mnt/data/jglee/envs/boltz/bin:${AMBER_PATH}$PATH"

# --- shared-box safety + UTF-8 report I/O ---
export PYTHONUTF8=1; export LC_ALL="${LC_ALL:-C.UTF-8}"; export LANG="${LANG:-C.UTF-8}"
export EVOLIEZ_NUM_THREADS="${EVOLIEZ_NUM_THREADS:-8}"
export OMP_NUM_THREADS="$EVOLIEZ_NUM_THREADS"; export MKL_NUM_THREADS="$EVOLIEZ_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$EVOLIEZ_NUM_THREADS"; export NUMEXPR_NUM_THREADS="$EVOLIEZ_NUM_THREADS"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

cd "$CAR_ROOT"
echo ">> mode=$MODE  config=$CONFIG  out=$OUT"
echo ">> code=$($EVO_PY -c 'import evoliez,os;print(os.path.dirname(evoliez.__file__))')"

case "$MODE" in
  preflight)
    echo ">> V7 preflight: portfolio unit tests + config load (no GPU)"
    "$EVO_PY" -m pytest tests/portfolio -q
    "$EVO_PY" - <<'PY'
from evoliez.config import load_config
c = load_config("configs/car_srcar_3hp_v7.yaml")
print("  mechanism:", c.mechanism.reaction.cls, "| portfolio:", c.portfolio.enabled,
      "panel", c.portfolio.panel_size, "| max_candidates:", c.mutation_generation.max_candidates)
PY
    echo ">> preflight OK";;

  prep)
    # Seed srcar_3hp_v7 from srcar_3hp_v5's candidate-INDEPENDENT cached stages (s01-s06b:
    # Boltz WT structure, MSA, docking, family interaction model). Re-run only s07-s11.
    [ -d "$SRC_RUN" ] || { echo "!! source run $SRC_RUN not found"; exit 1; }
    if [ ! -d "$OUT" ]; then
      echo ">> copying cached s01-s06b artifacts $SRC_RUN -> $OUT (excluding md/validation/mutations/reports)"
      rsync -a --exclude 'md' --exclude 'validation' --exclude 'mutations' \
            --exclude 'reports' --exclude 'logs' "$SRC_RUN/" "$OUT/"
    else
      echo ">> $OUT already exists — reusing"
    fi
    # reset resume state: keep s01..s06b done, drop s07-s11 so --resume re-runs them.
    "$EVO_PY" - "$OUT" <<'PY'
import json, sys
p = f"{sys.argv[1]}/_state.json"
s = json.load(open(p))
keep = ("s01_input","s02_homolog","s03_msa","s04_complex","s05_docking","s06_graph","s06b_interaction")
comp = s.get("completed_stages", {})
if isinstance(comp, dict):
    s["completed_stages"] = {k:v for k,v in comp.items() if k in keep}
else:
    s["completed_stages"] = [k for k in comp if k in keep]
json.dump(s, open(p,"w"), indent=2)
print("  kept completed stages:", sorted(s["completed_stages"]) if isinstance(s["completed_stages"],dict) else s["completed_stages"])
PY
    # align the stored fingerprint to the V7 config (only config_sha1 changed; input identical).
    echo ">> patching run fingerprint to the V7 config (config_sha1-only change)"
    "$EVO_PY" "$CAR_ROOT/scripts/check_run_fingerprint.py" "$CONFIG" "$OUT" --patch || {
      echo "!! fingerprint patch refused (a non-config field changed) — inspect before proceeding"; exit 1; }
    echo ">> prep OK — ready for: bash scripts/run_car_v7.sh gen";;

  gen)
    # HALT-ON-BUG GATE: run s07 ONLY, reusing s01-s06b, and inspect the generated universe
    # before committing GPU time to s08b/s10.
    echo ">> s07 full-universe generation (reuse s01-s06b)"
    # NOTE: no --from. --resume walks s01->s07: the DONE s01-s06b are reloaded via load()
    # (repopulating ctx: wt_complex, position_features, ...), then the not-done s07 runs.
    # (--from s07 would slice s01-s06b out of the walk, so their load() never runs -> ctx empty.)
    "$EVO_PY" -m evoliez.cli run -c "$CONFIG" --to s07_mutation_gen --resume "$@"
    "$EVO_PY" - "$OUT" <<'PY'
import json, sys
d = json.load(open(f"{sys.argv[1]}/reports/provenance/generated_candidates.json"))
gens = {}
for r in d: gens[r.get("generator","?")] = gens.get(r.get("generator","?"),0)+1
print(f"  generated {len(d)} candidates | by generator: {gens}")
PY
    echo ">> gen OK — inspect the counts above, then: bash scripts/run_car_v7.sh run";;

  run)
    # Full s07->s11 over the full universe. s08b (Boltz) + s10 (MD) stay subset-only via the
    # top_n caps + selection_lanes. NO seed CSV (full generation). GPU-first.
    echo ">> full s07->s11 run (subset Boltz/MD)  GPUs=$CUDA_VISIBLE_DEVICES  threads=$EVOLIEZ_NUM_THREADS"
    # no --from: --resume reloads the DONE s01-s06b via load() then runs the not-done s07-s11.
    "$EVO_PY" -m evoliez.cli run -c "$CONFIG" --resume "$@"
    echo ">> pipeline done — building V7 portfolio"
    "$EVO_PY" -m evoliez.cli portfolio --run "$OUT" -c "$CONFIG" --subset-level
    echo ">> run OK — portfolio in $OUT/reports/v7_portfolio.{html,json,csv}";;

  portfolio)
    echo ">> building V7 mechanism-ranked portfolio from $OUT"
    "$EVO_PY" -m evoliez.cli portfolio --run "$OUT" -c "$CONFIG" --subset-level "$@"
    echo ">> portfolio in $OUT/reports/v7_portfolio.{html,json,csv}";;

  *) echo "unknown mode: $MODE (use preflight|prep|gen|run|portfolio)"; exit 2;;
esac
