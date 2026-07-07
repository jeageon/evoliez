#!/usr/bin/env bash
# Recover the fdh_5track s06b run after the shared-outdir cross-contamination fix.
#
# The old run pointed every representative at ONE output dir, so predict_complex
# parsed found[0]=hom_000 + mixed all reps' confidence files (corrupt data + an
# O(N^2) re-parse). The Boltz COMPUTE per rep is correct, only evoliez's parsing
# was wrong -- so this:
#   1. kills the corrupt run (its boltz_results stay on disk),
#   2. migrates each boltz_results_hom_NNN_boltz_input into a per-rep subdir
#      representatives/hom_NNN/ (what the fixed code scopes its globbing to),
#   3. restarts s06b: the migrated reps re-parse CORRECTLY via skip-recompute,
#      the remaining reps compute in true 3-GPU parallel.
# Idempotent: re-running re-migrates nothing already moved and relaunches.
set -euo pipefail

source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
export PATH=/mnt/data/jglee/envs/boltz/bin:$PATH      # boltz subprocess env
export BOLTZ_CACHE=/mnt/data/jglee/evoliez_assets/boltz_cache
export CUDA_VISIBLE_DEVICES=0,2,3                     # GPU1 left for the other user
export EVOLIEZ_NUM_THREADS=6
REPO=/mnt/data/jglee/EvoLiEZ
RD=/mnt/data/jglee/runs/fdh_5track/structures/representatives
PAT="evoliez run -c configs/target.local.5track"

echo ">> [1/3] stopping the old (corrupt) s06b run"
pkill -TERM -u jglee -f "$PAT" 2>/dev/null || true
pkill -TERM -u jglee -f "boltz predict" 2>/dev/null || true
for _ in 1 2 3 4 5 6; do pgrep -u jglee -f "$PAT" >/dev/null || break; sleep 2; done
pkill -KILL -u jglee -f "$PAT" 2>/dev/null || true
pkill -KILL -u jglee -f "boltz predict" 2>/dev/null || true
sleep 2
echo "   still alive: $(pgrep -u jglee -f "$PAT" | wc -l) evoliez / $(pgrep -u jglee -f 'boltz predict' | wc -l) boltz"

echo ">> [2/3] migrating boltz_results into per-rep subdirs"
cd "$RD"
moved=0
for d in boltz_results_hom_*_boltz_input; do
  [ -d "$d" ] || continue                            # literal glob (none left) -> skip
  rep=$(printf '%s\n' "$d" | sed -E 's/boltz_results_(hom_[0-9]+)_boltz_input/\1/')
  mkdir -p "$rep"
  mv "$d" "$rep/"
  moved=$((moved + 1))
done
echo "   migrated $moved dir(s); per-rep subdirs now: $(ls -d hom_*/ 2>/dev/null | wc -l)"

echo ">> [3/4] patching _state.json fingerprint to current config (else --resume PURGES)"
# config.py may have gained fields since the run started -> config_sha1 differs
# -> --resume would 'invalidate resume' and rmtree structures/ (deleting the
# migrated reps). Patch the stored fingerprint to the CURRENT config's, built
# WITHOUT .setup() so nothing purges. (Learned the hard way: this step, skipped
# once, deleted 135 salvaged Boltz computes.)
cd "$REPO"
python - <<'PY'
import json
from evoliez.config import load_config
from evoliez.context import RunContext
sf = "/mnt/data/jglee/runs/fdh_5track/_state.json"
fp = RunContext(load_config("configs/target.local.5track.yaml")).run_fingerprint()
d = json.load(open(sf))                       # fingerprint is a DICT, do NOT slice
changed = d.get("fingerprint") != fp
d["fingerprint"] = fp
json.dump(d, open(sf, "w"))
print(f"   fingerprint patched (was_stale={changed}); completed preserved: {d['completed_stages']}")
PY

echo ">> [4/4] relaunching s06b (fixed code: per-rep outdir + ProcessPool + skip)"
LOG=/mnt/data/jglee/runs/fdh_5track/s06b_rerun_$(date +%s).log
echo "$LOG" > /tmp/s06b_logpath
nohup evoliez run -c configs/target.local.5track.yaml --resume --to s06b_interaction \
      > "$LOG" 2>&1 < /dev/null &
echo "   launched PID $! -> log: $LOG"
sleep 4
if pgrep -u jglee -f "$PAT" >/dev/null; then echo "   confirmed running"; else
  echo "   WARNING: not running; tail of log:"; tail -20 "$LOG"; fi
