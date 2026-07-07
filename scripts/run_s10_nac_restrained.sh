#!/usr/bin/env bash
# Launch the RESTRAINED-NAC s10 production run (NAC-3): the formate co-substrate
# retention restraint is ON, so the catalytic NAC measures active-site geometry rather
# than implicit-solvent diffusion. Idempotent + sentinel-guarded (launches at most once);
# meant to be invoked by the hourly server-readiness cron ONLY when the shared box is
# free. Reuses run_s10_paper.sh for the full env + GPU + fingerprint guard.
set -euo pipefail
SENTINEL=/tmp/fdh_nac_restrained_launched
CFG=${1:-configs/target.local.5track.yaml}
RD=${2:-/mnt/data/jglee/runs/fdh_5track}
cd /mnt/data/jglee/EvoLiEZ
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
GUARD_PY="$CONDA_PREFIX/bin/python"

if [ -f "$SENTINEL" ]; then
  echo "ALREADY-LAUNCHED ($SENTINEL exists); not re-launching."; exit 0
fi

# 1. Turn the retention restraint ON in the config (idempotent).
"$GUARD_PY" - "$CFG" <<'PY'
import sys
p = sys.argv[1]; L = open(p).read().splitlines()
if any("restrain_cosubstrate" in x for x in L):
    print("  restraint already in config"); raise SystemExit
out = []
for i, ln in enumerate(L):
    out.append(ln)
    if ln.strip() == "reactive_geometry:":
        ind = "        "
        for nxt in L[i + 1:]:
            if nxt.strip():
                ind = " " * (len(nxt) - len(nxt.lstrip())); break
        out += [ind + "restrain_cosubstrate: true",
                ind + "template_cosubstrate_placement: true",
                ind + "restraint_radius_A: 5.0", ind + "restraint_k: 2.0"]
open(p, "w").write("\n".join(out) + "\n"); print("  restraint enabled in config")
PY

# 2. Reconcile fingerprint (config changed) + reset s10 so it re-runs (no purge).
"$GUARD_PY" scripts/check_run_fingerprint.py "$CFG" "$RD" --patch
"$GUARD_PY" - "$RD" <<'PY'
import json, sys
st = sys.argv[1] + "/_state.json"
d = json.load(open(st)); cs = d.get("completed_stages", [])
if "s10_md" in cs:
    cs.remove("s10_md"); d["completed_stages"] = cs
    json.dump(d, open(st, "w"), indent=2); print("  s10_md reset -> will re-run")
PY

# 3. Launch (run_s10_paper.sh re-checks the fingerprint [now matching] and nohups it).
touch "$SENTINEL"
echo "LAUNCHING restrained-NAC run ..."
bash scripts/run_s10_paper.sh "$CFG" "$RD"
