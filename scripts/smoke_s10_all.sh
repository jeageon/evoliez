#!/usr/bin/env bash
# =============================================================================
# Comprehensive s10 MD-layer smoke suite (server).
#
# Exercises EVERY s10 capability with tiny (ps-scale, 1-2 replica, ~100-frame)
# runs and prints a PASS/FAIL matrix. This is a functional smoke, NOT the real
# 50ns x 5 production. Idempotent. Shared-A6000 friendly: picks the freest GPU,
# single-threaded host tools (OMP capped), ~20 min total (the tier-3 explicit +
# 2-replica + MM-PB/GBSA step alone is ~15 min; the other five are seconds).
#
#   1. tier-2  OpenMM real MD (GPU/OpenCL) ........ default screening engine
#   2. tier-3  Amber ligand param (AM1-BCC) ....... parameterization foundation
#   3. tier-3  Amber implicit-GB engine ........... run_md_amber (implicit)
#   4. tier-3  Amber explicit + replicas + MMPBSA . rigorous path + dG_bind
#   5. Phase-B TI engine (pmemd.cuda softcore) .... alchemical dV/dl
#   6. Phase-B hybrid mutation topology ........... softcore_setup_py3 + masks
#
# Usage: bash scripts/smoke_s10_all.sh        (run from the repo on the server)
# =============================================================================
set -u

REPO=${EVOLIEZ_REPO:-/mnt/data/jglee/EvoLiEZ}
AMBERTOOLS_ENV=${AMBERTOOLS_ENV:-/mnt/data/jglee/anaconda3/envs/AmberTools25}
PMEMD_BUILD=${PMEMD_BUILD:-/mnt/data/jglee/pmemd24_src/build}
PY=${EVOLIEZ_PY:-/mnt/data/jglee/miniforge3/envs/evoliez-cu124/bin/python}
BOLTZ_PY=${BOLTZ_PY:-/mnt/data/jglee/envs/boltz/bin/python}
SMOKE_TIMEOUT=${SMOKE_TIMEOUT:-1500}   # tier-3 explicit+MMPBSA alone is ~15 min

# --- env: AmberTools (tleap/antechamber/cpptraj/ndfes) + pmemd.cuda ---------
# shellcheck disable=SC1090
[ -f "$PMEMD_BUILD/pathscripts/amber.sh" ] && source "$PMEMD_BUILD/pathscripts/amber.sh"
export AMBERHOME=$AMBERTOOLS_ENV
export PATH=$AMBERHOME/bin:$PATH
export LD_LIBRARY_PATH=$AMBERHOME/lib:${LD_LIBRARY_PATH:-}
export EVOLIEZ_AMBERTOOLS_BIN=$AMBERHOME/bin
export EVOLIEZ_PMEMD_CUDA=$PMEMD_BUILD/src/pmemd/src/pmemd.cuda_SPFP
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH=$REPO/src:${PYTHONPATH:-}   # use the freshly-scp'd repo src

# --- pick the freest GPU (evoliez-cu124 torch is cu130/GPU-broken on the 12.4
#     driver, so query with the boltz cu124 torch; fall back to GPU 0) --------
unset CUDA_VISIBLE_DEVICES
GPU=$("$BOLTZ_PY" -c "import torch
f=[(torch.cuda.mem_get_info(i)[0],i) for i in range(torch.cuda.device_count())]
print(sorted(f,reverse=True)[0][1] if f else 0)" 2>/dev/null || echo 0)
export CUDA_VISIBLE_DEVICES=${GPU:-0}

PROT=$(find /mnt/data/jglee/runs/fdh_5track -name "hom_012_boltz_input_model_0.pdb" 2>/dev/null | head -1)
[ -z "${PROT:-}" ] && { echo "FATAL: no test protein PDB found under runs/fdh_5track"; exit 2; }

echo "=============================================================="
echo " s10 smoke suite"
echo "   repo=$REPO  GPU=$CUDA_VISIBLE_DEVICES  OMP=$OMP_NUM_THREADS"
echo "   pmemd=$EVOLIEZ_PMEMD_CUDA"
echo "   prot=$(basename "$PROT")"
echo "=============================================================="
cd "$REPO" || exit 2

echo ">>> evoliez src import sanity ..."
"$PY" -c "import evoliez.adapters.amber_engine, evoliez.adapters.openmm_engine, evoliez.config; print('  src import OK')" \
  || { echo "FATAL: evoliez src import failed (see above)"; exit 2; }

RESULTS=/tmp/s10_smoke_results.txt
: > "$RESULTS"
run() {                       # name  logfile  pass_regex  cmd...
  local name="$1" log="$2" passre="$3"; shift 3
  local t0 verdict=FAIL
  t0=$SECONDS
  echo ">>> [$name] running ..."
  if timeout "$SMOKE_TIMEOUT" "$@" > "$log" 2>&1; then
    grep -qE "$passre" "$log" && verdict=PASS
  fi
  echo "$name $verdict $((SECONDS - t0))s" >> "$RESULTS"
  echo "    [$name] $verdict ($((SECONDS - t0))s)  log=$log"
}

# 1. tier-2 OpenMM real MD on the GPU (OpenCL auto-fallback). #1 solvent honesty
#    + #2 real-trajectory geometry.
run tier2_openmm        /tmp/s10_1_openmm.log "VERIFY: PASS" \
    env VERIFY_FORCE_CPU=0 "$PY" scripts/verify_s10_md_real.py "$PROT"

# 2. tier-3 Amber ligand parameterization (pdb4amber+antechamber gaff2+tleap).
run amber_param         /tmp/s10_2_param.log "SMOKE: PASS" \
    env "$PY" scripts/smoke_amber_param.py "$PROT"

# 3. tier-3 Amber implicit-GB full engine -> MDResult.
run amber_implicit      /tmp/s10_3_impl.log "VERIFY: PASS" \
    env VERIFY_SOLVENT=implicit VERIFY_REPLICAS=1 "$PY" scripts/verify_amber_engine.py "$PROT"

# 4. tier-3 Amber explicit (ff19SB/OPC/NPT) + 2 replicas + MM-PB/GBSA dG_bind.
#    Also leaves /tmp/verify_amber (explicit complex + equil.rst) for #5.
run amber_explicit_mmpbsa /tmp/s10_4_expl.log "VERIFY: PASS" \
    env VERIFY_SOLVENT=explicit VERIFY_REPLICAS=2 "$PY" scripts/verify_amber_engine.py "$PROT"

# 5. Phase-B alchemical TI engine (softcore, explicit) -> DV/DL.
run ti_engine           /tmp/s10_5_ti.log "SMOKE: PASS" \
    env "$PY" scripts/smoke_amber_ti.py

# 6. Phase-B hybrid mutation topology (vendored Py3 softcore_setup) -> masks.
run softcore_topology   /tmp/s10_6_sc.log "SMOKE: PASS" \
    env "$PY" scripts/smoke_amber_softcore.py "$PROT" 84

# --- key findings (one salient line per smoke) -------------------------------
echo
echo "------------------------- key findings -----------------------"
grep -hE "solvent_mode|key_distances|VERIFY: PASS - real" /tmp/s10_1_openmm.log 2>/dev/null | head -2 | sed 's/^/  openmm:  /'
grep -hE "prmtop: True|gaff" /tmp/s10_2_param.log 2>/dev/null | head -1 | sed 's/^/  param:   /'
grep -hE "key_distances|binding_dg|solvent_mode" /tmp/s10_4_expl.log 2>/dev/null | head -3 | sed 's/^/  explicit:/'
grep -hE "DV/DL samples" /tmp/s10_5_ti.log 2>/dev/null | head -1 | sed 's/^/  ti:      /'
grep -hE "scmask=|generated:" /tmp/s10_6_sc.log 2>/dev/null | head -2 | sed 's/^/  topo:    /'

# --- matrix ------------------------------------------------------------------
echo
echo "==================== s10 SMOKE MATRIX ========================"
nfail=0; ntot=0
while read -r name verdict dur; do
  printf "  %-24s %-5s %s\n" "$name" "$verdict" "$dur"
  ntot=$((ntot + 1)); [ "$verdict" != PASS ] && nfail=$((nfail + 1))
done < "$RESULTS"
echo "=============================================================="
echo "RESULT: $((ntot - nfail))/$ntot passed"
exit "$nfail"
