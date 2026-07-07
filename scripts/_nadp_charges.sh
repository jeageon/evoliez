#!/usr/bin/env bash
# One-time: derive AM1-BCC partial charges for the -3 NADP at its Boltz (bound) geometry.
# maxcyc=0 = single-point SCF at the input geometry -- skips the gas-phase minimization
# that (a) is slow and (b) would distort the tri-anionic phosphate group. We KEEP the
# default AM1 + SCF-convergence settings (the stripped minimal namelist failed SCF);
# only maxcyc is overridden. Output mol2 is in the INPUT atom order (no atom-name map).
set -euo pipefail
source /mnt/data/jglee/miniforge3/etc/profile.d/conda.sh
conda activate evoliez-cu124
AMBERTOOLS_ENV=/mnt/data/jglee/anaconda3/envs/AmberTools25
export AMBERHOME=$AMBERTOOLS_ENV
export PATH=$AMBERTOOLS_ENV/bin:$PATH
export LD_LIBRARY_PATH=$AMBERTOOLS_ENV/lib:${LD_LIBRARY_PATH:-}
cd /tmp/nadp_amber
rm -f NAP.fixed.mol2 ac3.log sqm.in sqm.out
S=$(date +%s)
antechamber -i ligand.sdf -fi sdf -o NAP.fixed.mol2 -fo mol2 \
  -c bcc -nc -3 -at gaff2 -rn LIG \
  -ek "qm_theory='AM1', maxcyc=0, scfconv=1.d-10, ndiis_attempts=700, printcharges=1" \
  >ac3.log 2>&1 || true
echo "  wall: $(( $(date +%s) - S ))s"
echo "  --- sqm.in &qmmm ---"; sed -n '/&qmmm/,/\//p' sqm.in 2>/dev/null
echo "  --- sqm SCF result ---"; grep -iE "SCF|converg|Heat of formation|Total Mulliken" sqm.out 2>/dev/null | tail -4
if [ -f NAP.fixed.mol2 ]; then
  python - <<'PY'
L=open("/tmp/nadp_amber/NAP.fixed.mol2").read().splitlines()
i=L.index("@<TRIPOS>ATOM"); j=L.index("@<TRIPOS>BOND")
q=[float(l.split()[-1]) for l in L[i+1:j]]
print(f"  OK: {len(q)} atoms, charge sum = {sum(q):+.4f} (target -3.0)  range [{min(q):+.3f},{max(q):+.3f}]")
PY
else
  echo "  NO mol2 produced"; tail -6 ac3.log
fi
