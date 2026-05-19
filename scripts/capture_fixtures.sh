#!/usr/bin/env bash
# Collect REAL tool outputs from a smoke run into tests/fixtures/tool_outputs/
# captured/ so they can become parser regression tests. "Server execution is
# a real-backend validation-data collection step" (expert).
#
#   bash scripts/capture_fixtures.sh <run_dir>
set -euo pipefail
RUN="${1:?usage: capture_fixtures.sh <run_dir>}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tests/fixtures/tool_outputs/captured"
mkdir -p "$DEST"

copy_first() {  # <subdir> <name-pattern> <dest-name>
  local f
  f="$(find "$RUN/$1" -type f -name "$2" 2>/dev/null | head -1 || true)"
  if [ -n "${f:-}" ]; then
    cp "$f" "$DEST/$3"
    echo "captured $3  <-  $f"
  fi
}

copy_first complexes "*.cif"            boltz_real.cif
copy_first complexes "*.pdb"            boltz_real.pdb
copy_first complexes "confidence*.json" boltz_confidence_real.json
copy_first complexes "affinity*.json"   boltz_affinity_real.json
copy_first complexes "plddt*.npz"       boltz_plddt_real.npz
copy_first docking   "*vina_out.pdbqt"  vina_real.pdbqt
copy_first docking   "*gnina*.sdf"      gnina_real.sdf
copy_first docking   "rank1*.sdf"       diffdock_real.sdf
copy_first homologs  "*.m8"             homolog_real.m8
copy_first homologs  "*.sto"            jackhmmer_real.sto
copy_first validation "Dif_*.fxout"     foldx_real.fxout
copy_first validation "*.ddg"           rosetta_real.ddg

cat > "$DEST/MANIFEST.txt" <<EOF
Captured from: $RUN
Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Paste these back so parser regression tests can be added in
tests/test_tool_output_parsers.py (compare real vs expected).
EOF
echo ">> fixtures in $DEST  (commit + share for parser hardening)"
