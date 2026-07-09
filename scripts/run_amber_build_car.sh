#!/usr/bin/env bash
# V6-1 — build the CAR Amber system on the server. One-shot: sources the canonical
# Amber env then runs the builder against a complex PDB (protein + 3HP + ATP + Mg).
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPLEX="${1:-/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_v5_e4a/md/mut_00000/mut_00000_anchored.pdb}"
WORK="${WORK:-/mnt/data/jglee/v6_build_work/car}"
OUT="${OUT:-$REPO_DIR/reports/provenance/amber_system_manifest.json}"

# shellcheck disable=SC1091
source "$REPO_DIR/scripts/amber_env.sh"

echo "== V6-1 CAR Amber build =="
echo "repo    : $REPO_DIR"
echo "complex : $COMPLEX"
echo "work    : $WORK"
echo "tleap   : $(command -v tleap)"
echo "python  : $(command -v python)"
mkdir -p "$WORK" "$(dirname "$OUT")"

cd "$REPO_DIR"          # so params/atp_4minus/ATP.fixed.mol2 resolves
python "$REPO_DIR/scripts/build_amber_system.py" "$COMPLEX" --work "$WORK" --out "$OUT"
