#!/usr/bin/env bash
# V6-2 — build WT + 2 CAR candidates and run explicit-solvent pmemd.cuda MD across
# GPUs 0,1,2 (one independent job per GPU). One-shot: sources the Amber env, strips
# the WT reference to a clean complex, then runs the scheduler driver.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN=/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_v5_e4a
WORKROOT="${WORKROOT:-/mnt/data/jglee/v6_md_work/car}"
GPUS="${GPUS:-0,1,2}"
PROD_NS="${PROD_NS:-0.1}"

# shellcheck disable=SC1091
source "$REPO_DIR/scripts/amber_env.sh"      # AMBER_GPU unset -> scheduler pins per job
mkdir -p "$WORKROOT"

# clean WT complex from the solvated minimized reference (keep protein + UNK ligands + MG)
WT_CLEAN="$WORKROOT/wt_clean.pdb"
WT_MIN="$RUN/md/_wt_reference/_wt_reference_minimized.pdb"
# keep protein + UNK ligands + MG + CONECT (ligand bonds — graph matching needs them)
awk 'substr($0,1,4)=="ATOM" || (substr($0,1,6)=="HETATM" && (substr($0,18,3)=="UNK" || substr($0,18,3)~/MG/)) || substr($0,1,6)=="CONECT" || substr($0,1,3)=="TER" || substr($0,1,3)=="END"' "$WT_MIN" > "$WT_CLEAN"
echo "WT clean complex: $(grep -c '^ATOM' "$WT_CLEAN") protein atoms, $(grep -c '^HETATM' "$WT_CLEAN") ligand/metal atoms"

cd "$REPO_DIR"
python "$REPO_DIR/scripts/run_amber_md_car.py" \
  --job "wt:$WT_CLEAN" \
  --job "mut_00000:$RUN/md/mut_00000/mut_00000_anchored.pdb" \
  --job "mut_00001:$RUN/md/mut_00001/mut_00001_anchored.pdb" \
  --gpus "$GPUS" --production-ns "$PROD_NS" --root "$WORKROOT" \
  --jsonl "$REPO_DIR/reports/provenance/amber_gpu_jobs.jsonl"
