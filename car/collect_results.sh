#!/usr/bin/env bash
# Collect the SrCAR/3-HP production deliverables from the server run dir into car/results/.
# Run at completion (after s11). Pulls reports, final ranking, key structures, MD results.
set -euo pipefail
LOCAL="/Users/jg/Documents/EvoLiEZ/car/results"
REMOTE="evo:/mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_full"
mkdir -p "$LOCAL"/{reports,structures,ranking,md}

# stage reports (HTML) + provenance
rsync -az --include='*/' --include='*.html' --include='*.csv' --exclude='*' \
  "$REMOTE/reports/" "$LOCAL/reports/" 2>/dev/null || true
# final ranking / evidence cards / candidate provenance
rsync -az "$REMOTE/reports/provenance/" "$LOCAL/ranking/" 2>/dev/null || true
for f in evidence_cards.json final_library.csv final_ranking.json triage_report.html paper_report.html; do
  rsync -az "$REMOTE/$f" "$LOCAL/ranking/" 2>/dev/null || true
  rsync -az "$REMOTE/reports/$f" "$LOCAL/ranking/" 2>/dev/null || true
done
# MD results (s10)
rsync -az "$REMOTE"/**/md_candidates.json "$LOCAL/md/" 2>/dev/null || true
find_md=$(ssh evo 'find /mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_full -name "md_candidates.json" 2>/dev/null | head -1')
[ -n "$find_md" ] && rsync -az "evo:$find_md" "$LOCAL/md/" 2>/dev/null || true
# WT complex structure (top model)
wt=$(ssh evo 'ls /mnt/data/jglee/EvoLiEZ_car/runs/srcar_3hp_full/complexes/boltz/**/predictions/**/wt_boltz_input_model_0.pdb 2>/dev/null | head -1')
[ -n "$wt" ] && rsync -az "evo:$wt" "$LOCAL/structures/wt_complex_model_0.pdb" 2>/dev/null || true

echo "Collected into $LOCAL:"; find "$LOCAL" -type f | sed "s#$LOCAL/#  #" | head -40
