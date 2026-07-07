#!/usr/bin/env python3
"""Summarise the s10 NAC catalytic screen: WT reference + each candidate's
near-attack status/occupancy, and ΔNAC vs WT. Reads per-candidate analysis.json
(written as each MD finishes) and, if present, the final md_candidates.json."""
import glob
import json
import os

BASE = "runs/srcar_3hp_full/md"


def _nac(d):
    nac = d.get("nac") or {}
    return (d.get("nac_status") or nac.get("status"),
            d.get("nac_occupancy") if d.get("nac_occupancy") is not None
            else nac.get("occupancy"))


rows = []
for aj in sorted(glob.glob(BASE + "/*/analysis.json")):
    cid = os.path.basename(os.path.dirname(aj))
    try:
        d = json.load(open(aj))
    except Exception as e:  # noqa: BLE001
        rows.append((cid, f"READ-ERR {e}", None, None, None))
        continue
    st, occ = _nac(d)
    rows.append((cid, st, occ, d.get("md_lite_status"), d.get("md_lite_score")))

wt = [r for r in rows if "wt" in r[0].lower()]
cand = [r for r in rows if "wt" not in r[0].lower()]
wt_occ = wt[0][2] if wt and isinstance(wt[0][2], (int, float)) else None

print(f"{'candidate':20s} {'nac_status':34s} {'occ':>6s} {'dNAC':>7s} {'md_lite':>8s}")
for cid, st, occ, mds, score in wt + cand:
    docc = (f"{occ - wt_occ:+.3f}" if isinstance(occ, (int, float))
            and wt_occ is not None and "wt" not in cid.lower() else "")
    occs = f"{occ:.3f}" if isinstance(occ, (int, float)) else str(occ)
    print(f"{cid:20s} {str(st):34s} {occs:>6s} {docc:>7s} {str(mds):>8s}")

# valid-status tally
valid = sum(1 for _, st, _, _, _ in cand if st and "valid" in str(st) and "invalid" not in str(st))
print(f"\ncandidates: {len(cand)}  |  valid-retention NAC: {valid}  |  WT occ: {wt_occ}")

mc = glob.glob("runs/srcar_3hp_full/reports/provenance/md_candidates.json")
print(f"md_candidates.json present: {bool(mc)} (final s10 output; ΔNAC ranking lands here)")
