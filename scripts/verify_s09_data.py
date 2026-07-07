#!/usr/bin/env python
"""Anti-s06 Part-2 verification of s09 stored data (ThermoMPNN ddg_fold + redock
metrics) from evoliez.sqlite. Schema-discovering: lists tables and scans every row's
text (the per-candidate scores/details are JSON blobs in some column) for the keys,
then reports distributions. REAL paper-grade data must be distributed/non-trivial —
NOT all-0 (silent failure), NOT all-identical (free pass), NOT all-null.

Usage: verify_s09_data.py [evoliez.sqlite]
"""
import re
import sqlite3
import statistics
import sys

db = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track/evoliez.sqlite"
c = sqlite3.connect(db)
tabs = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("tables:", tabs)

blob_parts = []
for t in tabs:
    try:
        n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    except sqlite3.Error:
        continue
    print(f"  {t}: {n} rows")
    for r in c.execute(f"SELECT * FROM {t}"):
        for v in r:
            if isinstance(v, (str, bytes)) and len(v) > 20:
                blob_parts.append(v.decode() if isinstance(v, bytes) else v)
blob = "\n".join(blob_parts)


def nums(key):
    return [float(x) for x in re.findall(r'"%s"\s*:\s*(-?\d+\.?\d*)' % key, blob)]


def nulls(key):
    return len(re.findall(r'"%s"\s*:\s*null' % key, blob))


def strs(key):
    return re.findall(r'"%s"\s*:\s*"([^"]*)"' % key, blob)


def report(key):
    v, nl = nums(key), nulls(key)
    if not v and not nl:
        print(f"  {key}: NOT FOUND")
        return
    if v:
        print(f"  {key}: n={len(v)} null={nl} min={min(v):.3f} max={max(v):.3f} "
              f"mean={statistics.mean(v):.3f} std={statistics.pstdev(v):.3f} "
              f"near0={sum(1 for x in v if abs(x) < 1e-6)} "
              f"distinct={len(set(round(x, 3) for x in v))}")
    else:
        print(f"  {key}: n=0 null={nl}")


print("\n=== ThermoMPNN stability (must be distributed; near0/distinct sane) ===")
report("ddg_fold")
report("stability_score")
print("  stability_unavailable=true:",
      len(re.findall(r'"stability_unavailable"\s*:\s*true', blob)))

print("\n=== redock (P1.1/P1.3/consistency) ===")
report("redocking_consistency")
report("docking_score")
report("docking_uncertainty")
rcc = re.findall(r'"redock_context_chains"\s*:\s*(\[[^\]]*\])', blob)
nonempty = sum(1 for x in rcc if x.strip("[] "))
print(f"  redock_context_chains: {len(rcc)} found, {nonempty} non-empty (formate kept = P1.1)")

print("\n=== mechanism / geometry source (P2.6) ===")
ms = strs("mechanism_source")
print("  mechanism_source:", {s: ms.count(s) for s in set(ms)} or "NOT FOUND")
cg = strs("catalytic_geometry_source")
print("  catalytic_geometry_source:", {s: cg.count(s) for s in set(cg)} or "NOT FOUND")
bd = strs("boltz_delta_source")
print("  boltz_delta_source:", {s: bd.count(s) for s in set(bd)} or "NOT FOUND")
