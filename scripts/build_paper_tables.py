#!/usr/bin/env python
"""Paper Table 1 (per-candidate s10 summary) + ML-vs-MD enrichment, from the CLEAN
s10 provenance (reports_out/paper/*.json -- pulled from the server).

Honest hierarchy (the user's point-3): md_lite + NAC = PRIMARY; RBFE = confirmatory
(converged windows only); GBSA = auxiliary, EXCLUDED from ranking (per-mutant Boltz
5-sample structure confound). ML enriches MD-binding-valid (AUC ~0.83) but NOT
catalysis -- so the safe claim is "enriches binding-valid", not "predicts activity".

numpy-only. Run from repo root after pulling the provenance JSONs.
"""
import json
from pathlib import Path

import numpy as np

P = Path("reports_out/paper")
mc = json.load(open(P / "md_candidates.json"))


def _idx(fn):
    d = json.load(open(P / fn))
    return {c["candidate_id"]: c for c in (d if isinstance(d, list) else d.values())}


rr, vv = _idx("reranked_candidates.json"), _idx("validated_candidates.json")
allml = sorted(c["ml_score"] for c in rr.values() if c.get("ml_score") is not None)
pct = lambda ms: None if ms is None or not allml else round(
    100 * sum(m <= ms for m in allml) / len(allml), 1)

cands = sorted(mc["candidates"], key=lambda c: -(c.get("md_lite_score") or -9))


def gbsa_of(c):
    b = c.get("binding_dg")
    return b.get("gbsa") if isinstance(b, dict) else None


# ---- Table 1 (markdown + CSV) ----
hdr = ["candidate", "mutation", "md_lite", "ml_pctile", "NAC", "dNAC",
       "NAC_occ", "GBSA(aux)", "RBFE(confirm)", "MD_pass", "note"]
md_rows, csv_rows = [], [",".join(hdr)]
for c in cands:
    cid = c["candidate_id"]
    ms = (rr.get(cid) or {}).get("ml_score")
    nac_ok = str(c.get("nac_status") or "").startswith("valid")
    dnac = c.get("nac_delta_vs_wt")
    g = gbsa_of(c)
    rbfe = c.get("rbfe_ddg_bind")
    rbfe_s = ("NaN(unconverged)" if c.get("rbfe_mode") == "failed_softcore_ti_nan"
              else (f"{rbfe:+.3f}" if rbfe is not None else "--"))
    lead = (dnac or 0) > 0
    note = "; ".join(c.get("failure_reasons") or []) or (
        "** catalytic lead (only productive NAC)" if lead else "")
    mll = c.get("md_lite_score")
    cells = [cid, c["mutation_string"], (f"{mll:.3f}" if mll is not None else "--"),
             str(pct(ms)), "valid" if nac_ok else "INVALID(diffused)",
             (f"{dnac:+.2f}" if dnac is not None else "--"),
             (f"{c.get('nac_occupancy'):.2f}" if c.get("nac_occupancy") is not None else "--"),
             (f"{g:+.2f}" if g is not None else "--"), rbfe_s,
             "yes" if c.get("passed") else "no", note]
    md_rows.append("| " + " | ".join(cells) + " |")
    csv_rows.append(",".join('"%s"' % x if "," in str(x) else str(x) for x in cells))

table_md = (
    "# Paper Table 1 -- s10 per-candidate MD / NAC / GBSA / RBFE (clean run, n=9)\n\n"
    "Hierarchy: **md_lite + NAC = primary**; RBFE = confirmatory (converged only); "
    "GBSA = auxiliary, **excluded from ranking** (per-mutant Boltz structure confound). "
    "Sorted by md_lite (the primary binding-stability metric). dNAC = NAC occupancy "
    "minus WT (WT = %.2f).\n\n" % mc.get("wt_nac_occupancy", 0.0)
    + "| " + " | ".join(hdr) + " |\n| " + " | ".join("---" for _ in hdr) + " |\n"
    + "\n".join(md_rows) + "\n")
(P / "table1_candidates.md").write_text(table_md)
(P / "table1_candidates.csv").write_text("\n".join(csv_rows) + "\n")


# ---- ML-vs-MD enrichment (recomputed locally, reproducible) ----
def _rankavg(a):
    """Ranks with AVERAGED ties (like scipy.rankdata) -- argsort(argsort()) gives
    tied values distinct, order-dependent ranks, which makes Spearman unstable on
    tie-heavy data (e.g. nac_occ = mostly 0)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(xs, ys):
    p = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(p) < 3:
        return None, len(p)
    x = np.array([a for a, _ in p], float); y = np.array([b for _, b in p], float)
    return round(float(np.corrcoef(_rankavg(x), _rankavg(y))[0, 1]), 3), len(p)


def auc(s, l):
    p = [(a, b) for a, b in zip(s, l) if a is not None]
    pos = [a for a, b in p if b]; neg = [a for a, b in p if not b]
    if not pos or not neg:
        return None
    return round(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
                 / (len(pos) * len(neg)), 3)


R = []
for c in cands:
    cid = c["candidate_id"]
    R.append(dict(ml=(rr.get(cid) or {}).get("ml_score"),
                  stab=(vv.get(cid) or {}).get("stability_score"),
                  md_lite=c.get("md_lite_score"),
                  nac_valid=str(c.get("nac_status") or "").startswith("valid"),
                  nac_occ=c.get("nac_occupancy"), md_pass=bool(c.get("passed"))))
ml = [r["ml"] for r in R]
vr = [r for r in R if r["nac_valid"]]
lines = [
    "# ML-vs-MD enrichment (clean s10, n=9 -- suggestive, within-shortlist)", "",
    f"- AUC(ml_score -> MD-pass)        : {auc(ml, [r['md_pass'] for r in R])}"
    "   <- ML enriches MD-binding-valid",
    f"- AUC(ml_score -> NAC-valid)      : {auc(ml, [r['nac_valid'] for r in R])}"
    "   <- <0.5: ML does NOT enrich catalysis",
    f"- Spearman(ml_score, md_lite)     : {spearman(ml, [r['md_lite'] for r in R])}",
    f"- Spearman(ml_score, nac_occ|val) : {spearman([r['ml'] for r in vr], [r['nac_occ'] for r in vr])}"
    "   <- one-point-dominated (only the lead has nonzero NAC occ); NOT interpretable",
    f"- Spearman(stability, md_lite)    : {spearman([r['stab'] for r in R], [r['md_lite'] for r in R])}"
    "   <- s09 stability = best md_lite predictor", "",
    "**Safe claim: 'ML/s09 enriches MD-binding-valid candidates' (AUC ~0.83).**",
    "**NOT 'ML predicts activity'** -- ML does NOT enrich NAC-validity (AUC 0.357; the",
    "single highest-ML candidate is NAC-invalid/diffused). Catalytic NAC occupancy is too",
    "sparse (one nonzero candidate) to correlate. The MD/NAC layer adds orthogonal",
    "catalytic discrimination that ML alone misses -- which is precisely its value.",
]
(P / "ml_vs_md_summary.md").write_text("\n".join(lines) + "\n")

print(table_md)
print("\n".join(lines))
print("\nWROTE: reports_out/paper/{table1_candidates.md,table1_candidates.csv,ml_vs_md_summary.md}")
