#!/usr/bin/env python
"""Regenerate the s08 family-specific reranker HTML report from a finished run's
on-disk provenance -- WITHOUT re-running s08 (no GNN, no XGBoost re-fit).

s08 reranks the s07 mutation library with a family-conditioned model (XGBoost when
experimental labels exist, else a heuristic) and routes the top candidates to the
s08b real-Boltz fold + s09 validation. The report reads:
  * reports/provenance/reranked_candidates.json   (per-candidate ml_score + features)
  * reports/provenance/generated_candidates.json  (s07 generator, cross-ref)
  * reports/provenance/validated_candidates.json  (which advanced to the s08b fold)
  * _state.json meta                              (reranker model, positions)

Usage:  python scripts/gen_s08_report.py [RUN_DIR] [OUT_HTML]
"""
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_kit as kit  # noqa: E402

RD = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RD / "reports" / "s08_reranker_report.html"
PROV = RD / "reports" / "provenance"


def _load(name):
    p = PROV / name
    return json.loads(p.read_text()) if p.exists() else []


rr = _load("reranked_candidates.json")
if not rr:
    sys.exit(f"ERROR: {PROV}/reranked_candidates.json missing — re-run --to s08_reranker.")
gen = _load("generated_candidates.json")
val = _load("validated_candidates.json")
meta = (json.loads((RD / "_state.json").read_text()).get("meta") or {}) \
    if (RD / "_state.json").exists() else {}

gen_by_id = {g.get("candidate_id"): (g.get("generator") or g.get("source")) for g in gen}
fold_ids = {v["candidate_id"] for v in val if v.get("mechanism_source") == "mutant_boltz"}
for r in rr:
    r["generator"] = r.get("generator") or gen_by_id.get(r["candidate_id"]) or "—"
    r["folded"] = r["candidate_id"] in fold_ids

n = len(rr)
n_fold = len(fold_ids)
n_md = sum(bool(v.get("for_md")) for v in val)
ml = [r["ml_score"] for r in rr if r.get("ml_score") is not None]
model = meta.get("reranker_model", "heuristic")
nm_counts = Counter(r.get("n_mutations") for r in rr)
gen_counts = Counter(r["generator"] for r in rr)
GCOL = {"chemistry_rules": "#BA7517", "msa_sampler": "#378ADD", "ligandmpnn": "#1D9E75",
        "multipoint": "#9b59b6", "—": "#888780"}

# fold cutoff: the lowest ml_score still folded (the top-N boundary)
fold_scores = sorted((r["ml_score"] for r in rr if r["folded"] and r["ml_score"] is not None))
cutoff = fold_scores[0] if fold_scores else None

cards = "".join([
    kit.card("library reranked", f"{n:,}", "s07 candidate mutations"),
    kit.card("reranker model", kit.esc(model), "family-conditioned"),
    kit.card("ml_score range", f"{min(ml):.2f} … {max(ml):.2f}", f"median {statistics.median(ml):.2f}"),
    kit.card("advanced to fold", f"{n_fold}", "top by ml_score → s08b real Boltz"),
    kit.card("reach MD", f"{n_md}", "after s09 validation", "#185fa5"),
    kit.card("single / multi", f"{nm_counts.get(1,0)} / {n-nm_counts.get(1,0)}", "1-point vs ≥2-point"),
])

# feature means: folded-40 vs the rest (what the reranker favours)
FEATS = [("family_interaction_score", "family interaction"),
         ("interaction_gain", "interaction gain"),
         ("specificity_divergence", "specificity divergence"),
         ("msa_permissiveness", "MSA permissiveness"),
         ("conservation_penalty", "conservation penalty")]


def _mean(rows, k):
    v = [r.get(k) for r in rows if r.get(k) is not None]
    return statistics.mean(v) if v else 0.0


folded = [r for r in rr if r["folded"]]
rest = [r for r in rr if not r["folded"]]
feat_data = {"labels": [lab for _k, lab in FEATS],
             "folded": [round(_mean(folded, k), 3) for k, _ in FEATS],
             "rest": [round(_mean(rest, k), 3) for k, _ in FEATS]}

# top-40 table
rows = ""
for r in rr[:40]:
    g = r["generator"]
    rows += (
        f'<tr><td>{r.get("rank")}</td><td><b>{kit.esc(r.get("mutation_string"))}</b></td>'
        f'<td>{r.get("n_mutations")}</td>'
        f'<td><span class="sw" style="background:{GCOL.get(g,"#888")}"></span>{kit.esc(g)}</td>'
        f'<td><b>{(r.get("ml_score") or 0):.3f}</b></td>'
        f'<td>{(r.get("family_interaction_score") or 0):.2f}</td>'
        f'<td>{(r.get("interaction_gain") or 0):.2f}</td>'
        f'<td>{(r.get("specificity_divergence") or 0):.2f}</td>'
        f'<td>{(r.get("conservation_penalty") or 0):.2f}</td>'
        f'<td>{"✓" if r["folded"] else "—"}</td></tr>')

ml_h = kit.histogram(ml, bins=24)
nm_bar = {"labels": [f"{k}-point" for k in sorted(nm_counts)],
          "counts": [nm_counts[k] for k in sorted(nm_counts)]}

pdb_text = None
try:
    from evoliez.io.s07_report import _read_pdb_text, find_wt_complex_pdb
    p = find_wt_complex_pdb(RD / "complexes")
    pdb_text = _read_pdb_text(p) if p else None
except Exception as exc:  # noqa: BLE001
    print("WT PDB unavailable:", exc)
CAT = meta.get("catalytic_positions") or []
DES = meta.get("designable_positions") or []

target = RD.name
gleg = kit.legend([(k, GCOL.get(k, "#888")) for k in gen_counts])
body = f"""
<h1>{kit.esc(target)} — s08 family-specific reranker</h1>
<p class="sub">A family-conditioned model scores every s07 candidate (model:
<b>{kit.esc(model)}</b>); the top {n_fold} by <code>ml_score</code> advance to the
s08b real-Boltz fold and s09 validation. Features blend interaction gain, family
specificity divergence, MSA permissiveness and a conservation penalty.</p>
{cards}

<div class="note">The reranker is a <b>prioritiser</b>, not a verdict: it spends the
expensive real-Boltz fold budget on the most promising candidates. The downstream s09
multi-signal validation is what actually <b>tests</b> whether those folds retain NADP
binding — and (this run) shows the proxy-top picks largely do not.</div>

<div class="grid2">
  <div><h3>ml_score distribution</h3>
    <div class="chart-box"><canvas id="ml"></canvas></div>
    <div class="cap">{n:,} candidates. The top {n_fold} (ml_score ≥
    {("%.2f"%cutoff) if cutoff is not None else "—"}) are folded by Boltz.</div></div>
  <div><h3>library composition (mutation order)</h3>
    <div class="chart-box"><canvas id="nm"></canvas></div>
    <div class="cap">Single- and multi-point candidates from the s07 generators.</div></div>
</div>

<h3>What the reranker favours — mean feature value, folded-{n_fold} vs rest</h3>
<div class="chart-box" style="height:300px"><canvas id="ft"></canvas></div>
<div class="cap">Higher interaction gain / specificity divergence and lower
conservation penalty distinguish the selected folds.</div>

<h2>Generator mix</h2>
{gleg}
<p class="cap">{" · ".join(f"{kit.esc(k)}: {v}" for k, v in gen_counts.most_common())}</p>

<h2>Top {min(40,n)} candidates by ml_score</h2>
<div class="scroll"><table><thead><tr><th>rank</th><th>mutation</th><th>n</th>
<th>generator</th><th>ml_score</th><th>fam int</th><th>int gain</th><th>spec div</th>
<th>cons pen</th><th>folded</th></tr></thead><tbody>{rows}</tbody></table></div>

<h2>WT complex — catalytic site &amp; design positions</h2>
{kit.legend([("catalytic (fixed)", "#d85a30"), ("designable", "#1D9E75"),
             ("NADP / formate (HETATM)", "#22b8cf")])}
{kit.viewer_block(pdb_text, "Arabidopsis FDH (D227Q+L229H NADP-switch baseline); the "
                  "40 designable positions (green) are where the reranked mutations sit.")}

{kit.citations([
    'Chen &amp; Guestrin (2016) <b>KDD</b> 785 — XGBoost (gradient-boosted reranker).',
    'Dauparas et al. (2025) <b>Nat. Methods</b> 22:717 — LigandMPNN; (2022) <b>Science</b> 378:49 — ProteinMPNN (design generators).',
    'Khersonsky et al. (2018) <b>Mol. Cell</b> 72:178 — FuncLib (active-site multipoint libraries).',
    'Wohlwend et al. (2024) <b>bioRxiv</b> 2024.11.19.624167 — Boltz-1 (the real-fold backend the top picks are routed to).',
    'Rego &amp; Koes (2015) <b>Bioinformatics</b> 31:1322 — 3Dmol.js.',
])}
{kit.footer(datetime.now().strftime("%Y-%m-%d %H:%M"), target,
            f"s08 · {kit.esc(model)} reranker · {n:,} candidates → {n_fold} folds")}
"""

scripts = kit.viewer_js(CAT, DES) + f"""
var ML={json.dumps(ml_h)}, NM={json.dumps(nm_bar)}, FT={json.dumps(feat_data)};
function barOpts(xt,yt){{return{{responsive:true,maintainAspectRatio:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{x:{{title:{{display:true,text:xt,color:MUT}},ticks:{{color:MUT,maxTicksLimit:12}},grid:{{display:false}}}},
    y:{{ticks:{{color:MUT}},grid:{{color:GRID}},beginAtZero:true,title:{{display:true,text:yt||'candidates',color:MUT}}}}}}}}}}
new Chart(ml,{{type:'bar',data:{{labels:ML.labels,datasets:[{{data:ML.counts,backgroundColor:'#185fa5'}}]}},
  options:barOpts('ml_score')}});
new Chart(nm,{{type:'bar',data:{{labels:NM.labels,datasets:[{{data:NM.counts,backgroundColor:'#1D9E75'}}]}},
  options:barOpts('mutation order')}});
new Chart(ft,{{type:'bar',data:{{labels:FT.labels,datasets:[
  {{label:'folded',data:FT.folded,backgroundColor:'#185fa5'}},
  {{label:'not folded',data:FT.rest,backgroundColor:'#b4b2a9'}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:true,labels:{{color:MUT}}}}}},
    scales:{{x:{{ticks:{{color:MUT}},grid:{{display:false}}}},y:{{ticks:{{color:MUT}},grid:{{color:GRID}}}}}}}}}});
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(kit.page(f"{target} — s08 reranker", body, scripts=scripts))
print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
print(f"  n={n} model={model} fold={n_fold} md={n_md} 3D={'yes' if pdb_text else 'no'}")
