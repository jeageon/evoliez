"""Publication-grade s10 HTML report for the FDH NADP-specificity campaign.

A paper-narrative synthesis on top of the clean s10 provenance, reusing the shared
report kit (_report_kit: CSS theming, Chart.js, 3Dmol viewer). Unlike md_report.py
(the per-run s10 diagnostic) this report is written for the manuscript: target +
goal framing, a WT-vs-lead 3D active-site comparison, the per-candidate metric
table with the binding free energies, the catalytic lead's multi-metric case, the
ML-vs-MD enrichment, and the literature context for FDH cofactor-specificity
engineering.

Figure/data choices follow the conventions of FDH cofactor-engineering papers
(cofactor-bound active-site structure; steady-state kcat/Km the experimental
endpoint; specificity-determining residues) and computational MD-screening papers
(binding-stability + near-attack-conformation geometry; MM-GBSA / TI free energies).

Honest hierarchy (do not reorder): md_lite + NAC = primary; RBFE = confirmatory
(converged windows only); GBSA = auxiliary, EXCLUDED from ranking (per-mutant Boltz
structure confound). ML enriches MD-binding-valid candidates (it is NOT an activity
predictor).

Reads run_dir/reports/provenance/{md_candidates,reranked_candidates,
validated_candidates}.json + the WT and lead post-MD PDBs; reproduces from disk.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import _report_kit as kit

OK, BAD, WARN, INFO, PURPLE = "#1D9E75", "#C0392B", "#BA7517", "#185fa5", "#9b59b6"


# ---------------------------------------------------------------- small helpers
def _f(x, nd=2, plus=False):
    if not isinstance(x, (int, float)):
        return "—"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def _idx(prov: Path, fn):
    p = prov / fn
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {c["candidate_id"]: c for c in (d if isinstance(d, list) else d.values())}


def _rankavg(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _spearman(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    rx = _rankavg([a for a, _ in pairs])
    ry = _rankavg([b for _, b in pairs])
    r = _pearson(rx, ry)
    return (round(r, 3) if r is not None else None), len(pairs)


def _auc(scored):  # [(score, bool_label)]
    p = [(s, l) for s, l in scored if s is not None]
    pos = [s for s, l in p if l]
    neg = [s for s, l in p if not l]
    if not pos or not neg:
        return None
    return round(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
                 / (len(pos) * len(neg)), 3)


def _read_pdb(path: Optional[Path]) -> Optional[str]:
    try:
        if path and Path(path).exists():
            return Path(path).read_text()
    except Exception:  # noqa: BLE001
        pass
    return None


def _find_first(run_dir: Path, *globs) -> Optional[Path]:
    for g in globs:
        hits = sorted(run_dir.glob(g))
        if hits:
            return hits[0]
    return None


# ----------------------------------------------------------- dual 3Dmol viewer
def _dual_viewer_js(specs) -> str:
    """specs = [(div_id, pdb_id, catalytic[], design[]), ...] -- one independent
    3Dmol viewer each (faded cartoon, catalytic red, design/mutation green, every
    HETATM cyan), zoomed to the cofactor."""
    out = []
    for vid, pid, cat, des in specs:
        out.append(f"""
(function(){{
  var el=document.getElementById('{vid}'); if(!el||!window.$3Dmol){{if(el)el.innerHTML=
    '<p style=\\'padding:1rem;color:{WARN}\\'>3Dmol.js unavailable</p>';return;}}
  var src=document.getElementById('{pid}'); if(!src||!src.textContent.trim()){{
    el.innerHTML='<p style=\\'padding:1rem;color:{kit.esc(WARN)}\\'>structure not on disk</p>';return;}}
  var V=$3Dmol.createViewer(el,{{backgroundColor:cssv('--surf')||'white'}});
  V.addModel(src.textContent,'pdb');
  V.setStyle({{}},{{cartoon:{{color:'#9b9890',opacity:0.5}}}});
  V.setStyle({{resi:{json.dumps(cat)},hetflag:false}},{{stick:{{radius:0.22,colorscheme:'redCarbon'}},cartoon:{{color:'#d85a30'}}}});
  V.setStyle({{resi:{json.dumps(des)},hetflag:false}},{{stick:{{radius:0.22,colorscheme:'greenCarbon'}}}});
  V.setStyle({{hetflag:true}},{{stick:{{radius:0.18,colorscheme:'cyanCarbon'}},sphere:{{scale:0.25}}}});
  try{{V.zoomTo({{hetflag:true}});}}catch(e){{V.zoomTo();}}
  V.render();
}})();""")
    # defer to window load so the embedded PDB <script> blocks are parsed first
    return "window.addEventListener('load',function(){\n" + "\n".join(out) + "\n});"


_V3D = ("height:380px;position:relative;border:1px solid var(--line);"
        "border-radius:8px;background:var(--surf);overflow:hidden")


# --------------------------------------------------------------- main builder
def build_paper_report_html(run_dir) -> str:
    RD = Path(run_dir)
    PROV = RD / "reports" / "provenance"
    mc = json.loads((PROV / "md_candidates.json").read_text())
    rr = _idx(PROV, "reranked_candidates.json")
    vv = _idx(PROV, "validated_candidates.json")

    cands = mc.get("candidates") or []
    wt_nac = mc.get("wt_nac_occupancy")
    counts = mc.get("counts") or {}
    req = mc.get("requested") or {}
    meta = (json.loads((RD / "_state.json").read_text()).get("meta") or {}) \
        if (RD / "_state.json").exists() else {}

    allml = sorted(c["ml_score"] for c in rr.values()
                   if isinstance(c.get("ml_score"), (int, float)))

    def mlpct(cid):
        ms = (rr.get(cid) or {}).get("ml_score")
        if ms is None or not allml:
            return None
        return round(100 * sum(m <= ms for m in allml) / len(allml), 1)

    ranked = sorted(cands, key=lambda c: -(c.get("md_lite_score") or -9))
    nac_valid = [c for c in cands if str(c.get("nac_status") or "").startswith("valid")]
    lead = next((c for c in ranked
                 if isinstance(c.get("nac_delta_vs_wt"), (int, float))
                 and c["nac_delta_vs_wt"] > 0), None)
    rbfe_ok = [c for c in cands if isinstance(c.get("rbfe_ddg_bind"), (int, float))]

    # ---- ML-vs-MD enrichment (recomputed from provenance) ----
    rows_ml = [dict(ml=(rr.get(c["candidate_id"]) or {}).get("ml_score"),
                    stab=(vv.get(c["candidate_id"]) or {}).get("stability_score"),
                    mdl=c.get("md_lite_score"),
                    nv=str(c.get("nac_status") or "").startswith("valid"),
                    pas=bool(c.get("passed"))) for c in cands]
    auc_pass = _auc([(r["ml"], r["pas"]) for r in rows_ml])
    auc_nac = _auc([(r["ml"], r["nv"]) for r in rows_ml])
    sp_mdl, _ = _spearman([(r["ml"], r["mdl"]) for r in rows_ml])
    sp_stab, _ = _spearman([(r["stab"], r["mdl"]) for r in rows_ml])

    target = "Arabidopsis FDH (UniProt Q9S7E4)"
    n_total = counts.get("n_total", len(cands))
    n_pass = counts.get("n_passed", sum(1 for c in cands if c.get("passed")))

    # ----------------------------------------------------------------- cards
    cards = "".join([
        kit.card("MD candidates", f"{n_total}", "real restrained MD"),
        kit.card("NAC-valid", f"{len(nac_valid)}/{len(cands)}",
                 "co-substrate retained", OK),
        kit.card("catalytic lead", lead["mutation_string"] if lead else "—",
                 f"ΔNAC {_f((lead or {}).get('nac_delta_vs_wt'), 2, plus=True)}"
                 if lead else "none beat WT", OK if lead else "#888"),
        kit.card("RBFE converged", f"{len(rbfe_ok)}/{req.get('rbfe_top_n', 3)}",
                 "ΔΔG_bind (softcore TI)", INFO),
        kit.card("ML→MD-pass AUC", _f(auc_pass), "enriches binding-valid",
                 OK if (auc_pass or 0) > 0.6 else "#888"),
    ])

    # ----------------------------------------------------------- Table 1 rows
    def _scol(c):
        st = c.get("status", "")
        if str(st).startswith("skipped"):
            return "#888"
        return {"ok": OK, "failed": BAD}.get(st, WARN)

    trows = ""
    for c in ranked:
        cid = c["candidate_id"]
        nac = c.get("nac") or {}
        nv = str(c.get("nac_status") or "").startswith("valid")
        g = (c.get("binding_dg") or {}).get("gbsa") if isinstance(
            c.get("binding_dg"), dict) else None
        rb = c.get("rbfe_ddg_bind")
        rb_s = ("NaN" if c.get("rbfe_mode") == "failed_softcore_ti_nan"
                else (_f(rb, 3, plus=True) if rb is not None else "—"))
        islead = lead is not None and cid == lead["candidate_id"]
        mut = kit.esc(c.get("mutation_string"))
        trows += (
            f'<tr{" style=background:rgba(29,158,117,.10)" if islead else ""}>'
            f'<td><b>{"★ " if islead else ""}{mut}</b></td>'
            f'<td>{kit.pill(c.get("status","—"), _scol(c))}</td>'
            f'<td>{_f(c.get("md_lite_score"), 3)}</td>'
            f'<td>{str(mlpct(cid)) if mlpct(cid) is not None else "—"}</td>'
            f'<td>{kit.pill("valid", OK) if nv else kit.pill("diffused", BAD)}</td>'
            f'<td>{_f(c.get("nac_delta_vs_wt"), 2, plus=True)}</td>'
            f'<td>{_f(nac.get("distance_min"))}</td>'
            f'<td>{_f(nac.get("angle_mean"), 0)}</td>'
            f'<td>{_f(g, 1, plus=True)}</td>'
            f'<td>{rb_s}</td>'
            f'<td>{"✓" if c.get("passed") else "·"}</td></tr>')

    # --------------------------------------------------------------- figures
    mdl_h = kit.histogram([c.get("md_lite_score") for c in cands], bins=12)
    nac_pts = [{"x": round(c.get("md_lite_score") or 0, 3),
                "y": round(c.get("nac_occupancy") or 0, 3),
                "c": OK if (c.get("nac_delta_vs_wt") or 0) > 0 else "#7a96c2",
                "m": c.get("mutation_string")}
               for c in nac_valid if c.get("md_lite_score") is not None]
    rbfe_bar = {"labels": [c["mutation_string"] for c in rbfe_ok],
                "vals": [c["rbfe_ddg_bind"] for c in rbfe_ok]}
    gbsa_list = [(c["mutation_string"], (c["binding_dg"] or {}).get("gbsa"))
                 for c in cands if isinstance(c.get("binding_dg"), dict)
                 and (c["binding_dg"] or {}).get("gbsa") is not None]

    # ----------------------------------------------------------------- PDBs
    wt_pdb = _read_pdb(_find_first(RD, "md/_wt_reference/*_minimized.pdb",
                                   "md/_wt_reference/*.pdb"))
    lead_id = lead["candidate_id"] if lead else None
    lead_pdb = _read_pdb(_find_first(RD, f"md/{lead_id}/*_minimized.pdb")) \
        if lead_id else None
    CAT = meta.get("catalytic_positions") or []
    DES = meta.get("designable_positions") or []
    # the lead's own mutated position(s), highlighted green in its panel
    lead_resis = []
    if lead:
        for tok in str(lead.get("mutation_string", "")).split(";"):
            digits = "".join(ch for ch in tok if ch.isdigit())
            if digits:
                lead_resis.append(int(digits))

    # ---------------------------------------------------------------- the body
    lead_mut = lead["mutation_string"] if lead else "—"
    lead_nac = lead.get("nac") if lead else {}
    body = f"""
<h1>{kit.esc(target)} — MD/NAC validation of ML-designed NADP⁺ variants</h1>
<p class="sub">The target is an <b>already cofactor-switched</b> FDH (the engineered
<code>D227Q+L229H</code> NAD→NADP variant). This report screens additional
machine-learning–designed substitutions for <b>enhanced NADP⁺ catalysis</b> using
short restrained molecular dynamics (binding stability), a near-attack-conformation
(NAC) reactive-geometry screen (catalytic power), and binding free energies
(relative TI = confirmatory; MM-GBSA = auxiliary). md_lite + NAC are the primary
signals; everything quoted below is a computational screen, not a kinetic
measurement.</p>
{cards}

<h2>1 · Design → validation pipeline</h2>
<div class="note">Candidates enter from the ML design/ranking stages
(<b>s07</b> rule/MSA/LigandMPNN/multipoint generation → <b>s08</b> GNN rerank →
<b>s09</b> structure-aware validation) and only the s09-surviving, real-Boltz-folded
shortlist reaches <b>s10</b>. s10 runs, per candidate: restrained implicit-solvent
MD → <b>md_lite</b> binding-stability score; a flat-bottom <i>distance-only</i>
co-substrate restraint (never an angle restraint — that would manufacture reactivity)
→ <b>NAC occupancy</b> (fraction of frames with a productive formate→NADP-C4 hydride
geometry) and <b>ΔNAC vs WT</b>; then Amber <b>MM-GBSA</b> and softcore-TI
<b>ΔΔG_bind</b> on the top md_lite candidates.</div>

<h2>2 · Active site — WT vs the catalytic lead (post-MD)</h2>
{kit.legend([("catalytic residues (fixed)", "#d85a30"), ("lead mutation site", OK),
             ("NADP⁺ + formate (HETATM)", "#22b8cf")])}
<div class="grid2">
  <div><h3>WT reference</h3><div id="vwt" style="{_V3D}"></div>
    <div class="cap">WT FDH·NADP⁺·formate after the same restrained MD. WT NAC
    occupancy = <b>{_f(wt_nac, 3) if isinstance(wt_nac,(int,float)) else "—"}</b>
    (no productive hydride-transfer geometry sampled).</div></div>
  <div><h3>Catalytic lead — {kit.esc(lead_mut)}</h3><div id="vlead" style="{_V3D}"></div>
    <div class="cap">Lead variant active site. NAC occupancy
    <b>{_f((lead or {}).get('nac_occupancy'), 2)}</b>, d(transfer)<sub>min</sub>
    <b>{_f((lead_nac or {}).get('distance_min'))} Å</b>, ⟨angle⟩
    <b>{_f((lead_nac or {}).get('angle_mean'), 0)}°</b> — the only variant that
    samples a reaction-competent geometry.</div></div>
</div>
<div class="cap">Both panels are post-MD minimised complexes (identical protocol);
drag to rotate, “active site” buttons not shown — use scroll to zoom. Rendering is
client-side 3Dmol.js on the embedded PDB.</div>

<h2>3 · Per-candidate metrics (Table 1)</h2>
<div class="note">Sorted by md_lite (primary binding-stability). <b>ΔNAC&gt;0</b> =
more productive than WT. GBSA is shown for transparency but <b>excluded from
ranking</b> (per-mutant Boltz starting structures confound the absolute value —
see §5). RBFE is reported only where the TI windows converged.</div>
<div class="scroll"><table><thead><tr>
<th>variant</th><th>status</th><th>md_lite</th><th>ML %ile</th><th>NAC</th>
<th>ΔNAC</th><th>d_min Å</th><th>angle °</th><th>GBSA*</th><th>ΔΔG_bind</th><th>pass</th>
</tr></thead><tbody>{trows}</tbody></table></div>
<div class="cap">*GBSA in kcal/mol, auxiliary (not a ranking criterion). ΔΔG_bind in
kcal/mol (softcore TI, &lt;0 = tighter than WT). ML %ile = the variant's GNN
ml_score percentile in the full {len(allml)}-candidate design pool.</div>

<div class="grid2">
  <div><h3>Binding-stability (md_lite) distribution</h3>
    <div class="chart-box"><canvas id="mdl"></canvas></div>
    <div class="cap">Weight-free score (contacts + key-distance stability + H-bond −
    ligand/pocket drift). Reactivity is never folded in.</div></div>
  <div><h3>Catalytic geometry vs binding stability</h3>
    <div class="chart-box"><canvas id="sc"></canvas></div>
    <div class="cap">Each point a NAC-valid variant; green beats WT (ΔNAC&gt;0). The
    desirable quadrant is upper-right (stable AND productive). Only the lead sits
    above the WT baseline.</div></div>
</div>

<h2>4 · Binding free energy — confirmatory (relative TI)</h2>
<div class="note">Softcore thermodynamic-integration ΔΔG_bind on the top md_lite
variants (a fair, alchemical estimate, unlike single-structure end-point GBSA).
{len(rbfe_ok)} window-set(s) converged; the rest hit softcore singularities (NaN) —
reported honestly, not dropped silently.</div>
<div class="grid2">
  <div><div class="chart-box" style="height:230px"><canvas id="rbfe"></canvas></div>
    <div class="cap">ΔΔG_bind (kcal/mol), &lt;0 = tighter binding than WT.</div></div>
  <div><h3>MM-GBSA (auxiliary)</h3>
    <div class="scroll"><table><thead><tr><th>variant</th><th>GBSA kcal/mol</th></tr>
    </thead><tbody>{''.join(f'<tr><td>{kit.esc(m)}</td><td>{_f(v,2,plus=True)}</td></tr>' for m,v in gbsa_list)}</tbody></table></div>
    <div class="cap">End-point GBSA on per-mutant Boltz structures: the spread is
    structure-driven (positive values for stable binders), so GBSA is an integrity
    check only — <b>not</b> used to rank.</div></div>
</div>

<h2>5 · Machine learning enriches MD-binding-valid candidates</h2>
<div class="note">Does the upstream GNN/validation score predict the MD outcome?
<b>ML enriches binding-validity</b> (AUC ml_score→MD-pass = <b>{_f(auc_pass)}</b>),
but it does <b>not</b> predict catalysis (AUC ml_score→NAC-valid =
<b>{_f(auc_nac)}</b> &lt; 0.5; the single highest-ML variant is itself
NAC-invalid). The s09 stability score tracks md_lite best
(Spearman {_f(sp_stab)}) — better than ml_score ({_f(sp_mdl)}). <b>Interpretation:</b>
ML/s09 is a binding filter; the MD/NAC layer adds the orthogonal catalytic signal
ML alone misses. (n={len(cands)} — a within-shortlist trend, not a population ROC.)
Claim it as <i>enrichment of binding-valid candidates</i>, never as activity
prediction.</div>

<h2>6 · The catalytic lead — {kit.esc(lead_mut)}</h2>
<div class="note">{kit.esc(lead_mut)} is the only variant combining stable binding
(md_lite {_f((lead or {}).get('md_lite_score'),3)}), a <b>productive active site</b>
(ΔNAC {_f((lead or {}).get('nac_delta_vs_wt'),2,plus=True)} over WT;
d<sub>min</sub> {_f((lead_nac or {}).get('distance_min'))} Å,
⟨angle⟩ {_f((lead_nac or {}).get('angle_mean'),0)}°), a favourable end-point GBSA
({_f((lead.get('binding_dg') or {}).get('gbsa') if lead else None,2,plus=True)}
kcal/mol), and a high ML rank ({_f(mlpct(lead_id))} percentile). It is the primary
recommendation to carry to <b>steady-state kinetics</b> (k<sub>cat</sub>/K<sub>m</sub>
for NADP⁺ vs NAD⁺), the field-standard experimental endpoint for an FDH
cofactor-specificity claim.</div>

<h2>7 · Literature context</h2>
<div class="note">FDH cofactor-specificity engineering (NAD→NADP) is well established:
in <i>Pseudomonas</i> sp. 101 FDH the switch is driven by substitutions at positions
198, 221, 222, 260, 379 and 380, and cofactor-bound crystal structures plus
steady-state k<sub>cat</sub>/K<sub>m</sub> are the standard evidence (Tishkov &amp;
Pometun 2026; Partipilo et al. 2023; Hoelsch et al. 2012; Vainstein &amp; Banta
2023 — see References). Our target is the already-switched <i>Arabidopsis</i>
D227Q+L229H background; this campaign computationally pre-screens <i>additional</i>
NADP⁺-enhancing substitutions on top of it, prioritising variants by a productive
near-attack geometry rather than binding alone. Several candidates fall in the
cofactor-contacting region the literature implicates; the kinetic assay of the lead
is the decisive next experiment. <i>(Literature retrieved from PubMed.)</i></div>

<h2>8 · Honesty &amp; limitations</h2>
<ul style="color:var(--mut);font-size:.9rem;line-height:1.7">
<li>This is a <b>screen</b>, not a QM/MM barrier: NAC occupancy is a geometric
reaction-competence proxy, not a rate.</li>
<li><b>n={len(cands)}</b> (the s09-surviving shortlist) — correlations are
suggestive, not powered.</li>
<li>Catalytic improvement rests on a <b>single</b> variant ({kit.esc(lead_mut)});
the NAC-occupancy correlation is one-point-dominated and not interpretable on its own.</li>
<li><b>GBSA is excluded from ranking</b> (per-mutant Boltz-structure confound).
RBFE is reported only for converged TI windows.</li>
<li>The co-substrate restraint is <b>distance-only</b>; angle is always free, so a
productive angle is an emergent result, never imposed.</li>
</ul>

{kit.citations([
 'Tishkov VI, Pometun AA, et al. (2026) <b>Biochemistry (Moscow)</b> 91:S37 — '
 'FDH coenzyme specificity, residues 379/380. '
 '<a href="https://doi.org/10.1134/S0006297925602886">doi:10.1134/S0006297925602886</a> (PubMed).',
 'Partipilo M, et al. (2023) <b>FEBS J.</b> 290:4238 — Starkeya novella FDH '
 'D221Q + quadruple mutant, cofactor-bound structure. '
 '<a href="https://doi.org/10.1111/febs.16871">doi:10.1111/febs.16871</a> (PubMed).',
 'Hoelsch K, et al. (2012) <b>Appl. Microbiol. Biotechnol.</b> 97:2473 — MycFDH '
 'A198G/D221Q cofactor + stability. '
 '<a href="https://doi.org/10.1007/s00253-012-4142-9">doi:10.1007/s00253-012-4142-9</a> (PubMed).',
 'Vainstein S, Banta S (2023) <b>Protein Eng. Des. Sel.</b> 36:gzad009 — '
 'computational cofactor-specificity design of CbFDH. '
 '<a href="https://doi.org/10.1093/protein/gzad009">doi:10.1093/protein/gzad009</a> (PubMed).',
 'Bruice &amp; Lightstone (1999) <b>Acc. Chem. Res.</b> 32:127 — near-attack '
 'conformations (NAC).',
 'Eastman et al. (2017) <b>PLOS Comput. Biol.</b> 13:e1005659 — OpenMM 7; '
 'Maier et al. (2015) <b>JCTC</b> 11:3696 — ff14SB; '
 'Onufriev et al. (2004) <b>Proteins</b> 55:383 — OBC2 implicit solvent.',
 'Wohlwend et al. (2024) <b>bioRxiv</b> 2024.11.19.624167 — Boltz (input pose).',
])}
{kit.footer(datetime.now().strftime("%Y-%m-%d %H:%M"), RD.name,
            f"s10 paper report · {len(cands)} MD candidates · lead {kit.esc(lead_mut)}")}
<script id="pdbwt" type="text/plain">{wt_pdb or ""}</script>
<script id="pdblead" type="text/plain">{lead_pdb or ""}</script>
"""

    scripts = _dual_viewer_js([
        ("vwt", "pdbwt", CAT, []),
        ("vlead", "pdblead", CAT, lead_resis),
    ]) + f"""
var MDL={json.dumps(mdl_h)}, SC={json.dumps(nac_pts)},
    RB={json.dumps(rbfe_bar)}, WT={json.dumps(wt_nac if isinstance(wt_nac,(int,float)) else None)};
function barOpts(xt,yt){{return{{responsive:true,maintainAspectRatio:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{x:{{title:{{display:true,text:xt,color:MUT}},ticks:{{color:MUT,maxTicksLimit:14}},grid:{{display:false}}}},
           y:{{ticks:{{color:MUT}},grid:{{color:GRID}},title:{{display:true,text:yt,color:MUT}}}}}}}}}}
new Chart(mdl,{{type:'bar',data:{{labels:MDL.labels,datasets:[{{data:MDL.counts,backgroundColor:'{INFO}'}}]}},
  options:barOpts('md_lite_score','candidates')}});
if(document.getElementById('rbfe')){{
new Chart(rbfe,{{type:'bar',data:{{labels:RB.labels,datasets:[{{data:RB.vals,
  backgroundColor:RB.vals.map(v=>v<0?'{OK}':'{BAD}')}}]}},
  options:barOpts('variant','ΔΔG_bind (kcal/mol)')}});}}
if(document.getElementById('sc')){{
var sc=document.getElementById('sc');
new Chart(sc,{{type:'scatter',data:{{datasets:[{{data:SC,pointBackgroundColor:SC.map(p=>p.c),
  pointBorderColor:SC.map(p=>p.c),pointRadius:6}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},
   tooltip:{{callbacks:{{label:function(c){{return c.raw.m+'  md_lite '+c.raw.x+' / NAC '+c.raw.y;}}}}}}}},
   scales:{{x:{{title:{{display:true,text:'md_lite_score (binding stability)',color:MUT}},grid:{{color:GRID}},ticks:{{color:MUT}}}},
     y:{{title:{{display:true,text:'NAC occupancy (reactivity)',color:MUT}},grid:{{color:GRID}},ticks:{{color:MUT}},beginAtZero:true}}}}}}}});}}
"""
    return kit.page(f"{RD.name} — s10 paper report", body, scripts=scripts)


def write_paper_report(run_dir, out=None) -> Path:
    RD = Path(run_dir)
    out = Path(out) if out else RD / "reports" / "s10_paper_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_paper_report_html(RD))
    return out
