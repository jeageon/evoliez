#!/usr/bin/env python
"""Regenerate the s09 non-MD validation HTML report from a finished run's on-disk
provenance -- WITHOUT re-running s09 (no docking, no ThermoMPNN, no GPU).

s09 is the multi-signal FAILURE-DETECTION stage: each candidate WITH a real s08b
Boltz mutant complex is graded by four INDEPENDENT tests (GNINA reference-local
redocking, DiffDock unconstrained/global plausibility, Boltz d_ligand_iptm binding
confidence, catalytic-geometry preservation) plus ThermoMPNN ΔΔG fold stability.
The report reads:
  * reports/provenance/validated_candidates.json   (per-candidate s09 scores + class)
  * reports/provenance/reranked_candidates.json     (s08 ml_score rank, cross-ref)
  * reports/provenance/generated_candidates.json    (s07 generator, cross-ref)
  * _state.json meta                                (catalytic / designable positions)
  * the WT Boltz complex under complexes/           (3D viewer)

Usage:  python scripts/gen_s09_report.py [RUN_DIR] [OUT_HTML]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _report_kit as kit  # noqa: E402

RD = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jglee/runs/fdh_5track")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else RD / "reports" / "s09_validation_report.html"
PROV = RD / "reports" / "provenance"


def _load(name):
    p = PROV / name
    return json.loads(p.read_text()) if p.exists() else []


val = _load("validated_candidates.json")
if not val:
    sys.exit(f"ERROR: {PROV}/validated_candidates.json missing/empty — did s09 run?")
rerank = _load("reranked_candidates.json")
gen = _load("generated_candidates.json")
meta = (json.loads((RD / "_state.json").read_text()).get("meta") or {}) \
    if (RD / "_state.json").exists() else {}

rank_by_id = {r["candidate_id"]: r.get("rank") for r in rerank}
gen_by_id = {g.get("candidate_id"): (g.get("generator") or g.get("source")) for g in gen}

CLASS = ["pass", "caution", "penalty", "reject"]
CCOL = {"pass": "#1D9E75", "caution": "#BA7517", "penalty": "#D85A30", "reject": "#993C1D"}
CDESC = {"pass": "retained · no escape · binding+mechanism intact",
         "caution": "DiffDock-only escape (kept, re-verify)",
         "penalty": "escape corroborated by one structural signal",
         "reject": "escape + binding drop + catalytic disruption"}

mb = [v for v in val if v.get("mechanism_source") == "mutant_boltz"]   # real-Boltz fold set


def _rmsd(v, m):
    return ((v.get("redock") or {}).get(m) or {}).get("rmsd_to_reference")


def _ddconf(v):
    return ((v.get("redock") or {}).get("diffdock") or {}).get("score")


def _f(x, nd=1, plus=False):
    if not isinstance(x, (int, float)):
        return "—"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


n_val, n_pass, n_md = len(val), sum(bool(v.get("passed")) for v in val), \
    sum(bool(v.get("for_md")) for v in val)
ccount = {c: sum(1 for v in mb if v.get("failure_mode") == c) for c in CLASS}
unavail = sum(1 for v in val if v.get("stability_unavailable"))

# ---- summary cards --------------------------------------------------------- #
cards = "".join([
    kit.card("candidates validated", f"{n_val:,}", "from the s08 reranked library"),
    kit.card("passed non-MD", f"{n_pass:,}", f"{100*n_pass/n_val:.0f}% survived all filters"),
    kit.card("advanced to MD", f"{n_md:,}", "s10 explicit-solvent budget"),
    kit.card("real-Boltz mutants", f"{len(mb)}", "folded + 4-signal graded (s08b)"),
    kit.card("confirmed failures", f"{ccount['reject']}", "triple-corroborated reject",
             CCOL["reject"]),
    kit.card("ThermoMPNN ΔΔG", f"{n_val-unavail}/{n_val}", "real scores (0 unavailable)"
             if unavail == 0 else f"{unavail} unavailable"),
])

# ---- the four independent tests + thresholds (task: surface the tuning) ----- #
TESTS = [
    ("GNINA", "reference-local redocking", "pose retention near the Boltz pose",
     "RMSD &le; 3.0 &Aring;", "#185fa5"),
    ("DiffDock", "unconstrained / global", "does the pose survive a blind global search?",
     "escape if RMSD &gt; 4.0 &Aring;", "#9b59b6"),
    ("Boltz Δ", "structure-conditioned confidence", "mutant vs WT ligand-iptm",
     "drop if d_ligand_iptm &lt; &minus;0.05", "#1D9E75"),
    ("catalytic geometry", "mechanism preservation", "active-site distance shift vs WT",
     "disrupted if penalty &gt; 2.5", "#D85A30"),
]
tests_html = "".join(
    f'<tr><td><b>{n}</b></td><td>{role}</td><td style="text-align:left;color:var(--mut)">{q}</td>'
    f'<td style="text-align:left"><code>{thr}</code></td></tr>'
    for n, role, q, thr, _c in TESTS)

# ---- multi-signal table (40 real-Boltz mutants) ---------------------------- #
mb_sorted = sorted(mb, key=lambda v: (CLASS.index(v.get("failure_mode") or "caution"),
                                      -(_rmsd(v, "diffdock") or 0)), reverse=False)
rows = ""
for v in mb_sorted:
    cls = v.get("failure_mode") or "—"
    g, d, dc = _rmsd(v, "gnina"), _rmsd(v, "diffdock"), _ddconf(v)
    dcol = CCOL["reject"] if (d or 0) > 4 else "inherit"
    rows += (
        f'<tr><td><b>{kit.esc(v.get("mutation_string"))}</b></td>'
        f'<td>{rank_by_id.get(v["candidate_id"], "—")}</td>'
        f'<td>{_f(g)}</td><td style="color:{dcol}">{_f(d)}</td>'
        f'<td>{_f(dc, 2)}</td>'
        f'<td>{_f(v.get("d_ligand_iptm"), 2, plus=True)}</td>'
        f'<td>{_f(v.get("catalytic_geometry_penalty"), 1)}</td>'
        f'<td>{_f(v.get("ddg_fold"), 2)}</td>'
        f'<td>{kit.pill(cls, CCOL.get(cls, "#888"))}</td></tr>')

# ---- chart data ------------------------------------------------------------ #
ddg_h = kit.histogram([v.get("ddg_fold") for v in val], bins=22)
con_h = kit.histogram([v.get("redocking_consistency") for v in val], bins=20, lo=0)
scatter = [{"x": round(_rmsd(v, "gnina") or 0, 2), "y": round(_rmsd(v, "diffdock") or 0, 2),
            "c": CCOL.get(v.get("failure_mode"), "#888"),
            "m": v.get("mutation_string")} for v in mb]
classbar = {"labels": CLASS, "counts": [ccount[c] for c in CLASS],
            "colors": [CCOL[c] for c in CLASS]}

# ---- 3D viewer (WT complex) ------------------------------------------------ #
pdb_text = None
try:
    from evoliez.io.s07_report import _read_pdb_text, find_wt_complex_pdb
    p = find_wt_complex_pdb(RD / "complexes")
    pdb_text = _read_pdb_text(p) if p else None
except Exception as exc:  # noqa: BLE001
    print("WT PDB unavailable:", exc)
CAT = meta.get("catalytic_positions") or []
DES = meta.get("designable_positions") or []

# ---- assemble -------------------------------------------------------------- #
target = RD.name
body = f"""
<h1>{kit.esc(target)} — s09 non-MD validation</h1>
<p class="sub">Multi-signal failure-detection over the reranked mutant library:
ThermoMPNN ΔΔG fold stability + four independent pose/binding/mechanism tests on every
real-Boltz mutant complex. DiffDock is <b>not</b> treated as ground truth for a large
charged cofactor (NADP) — a DiffDock-only escape is <i>caution</i>; a failure is a
corroborated escape.</p>
{cards}

<h2>The four independent tests</h2>
<div class="note">s09 is a failure-<b>detection</b> stage, not a single docking score.
GNINA's autobox sits on the candidate's own Boltz pose, so its reproduction is
reference-local (partly self-fulfilling); DiffDock is the independent global check.
A pose counts as a confirmed failure only when its DiffDock escape is corroborated by
the structure-conditioned Boltz binding signal and/or the catalytic geometry.</div>
<div class="scroll"><table><thead><tr><th>test</th><th>type</th><th>question</th>
<th>threshold (tunable)</th></tr></thead><tbody>{tests_html}</tbody></table></div>

<h2>Outcome distribution</h2>
{kit.legend([(f"{c} — {CDESC[c]}", CCOL[c]) for c in CLASS])}
<div class="grid2">
  <div><h3>Failure-mode class (real-Boltz mutants)</h3>
    <div class="chart-box"><canvas id="cls"></canvas></div>
    <div class="cap">All {len(mb)} folded mutants graded. GNINA alone (~2 Å, post
    autobox fix) would pass every one; the failures surface only in the corroborating
    signals.</div></div>
  <div><h3>ThermoMPNN ΔΔG of folding (kcal/mol)</h3>
    <div class="chart-box"><canvas id="ddg"></canvas></div>
    <div class="cap">Open-source ProteinMPNN-based stability predictor (FoldX/Rosetta
    license-free). Positive = destabilising; all {n_val-unavail} candidates scored.</div></div>
</div>

<h3>GNINA (local) vs DiffDock (global) redocking RMSD</h3>
<div class="chart-box" style="height:340px"><canvas id="sc"></canvas></div>
<div class="cap">Each point a folded mutant, coloured by failure class. Left of the
vertical line (GNINA ≤ 3 Å) = local pose retained; above the horizontal line
(DiffDock &gt; 4 Å) = global escape. The cluster top-left (retained locally, escaped
globally) is the signature of a Boltz pose that independent blind docking will not
re-select.</div>

<h3>Redocking consistency (worst-of-both)</h3>
<div class="chart-box" style="height:240px"><canvas id="con"></canvas></div>
<div class="cap">consistency = max(0, 1 − worst-RMSD/4). Low for the folded mutants
(DiffDock-driven), high for the WT-frame proxy candidates.</div>

<h2>Per-mutant signals ({len(mb)} real-Boltz folds)</h2>
<div class="scroll"><table><thead><tr><th>mutation</th><th>ml-rank</th>
<th>GNINA Å</th><th>DiffDock Å</th><th>DD conf</th><th>d_iptm</th><th>cat pen</th>
<th>ΔΔG</th><th>class</th></tr></thead><tbody>{rows}</tbody></table></div>

<h2>WT complex — catalytic site &amp; design positions</h2>
{kit.legend([("catalytic (fixed)", "#d85a30"), ("designable", "#1D9E75"),
             ("NADP / formate (HETATM)", "#22b8cf")])}
{kit.viewer_block(pdb_text, "Arabidopsis FDH (D227Q+L229H NADP-switch baseline). "
                  "Catalytic Arg290·His338·Gln319·Asn152·Ile128 in red; the 40 "
                  "designable positions in green; NADP + formate as cyan sticks.")}

{kit.citations([
    'Dieckhaus, Brocidiacono, Coventry &amp; Kuhlman (2024) <b>PNAS</b> 121:e2314853121 — ThermoMPNN (transfer-learned ΔΔG of folding).',
    'McNutt et al. (2021) <b>J. Cheminform.</b> 13:43 — GNINA (CNN-scored molecular docking, <code>--autobox_ligand</code>).',
    'Corso, Stärk, Jing, Barzilay &amp; Jaakkola (2023) <b>ICLR</b> — DiffDock (diffusion generative blind docking).',
    'Wohlwend et al. (2024) <b>bioRxiv</b> 2024.11.19.624167 — Boltz-1 (open biomolecular complex structure prediction; ipTM / pLDDT confidence).',
    'Dauparas et al. (2025) <b>Nat. Methods</b> 22:717 — LigandMPNN; Dauparas et al. (2022) <b>Science</b> 378:49 — ProteinMPNN.',
    'Eswaramoorthy, Bonanno, Burley &amp; Swaminathan (2006) <b>PNAS</b> 103:8157 — FDH catalytic mechanism / NAD(P) binding geometry.',
    'Rego &amp; Koes (2015) <b>Bioinformatics</b> 31:1322 — 3Dmol.js; Chen &amp; Guestrin (2016) <b>KDD</b> — XGBoost.',
])}
{kit.footer(datetime.now().strftime("%Y-%m-%d %H:%M"), target,
            f"s09 · ThermoMPNN ΔΔG · gnina+diffdock redock · {len(mb)} real-Boltz folds")}
"""

scripts = kit.viewer_js(CAT, DES) + f"""
var SC={json.dumps(scatter)}, CB={json.dumps(classbar)},
    DG={json.dumps(ddg_h)}, CN={json.dumps(con_h)};
function barOpts(xt){{return{{responsive:true,maintainAspectRatio:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{x:{{title:{{display:true,text:xt,color:MUT}},ticks:{{color:MUT,maxTicksLimit:12}},grid:{{display:false}}}},
           y:{{ticks:{{color:MUT}},grid:{{color:GRID}},beginAtZero:true,title:{{display:true,text:'candidates',color:MUT}}}}}}}}}}
new Chart(cls,{{type:'bar',data:{{labels:CB.labels,datasets:[{{data:CB.counts,backgroundColor:CB.colors}}]}},
  options:barOpts('failure class')}});
new Chart(ddg,{{type:'bar',data:{{labels:DG.labels,datasets:[{{data:DG.counts,backgroundColor:'#185fa5'}}]}},
  options:barOpts('ΔΔG (kcal/mol)')}});
new Chart(con,{{type:'bar',data:{{labels:CN.labels,datasets:[{{data:CN.counts,backgroundColor:'#7f77dd'}}]}},
  options:barOpts('redocking consistency')}});
new Chart(sc,{{type:'scatter',data:{{datasets:[{{data:SC,pointBackgroundColor:SC.map(p=>p.c),
  pointBorderColor:SC.map(p=>p.c),pointRadius:5}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},
    tooltip:{{callbacks:{{label:function(c){{return c.raw.m+'  GNINA '+c.raw.x+'Å / DiffDock '+c.raw.y+'Å';}}}}}}}},
    scales:{{x:{{title:{{display:true,text:'GNINA reference-local RMSD (Å)',color:MUT}},min:0,
      grid:{{color:GRID}},ticks:{{color:MUT}}}},
      y:{{title:{{display:true,text:'DiffDock global RMSD (Å)',color:MUT}},min:0,
      grid:{{color:GRID}},ticks:{{color:MUT}}}}}}}}}});
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(kit.page(f"{target} — s09 validation", body, scripts=scripts))
print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
print(f"  classes: {ccount} | passed {n_pass}/{n_val} | MD {n_md} | 3D: {'yes' if pdb_text else 'no'}")
