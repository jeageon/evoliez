"""s10 molecular-dynamics validation HTML report builder.

Reads a finished run's on-disk MD provenance (reports/provenance/md_candidates.json
written by the s10 stage + _state.json meta + the WT Boltz complex) and renders one
standalone paper-grade HTML report:

- an HONEST "what actually ran" panel (actual solvent mode + actual simulated ns vs
  what was requested -- so a ~50 ps implicit SCREEN is never read as a 2 ns explicit
  production run),
- the real-ran / passed / skipped / failed breakdown (a skipped candidate is NOT
  "MD validated"),
- the catalytic-power (NAC) layer: WT baseline occupancy + per-candidate ΔNAC vs WT,
- per-candidate binding metrics, MM-PB/GBSA ΔG (when an Amber explicit run produced it),
  and the reaction geometry (transfer distance / angle),
- method citations.

Called from the s10 stage on completion; the data path is read-from-disk, so the
report reproduces with no stage re-run and no DB.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import _report_kit as kit

# status -> colour. skipped_* (neutral, NOT validated) grey; failed red.
SCOL = {"ok": "#1D9E75", "unstable": "#BA7517", "failed": "#993C1D"}


def _scolor(status: str) -> str:
    if not status:
        return "#888"
    if status.startswith("skipped"):
        return "#888"
    return SCOL.get(status, "#888")


def _f(x, nd=2, plus=False):
    if not isinstance(x, (int, float)):
        return "—"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def _dg_str(binding_dg) -> str:
    if not isinstance(binding_dg, dict) or not binding_dg:
        return "—"
    return ", ".join(f"{k}:{v:.1f}" for k, v in binding_dg.items())


def build_md_report_html(run_dir) -> str:
    RD = Path(run_dir)
    PROV = RD / "reports" / "provenance"
    p = PROV / "md_candidates.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run s10 first")
    data = json.loads(p.read_text())
    cands = data.get("candidates") or []
    req = data.get("requested") or {}
    counts = data.get("counts") or {}
    nac_on = bool(data.get("nac_enabled"))
    wt = data.get("wt_reference") or {}
    wt_nac = data.get("wt_nac_occupancy")
    meta = (json.loads((RD / "_state.json").read_text()).get("meta") or {}) \
        if (RD / "_state.json").exists() else {}

    n_total = counts.get("n_total", len(cands))
    n_ran = counts.get("n_real_ran", 0)
    n_pass = counts.get("n_passed", 0)
    n_skip = counts.get("n_skipped", 0)
    n_fail = counts.get("n_failed", 0)

    # what ACTUALLY ran (from the ran candidates), vs what was requested
    ran = [c for c in cands if not str(c.get("status", "")).startswith("skipped")
           and c.get("status") != "failed"]
    act_solvent = ran[0].get("solvent_mode") if ran else "—"
    act_ns = [c.get("simulation_time_ns") for c in ran
              if isinstance(c.get("simulation_time_ns"), (int, float))]
    act_ns_str = (f"{min(act_ns):.3f}–{max(act_ns):.3f}" if act_ns
                  and max(act_ns) != min(act_ns)
                  else (f"{act_ns[0]:.3f}" if act_ns else "—"))
    req_solv, req_ns = req.get("solvent", "—"), req.get("production_ns")
    plvl = req.get("protocol_level")
    capped = isinstance(plvl, int) and plvl < 3
    solvent_mismatch = (req_solv == "explicit" and act_solvent == "implicit")

    nac_cands = [c for c in cands if isinstance(c.get("nac_occupancy"), (int, float))]
    n_better = sum(1 for c in nac_cands
                   if isinstance(c.get("nac_delta_vs_wt"), (int, float))
                   and c["nac_delta_vs_wt"] > 0)

    cards = [
        kit.card("MD candidates", f"{n_total:,}", "sent to s10"),
        kit.card("actually ran", f"{n_ran:,}",
                 f"{100*n_ran/n_total:.0f}% real MD" if n_total else "",
                 SCOL["ok"] if n_ran else "#888"),
        kit.card("passed", f"{n_pass:,}", "binding-stable + intact"),
        kit.card("skipped (NOT validated)", f"{n_skip:,}",
                 "no full-atom / cofactor FF", "#888"),
        kit.card("failed", f"{n_fail:,}", "integration / assembly",
                 SCOL["failed"] if n_fail else "#888"),
    ]
    if nac_on:
        cards.append(kit.card("WT NAC occupancy",
                              _f(wt_nac, 3) if isinstance(wt_nac, (int, float))
                              and wt_nac >= 0 else "—",
                              "reaction-competent fraction"))
        cards.append(kit.card("beat WT (ΔNAC>0)", f"{n_better}/{len(nac_cands)}",
                              "more productive geometry",
                              SCOL["ok"] if n_better else "#888"))
    cards_html = "".join(cards)

    # honesty banner
    honesty = (
        f'<div class="note"><b>What actually ran.</b> Requested '
        f'<code>protocol_level={plvl}</code>, <code>solvent={req_solv}</code>, '
        f'<code>production_ns={req_ns}</code>. '
        f'Actual: <b>{kit.esc(act_solvent)}</b> solvent, '
        f'<b>{act_ns_str} ns</b> simulated per candidate.'
        + (f' protocol_level &lt; 3 CAPS production at 25,000 steps (~0.05 ns), so '
           f'this is a SHORT implicit-solvent SCREEN — not a {req_ns} ns production '
           f'run. Report it as such.' if capped else '')
        + (' solvent=explicit was requested but the OpenMM path runs implicit GBSA '
           '(no PME/box); explicit is the Amber tier only.' if solvent_mismatch else '')
        + '</div>'
    )

    # per-candidate table (ranked by md_lite desc, then ΔNAC)
    def _key(c):
        return (-(c.get("md_lite_score") or -9),
                -(c.get("nac_delta_vs_wt") or -9))
    rows = ""
    for c in sorted(cands, key=_key):
        st = c.get("status", "—")
        nac = c.get("nac") or {}
        rows += (
            f'<tr><td><b>{kit.esc(c.get("mutation_string"))}</b></td>'
            f'<td>{kit.pill(st, _scolor(st))}</td>'
            f'<td>{_f(c.get("md_lite_score"), 3)}</td>'
            f'<td>{"✓" if c.get("passed") else "·"}</td>'
            f'<td>{_f(c.get("ligand_rmsd_mean"))}</td>'
            f'<td>{_f(c.get("pocket_rmsd_mean"))}</td>'
            f'<td>{_f(c.get("hbond_occupancy"))}</td>'
            f'<td>{_f(c.get("nac_occupancy"), 3)}</td>'
            f'<td>{_f(c.get("nac_delta_vs_wt"), 3, plus=True)}</td>'
            f'<td>{_f(nac.get("distance_min"))}</td>'
            f'<td>{_f(nac.get("angle_mean"), 1)}</td>'
            f'<td>{_dg_str(c.get("binding_dg"))}</td></tr>')

    mdl_h = kit.histogram([c.get("md_lite_score") for c in cands], bins=20)
    nac_h = kit.histogram([c.get("nac_occupancy") for c in nac_cands], bins=20,
                          lo=0, hi=1) if nac_cands else {"labels": [], "counts": []}
    # ΔNAC vs md_lite scatter (one point per NAC candidate)
    scatter = [{"x": round(c.get("md_lite_score") or 0, 3),
                "y": round(c.get("nac_delta_vs_wt") or 0, 3),
                "c": SCOL["ok"] if (c.get("nac_delta_vs_wt") or 0) > 0 else "#993C1D",
                "m": c.get("mutation_string")} for c in nac_cands
               if isinstance(c.get("nac_delta_vs_wt"), (int, float))]

    pdb_text: Optional[str] = None
    try:
        from .s07_report import _read_pdb_text, find_wt_complex_pdb
        wp = find_wt_complex_pdb(RD / "complexes")
        pdb_text = _read_pdb_text(wp) if wp else None
    except Exception:  # noqa: BLE001
        pdb_text = None
    CAT = meta.get("catalytic_positions") or []
    DES = meta.get("designable_positions") or []

    nac_block = ""
    if nac_on:
        nac_block = f"""
<h2>Catalytic power — near-attack-conformation (NAC) occupancy</h2>
<div class="note">md_lite_score above measures BINDING stability only. NAC measures
REACTIVITY: the fraction of frames where the reacting atoms sit in a productive
geometry (transferring atom within the cutoff of the acceptor AND a near-linear
donor–transfer–acceptor angle). A mutant with <b>ΔNAC &gt; 0</b> has a geometrically
more productive active site than WT — the screenable proxy for catalytic-power short
of a QM/MM barrier. WT baseline occupancy = <b>{_f(wt_nac, 3) if isinstance(wt_nac,(int,float)) and wt_nac>=0 else "—"}</b>.</div>
<div class="grid2">
  <div><h3>NAC occupancy distribution</h3>
    <div class="chart-box"><canvas id="nac"></canvas></div>
    <div class="cap">Per-candidate reaction-competent frame fraction
    ({len(nac_cands)} scored). WT line marks the baseline to beat.</div></div>
  <div><h3>ΔNAC vs WT against binding stability</h3>
    <div class="chart-box"><canvas id="sc"></canvas></div>
    <div class="cap">Each point a candidate; green = beats WT reactivity (ΔNAC&gt;0).
    The desirable quadrant is upper-right: stable binding AND more productive geometry.</div></div>
</div>"""

    target = RD.name
    body = f"""
<h1>{kit.esc(target)} — s10 MD validation</h1>
<p class="sub">Short restrained implicit-solvent MD over the top reranked candidates:
ligand retention, pocket stability, key-contact / H-bond occupancy, and (when enabled)
the catalytic-power NAC reactive-geometry screen. Read the honesty panel below before
quoting any number as a production-MD result.</p>
{cards_html}
{honesty}

<h2>Per-candidate MD metrics</h2>
<div class="note">A <code>skipped_*</code> status means real MD never ran for that
candidate (CA-only structure, no per-mutant fold, or a cofactor the small-molecule FF
can't parameterize) — it is NEUTRAL for scoring and must NOT be counted as
"MD validated". Binding metrics + ΔG are the design ligand; NAC is the reaction.</div>
<div class="scroll"><table><thead><tr><th>mutation</th><th>status</th>
<th>md_lite</th><th>pass</th><th>ligRMSD Å</th><th>pktRMSD Å</th><th>hbond</th>
<th>NAC occ</th><th>ΔNAC</th><th>d_min Å</th><th>angle °</th><th>ΔG (kcal/mol)</th>
</tr></thead><tbody>{rows}</tbody></table></div>

<h3>MD-lite binding-stability score</h3>
<div class="chart-box" style="height:240px"><canvas id="mdl"></canvas></div>
<div class="cap">Weight-free binding-quality score (contacts + key-distance stability +
H-bond + catalytic geometry − ligand/pocket drift). Reactivity is reported separately
(below), never folded into this number.</div>
{nac_block}

<h2>WT complex — catalytic site &amp; design positions</h2>
{kit.legend([("catalytic (fixed)", "#d85a30"), ("designable", "#1D9E75"),
             ("NADP / formate (HETATM)", "#22b8cf")])}
{kit.viewer_block(pdb_text, "WT Boltz complex. Catalytic residues red, designable "
                  "positions green, cofactor + co-substrate (NADP / formate) cyan.")}

{kit.citations([
    'Eastman et al. (2017) <b>PLOS Comput. Biol.</b> 13:e1005659 — OpenMM 7 (GPU MD).',
    'Maier et al. (2015) <b>JCTC</b> 11:3696 — ff14SB protein force field.',
    'Wang et al. (2004) <b>J. Comput. Chem.</b> 25:1157 — GAFF; Jakalian et al. (2002) — AM1-BCC charges.',
    'Onufriev, Bashford &amp; Case (2004) <b>Proteins</b> 55:383 — OBC2 generalized-Born implicit solvent.',
    'Bruice &amp; Lightstone (1999) <b>Acc. Chem. Res.</b> 32:127 — near-attack conformations (NAC) and reactivity.',
    'Wohlwend et al. (2024) <b>bioRxiv</b> 2024.11.19.624167 — Boltz (complex prediction; the input pose).',
])}
{kit.footer(datetime.now().strftime("%Y-%m-%d %H:%M"), target,
            f"s10 · {n_ran}/{n_total} ran · {act_solvent} {act_ns_str} ns"
            + (f" · NAC on ({len(nac_cands)} scored)" if nac_on else " · NAC off"))}
"""

    scripts = kit.viewer_js(CAT, DES) + f"""
var MDL={json.dumps(mdl_h)}, NAC={json.dumps(nac_h)}, SC={json.dumps(scatter)},
    WT={json.dumps(wt_nac if isinstance(wt_nac,(int,float)) and wt_nac>=0 else None)};
function barOpts(xt){{return{{responsive:true,maintainAspectRatio:false,
  plugins:{{legend:{{display:false}}}},
  scales:{{x:{{title:{{display:true,text:xt,color:MUT}},ticks:{{color:MUT,maxTicksLimit:12}},grid:{{display:false}}}},
           y:{{ticks:{{color:MUT}},grid:{{color:GRID}},beginAtZero:true,title:{{display:true,text:'candidates',color:MUT}}}}}}}}}}
new Chart(mdl,{{type:'bar',data:{{labels:MDL.labels,datasets:[{{data:MDL.counts,backgroundColor:'#185fa5'}}]}},
  options:barOpts('md_lite_score')}});
if(document.getElementById('nac')){{
new Chart(nac,{{type:'bar',data:{{labels:NAC.labels,datasets:[{{data:NAC.counts,backgroundColor:'#1D9E75'}}]}},
  options:barOpts('NAC occupancy (frame fraction)')}});}}
if(document.getElementById('sc')){{
new Chart(sc,{{type:'scatter',data:{{datasets:[{{data:SC,pointBackgroundColor:SC.map(p=>p.c),
  pointBorderColor:SC.map(p=>p.c),pointRadius:5}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},
    tooltip:{{callbacks:{{label:function(c){{return c.raw.m+'  md_lite '+c.raw.x+' / ΔNAC '+c.raw.y;}}}}}}}},
    scales:{{x:{{title:{{display:true,text:'md_lite_score (binding stability)',color:MUT}},grid:{{color:GRID}},ticks:{{color:MUT}}}},
      y:{{title:{{display:true,text:'ΔNAC vs WT (reactivity)',color:MUT}},grid:{{color:GRID}},ticks:{{color:MUT}}}}}}}}}});}}
"""
    return kit.page(f"{target} — s10 MD validation", body, scripts=scripts)


def write_md_report(run_dir, out=None) -> Path:
    """Build + write the s10 MD report. Returns the output path."""
    RD = Path(run_dir)
    out = Path(out) if out else RD / "reports" / "s10_md_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_md_report_html(RD))
    return out
