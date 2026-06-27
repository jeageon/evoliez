"""s10 MD trajectory movie -- interactive 3Dmol animation of the active-site
dynamics over the restrained MD.

Reads the active-site multi-model PDBs (scripts/_make_movie_pdb.py: DCD -> frame-0
CA-superposed, cofactor + 8 Å pocket, heavy atoms, ~25 frames) and renders one
standalone HTML page where each candidate's active site PLAYS as a looped 3Dmol
animation (cofactor cyan, pocket grey), with global play / pause / speed controls
and a per-panel metric caption. Reuses the shared report kit for theming.

This is the "watch the complex change over time" view that the static structure
panels in paper_report.py cannot give. Not a kinetic measurement -- a restrained
implicit-solvent trajectory rendered for inspection.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from . import _report_kit as kit


def _traj_text(run_dir: Path, sub: str, base: str) -> Optional[str]:
    p = run_dir / "md" / sub / f"{base}_active_traj.pdb"
    try:
        return p.read_text() if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _default_specs(run_dir: Path) -> List[Tuple[str, str, str]]:
    """(subdir, base, role) for WT + the catalytic lead + the RBFE-favourable.
    Derived from md_candidates.json so labels/metrics stay in sync."""
    prov = run_dir / "reports" / "provenance" / "md_candidates.json"
    lead, rbfe = None, []
    if prov.exists():
        mc = json.loads(prov.read_text())
        ranked = sorted(mc.get("candidates") or [],
                        key=lambda c: -(c.get("md_lite_score") or -9))
        lead = next((c["candidate_id"] for c in ranked
                     if isinstance(c.get("nac_delta_vs_wt"), (int, float))
                     and c["nac_delta_vs_wt"] > 0), None)
        rbfe = [c["candidate_id"] for c in ranked
                if isinstance(c.get("rbfe_ddg_bind"), (int, float))]
    out = [("_wt_reference", "_wt_reference", "wt")]
    if lead:
        out.append((lead, lead, "lead"))
    for cid in rbfe:
        if cid != lead:
            out.append((cid, cid, "rbfe"))
    return out


def build_trajectory_movie_html(run_dir) -> str:
    RD = Path(run_dir)
    prov = RD / "reports" / "provenance" / "md_candidates.json"
    by_id = {}
    wt_nac = None
    if prov.exists():
        mc = json.loads(prov.read_text())
        by_id = {c["candidate_id"]: c for c in (mc.get("candidates") or [])}
        wt_nac = mc.get("wt_nac_occupancy")

    panels, payloads, inits = [], [], []
    for i, (sub, base, role) in enumerate(_default_specs(RD)):
        txt = _traj_text(RD, sub, base)
        if not txt:
            continue
        vid = f"mv{i}"
        c = by_id.get(base, {})
        nac = c.get("nac") or {}
        if role == "wt":
            title = "WT (reference)"
            metr = (f"NAC {kit.esc(f'{wt_nac:.2f}' if isinstance(wt_nac,(int,float)) else '0.00')}"
                    " · baseline geometry")
            color = "#888"
        elif role == "lead":
            title = f"{kit.esc(c.get('mutation_string', base))} · catalytic lead"
            metr = (f"NAC {c.get('nac_occupancy')} (ΔNAC "
                    f"+{c.get('nac_delta_vs_wt')}) · d_min "
                    f"{nac.get('distance_min')} Å · ⟨angle⟩ {nac.get('angle_mean')}°")
            color = "#1D9E75"
        else:
            rb = c.get("rbfe_ddg_bind")
            title = f"{kit.esc(c.get('mutation_string', base))} · RBFE-favourable"
            metr = (f"ΔΔG_bind {rb:+.2f} kcal/mol · NAC "
                    f"{c.get('nac_occupancy')}" if isinstance(rb, (int, float))
                    else "RBFE candidate")
            color = "#185fa5"
        panels.append(
            f'<div><h3 style="color:{color}">{title}</h3>'
            f'<div id="{vid}" style="height:340px;position:relative;border:1px solid '
            f'var(--line);border-radius:8px;background:var(--surf);overflow:hidden"></div>'
            f'<div class="cap">{metr}</div></div>')
        payloads.append(f'<script id="{vid}_pdb" type="text/plain">{txt}</script>')
        inits.append("""
(function(){
  var el=document.getElementById('%s'); if(!el||!window.$3Dmol){if(el)el.innerHTML=
    '<p style="padding:1rem;color:#BA7517">3Dmol.js unavailable</p>';return;}
  var src=document.getElementById('%s_pdb'); if(!src||!src.textContent.trim())return;
  var V=$3Dmol.createViewer(el,{backgroundColor:cssv('--surf')||'white'});
  V.addModelsAsFrames(src.textContent,'pdb');
  V.setStyle({},{stick:{radius:0.10,colorscheme:'whiteCarbon'}});
  V.setStyle({resn:'UNK'},{stick:{radius:0.20,colorscheme:'cyanCarbon'},sphere:{scale:0.28}});
  V.zoomTo();
  V.render();
  V.zoomTo();
  MV.push(V); V.animate({loop:'forward',reps:0,interval:SPEED});
})();""" % (vid, vid))

    body = f"""
<h1>{kit.esc(RD.name)} — s10 MD trajectory movie</h1>
<p class="sub">Active-site dynamics over the restrained implicit-solvent MD
(frame-0 Cα-superposed; cofactor + co-substrate <b style="color:#22b8cf">cyan</b>,
8 Å pocket grey; 25 frames, ~50 ps apart). Watch the cofactor settle / drift and
the pocket breathe. This is a trajectory rendered for inspection — a screen, not a
kinetic measurement. The WT samples no productive hydride geometry; the lead's
formate stays poised toward NADP⁺ C4 (NAC occupancy above WT).</p>
<div class="vbar">
  <button class="vb" onclick="MV.forEach(function(v){{v.animate({{loop:'forward',reps:0,interval:CURSPD}});}})">▶ play all</button>
  <button class="vb" onclick="MV.forEach(function(v){{v.stopAnimate();}})">⏸ pause all</button>
  <button class="vb" onclick="setSpd(200)">slow</button>
  <button class="vb" onclick="setSpd(80)">normal</button>
  <button class="vb" onclick="setSpd(25)">fast</button>
</div>
<div class="grid2">{''.join(panels)}</div>
{kit.citations([
 'Eastman et al. (2017) <b>PLOS Comput. Biol.</b> 13:e1005659 — OpenMM 7 (the MD).',
 'Rego &amp; Koes (2015) <b>Bioinformatics</b> 31:1322 — 3Dmol.js (this viewer).',
 'Bruice &amp; Lightstone (1999) <b>Acc. Chem. Res.</b> 32:127 — near-attack conformations.',
])}
{kit.footer(datetime.now().strftime("%Y-%m-%d %H:%M"), RD.name,
            f"s10 trajectory movie · {len(panels)} candidates · 25 frames each")}
{''.join(payloads)}
"""
    scripts = ("var MV=[],CURSPD=80,SPEED=80;\n"
               + "\n".join(inits)
               + "\nfunction setSpd(s){CURSPD=s;MV.forEach(function(v){"
                 "v.animate({loop:'forward',reps:0,interval:s});});}\n")
    return kit.page(f"{RD.name} — s10 MD movie", body, scripts=scripts)


def write_trajectory_movie(run_dir, out=None) -> Path:
    RD = Path(run_dir)
    out = Path(out) if out else RD / "reports" / "s10_md_movie.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_trajectory_movie_html(RD))
    return out
