"""Shared building blocks for the EvoLiEZ stage HTML reports (s08, s09, ...).

Matches the s07 / complex-report design system: CSS-variable theming (light + dark),
a 3Dmol.js structure viewer (PDB embedded as ``<script type="text/plain">``),
Chart.js graphs that read the CSS variables for dark-mode-correct colours, and a
citation footer. Self-contained: the gen scripts import these helpers and emit one
standalone HTML file (no network beyond the two pinned CDN libs).
"""
from __future__ import annotations

import html as _html
import json as _json
from typing import List, Optional, Sequence

CDN_3DMOL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"
CDN_CHARTJS = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"

CSS = """
:root{--bg:#fff;--fg:#1c1c1a;--mut:#5f5e5a;--line:#e7e5df;--surf:#f7f6f2;--accent:#185fa5}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a18;--fg:#ece9e3;--mut:#a8a69e;--line:#33322e;--surf:#232220;--accent:#85b7eb}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:2rem 1.4rem 4rem}
h1{font-size:1.55rem;margin:0 0 .2rem}
h2{font-size:1.16rem;margin:2.3rem 0 .7rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.3rem 0 .5rem}
a{color:var(--accent)}
.sub{color:var(--mut);font-size:.9rem;margin:0 0 1.2rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.6rem;margin:1rem 0}
.card{background:var(--surf);border-radius:8px;padding:.7rem .85rem;border:1px solid var(--line)}
.card .lab{font-size:.71rem;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.card .num{font-size:1.5rem;font-weight:650;margin:.15rem 0 0;line-height:1.1}
.card .sub2{font-size:.73rem;color:var(--mut);margin-top:.15rem}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1rem;margin:.5rem 0;font-size:.82rem;color:var(--mut)}
.lg{display:inline-flex;align-items:center;gap:.35rem}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block;flex:none}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media(max-width:720px){.grid2{grid-template-columns:1fr}}
.chart-box{position:relative;height:290px;margin:.5rem 0 .4rem}
.cap{font-size:.78rem;color:var(--mut);margin:.1rem 0 1rem}
.scroll{overflow-x:auto;margin:.5rem 0}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:.38rem .55rem;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:.74rem;text-transform:uppercase;letter-spacing:.03em;position:sticky;top:0;background:var(--bg)}
tbody tr:hover{background:var(--surf)}
.pill{display:inline-block;padding:.06rem .5rem;border-radius:10px;color:#fff;font-size:.71rem;font-weight:600}
.note{background:var(--surf);border-left:3px solid var(--accent);border-radius:0 6px 6px 0;padding:.7rem .95rem;margin:1rem 0;font-size:.9rem}
.note b{color:var(--fg)}
.cite{font-size:12px;color:var(--mut);line-height:1.85}
.cite b{color:var(--fg)}
#viewer{width:100%;height:460px;position:relative;border:1px solid var(--line);border-radius:8px;background:var(--surf);overflow:hidden}
.vbar{display:flex;flex-wrap:wrap;gap:.35rem;margin:.5rem 0}
.vb{background:var(--surf);border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:.3rem .65rem;font-size:.8rem;cursor:pointer}
.vb:hover{border-color:var(--accent)}
.foot{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--mut);font-size:.8rem}
code{background:var(--surf);padding:.05rem .3rem;border-radius:4px;font-size:.86em}
"""


def esc(s) -> str:
    return _html.escape(str(s if s is not None else ""))


def card(lab, val, sub, color: Optional[str] = None) -> str:
    st = f' style="color:{color}"' if color else ""
    return (f'<div class="card"><div class="lab">{esc(lab)}</div>'
            f'<div class="num"{st}>{esc(val)}</div>'
            f'<div class="sub2">{esc(sub)}</div></div>')


def legend(items: Sequence) -> str:
    return ('<div class="legend">' + "".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>{esc(lab)}</span>'
        for lab, c in items) + '</div>')


def pill(text, color) -> str:
    return f'<span class="pill" style="background:{color}">{esc(text)}</span>'


def histogram(values, bins: int = 20, lo=None, hi=None) -> dict:
    """Server-side histogram -> {labels, counts} for a Chart.js bar chart."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"labels": [], "counts": []}
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    w = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        i = min(bins - 1, max(0, int((v - lo) / w)))
        counts[i] += 1
    labels = [f"{lo + w * i:.2f}" for i in range(bins)]
    return {"labels": labels, "counts": counts}


def _strip_tags(html_str: str) -> str:
    """Lightweight HTML -> text for ClaimGuard linting (drops <script>/<style> and tags)."""
    import re as _re
    text = _re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", html_str)
    text = _re.sub(r"(?s)<[^>]+>", " ", text)
    return _html.unescape(text)


def _claimguard_gate(body: str, provenance, strict, allow, title: str) -> str:
    """ROADMAP_V3 B3 — enforce ClaimGuard on EVERY report that goes through ``page()``.

    Lints the rendered body against its claim provenance (``None`` -> the conservative
    floor verdict: no activity/kcat/stability claim is permitted without wet-lab
    evidence). On a violation:
      * strict (param ``strict=True`` or env ``EVOLIEZ_STRICT_CLAIMS``): raise — a hard
        gate for tests/CI, so an injected over-claim fails the build.
      * default: log a warning AND prepend a visible banner listing the flagged claims —
        structural + non-silent, but never breaks report generation.

    Returns a banner HTML fragment to prepend to the body ("" when clean).
    """
    import logging
    import os

    from evoliez.ranking.claim_guard import evaluate, lint_text

    verdict_allow = set(evaluate(provenance).allow())
    verdict_allow |= set(allow or ())
    viols = lint_text(_strip_tags(body), allow=sorted(verdict_allow))
    if not viols:
        return ""
    if strict is None:
        strict = os.environ.get("EVOLIEZ_STRICT_CLAIMS", "").lower() in (
            "1", "true", "yes", "on")
    detail = "\n  ".join(str(v) for v in viols)
    if strict:
        raise AssertionError(
            f"ClaimGuard: prohibited claim(s) in report {title!r}:\n  {detail}")
    logging.getLogger("evoliez.claim_guard").warning(
        "report %r has %d unguarded claim(s): %s", title, len(viols),
        "; ".join(str(v) for v in viols))
    items = "".join(f"<li>{esc(str(v))}</li>" for v in viols)
    return ('<div class="note" style="border-left-color:#c0392b">'
            f'<b>⚠ ClaimGuard: {len(viols)} unguarded claim(s)</b> — this report '
            'contains phrasing that exceeds its evidence provenance. Tighten the copy or '
            f'supply matching provenance.<ul>{items}</ul></div>')


def page(title: str, body: str, *, scripts: str = "", claim_provenance=None,
         strict: Optional[bool] = None, claim_allow: Optional[Sequence[str]] = None) -> str:
    body = _claimguard_gate(body, claim_provenance, strict, claim_allow, title) + body
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{esc(title)}</title><style>{CSS}</style></head><body>'
        f'<div class="wrap">{body}</div>'
        f'<script src="{CDN_CHARTJS}"></script>'
        f'<script src="{CDN_3DMOL}"></script>'
        '<script>function cssv(n){return getComputedStyle(document.documentElement)'
        ".getPropertyValue(n).trim()||'#888'}var MUT=cssv('--mut'),GRID=cssv('--line'),"
        "FG=cssv('--fg');Chart.defaults.color=MUT;Chart.defaults.font.family="
        "getComputedStyle(document.body).fontFamily;"
        f"{scripts}</script></body></html>")


def viewer_block(pdb_text: Optional[str], caption: str = "") -> str:
    if not pdb_text:
        return ('<div class="note">WT complex structure not available for this run '
                '(no Boltz prediction on disk) — 3D view omitted.</div>')
    cap = f'<div class="cap">{esc(caption)}</div>' if caption else ""
    return (
        '<div class="vbar">'
        "<button class=\"vb\" onclick=\"vStyle('cartoon')\">cartoon</button>"
        "<button class=\"vb\" onclick=\"vStyle('surface')\">surface</button>"
        '<button class="vb" onclick="vPocket()">active site</button>'
        '<button class="vb" onclick="vReset()">reset</button>'
        '<button class="vb" onclick="vSnap()">⤓ PNG</button></div>'
        '<div id="viewer"></div>' + cap +
        '<script id="pdbdata" type="text/plain">' + pdb_text + '</script>')


def viewer_js(catalytic: Sequence[int], design: Sequence[int]) -> str:
    """3Dmol viewer: faded cartoon, catalytic residues red sticks, design positions
    green sticks, every HETATM (NADP / formate) cyan ball-and-stick."""
    return ("""
var V=null;
function vInit(){
  var el=document.getElementById('viewer'); if(!el||!window.$3Dmol)return;
  V=$3Dmol.createViewer(el,{backgroundColor:cssv('--surf')||'white'});
  V.addModel(document.getElementById('pdbdata').textContent,'pdb');
  vStyle('cartoon'); V.zoomTo(); V.render();
}
function _base(){
  V.setStyle({},{cartoon:{color:'#9b9890',opacity:0.55}});
  V.setStyle({resi:__CAT__,hetflag:false},{stick:{radius:0.23,colorscheme:'redCarbon'},cartoon:{color:'#d85a30'}});
  V.setStyle({resi:__DES__,hetflag:false},{stick:{radius:0.18,colorscheme:'greenCarbon'}});
  V.setStyle({hetflag:true},{stick:{radius:0.18,colorscheme:'cyanCarbon'},sphere:{scale:0.2}});
}
function vStyle(m){ if(!V)return; _base();
  if(m==='surface'){V.removeAllSurfaces();V.addSurface($3Dmol.SurfaceType.VDW,{opacity:0.45,color:'#cfcdc6'},{hetflag:false});}
  else{V.removeAllSurfaces();}
  V.render(); }
function vPocket(){ if(!V)return; _base(); V.removeAllSurfaces();
  try{V.zoomTo({hetflag:true});}catch(e){V.zoomTo({resi:__CAT__});} V.render(); }
function vReset(){ if(!V)return; _base(); V.removeAllSurfaces(); V.zoomTo(); V.render(); }
function vSnap(){ if(!V)return; var a=document.createElement('a'); a.href=V.pngURI(); a.download='structure.png'; a.click(); }
window.addEventListener('load',vInit);
""".replace("__CAT__", _json.dumps(list(catalytic)))
   .replace("__DES__", _json.dumps(list(design))))


def citations(items: List[str]) -> str:
    return '<h2>References</h2><p class="cite">' + "\n".join(items) + '</p>'


def footer(generated: str, run_id: str, extra: str = "") -> str:
    return (f'<div class="foot">EvoLiEZ enzyme-design pipeline · {esc(run_id)} · '
            f'generated {esc(generated)}{(" · " + extra) if extra else ""}<br>'
            'Reproducible from on-disk provenance (reports/provenance/*.json); '
            'no stage re-run, no in-memory state.</div>')
