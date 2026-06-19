"""Self-contained HTML report for the s04 protein-ligand complex prediction.

Presents the Boltz-2 co-folded complex the way structure-prediction papers do:
an interactive 3D viewer (3Dmol.js, coloured by pLDDT), the per-residue pLDDT
confidence track, the PAE (predicted aligned error) matrix with chain blocks,
the inter-chain ipTM / affinity summary, and the diffusion-ensemble spread.
Everything is derived from the ACTUAL Boltz outputs (confidence JSON, plddt/pae
npz, affinity JSON, the model PDB), so it works for any complex. This top
section is the viz-agnostic DATA layer.
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

# AlphaFold pLDDT confidence bands (0-100) + canonical colours.
PLDDT_BANDS = [
    (90, 100, "very high", "#0053D6"),
    (70, 90, "confident", "#65CBF3"),
    (50, 70, "low", "#FFDB13"),
    (0, 50, "very low", "#FF7D45"),
]


def _np():
    import numpy as np
    return np


def _load_npz(path: str):
    np = _np()
    a = np.load(path)
    return np.asarray(a[a.files[0]])


def _parse_pdb_chains(pdb_text: str) -> List[dict]:
    """Per-chain token layout in the SAME order Boltz tokenises (protein = one
    token per residue, ligand = one token per atom), so chain spans line up with
    the pLDDT / PAE token axes. Returns [{id, kind, start, n}] over tokens."""
    chains: List[dict] = []
    cur = None
    seen_res = None
    for ln in pdb_text.splitlines():
        rec = ln[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        cid = ln[21]
        kind = "protein" if rec == "ATOM" else "ligand"
        if cur is None or cur["id"] != cid:
            cur = {"id": cid, "kind": kind, "start": sum(c["n"] for c in chains),
                   "n": 0}
            chains.append(cur)
            seen_res = set()
        if kind == "protein":
            ri = ln[22:27].strip()
            if ri not in seen_res:           # one token per residue
                seen_res.add(ri)
                cur["n"] += 1
        else:                                # one token per atom
            cur["n"] += 1
    return chains


def _downsample(mat, max_dim: int = 220):
    """Block-mean a square matrix down to <= max_dim per side (keeps the PAE
    chain-block structure while shrinking the embed)."""
    np = _np()
    n = mat.shape[0]
    if n <= max_dim:
        return mat, 1
    f = (n + max_dim - 1) // max_dim
    m = (n // f) * f
    sm = mat[:m, :m].reshape(m // f, f, m // f, f).mean(axis=(1, 3))
    return sm, f


def compute_complex_stats(pred_dir: str, name: Optional[str] = None) -> dict:
    """Everything the s04 report needs, from a Boltz predictions/<name>/ dir:
    best (highest-confidence) model + its pLDDT/PAE, per-chain confidence, the
    affinity head, the diffusion-ensemble confidence spread, and the model PDB
    text for the inline 3D viewer."""
    np = _np()
    confs = sorted(glob.glob(os.path.join(pred_dir, "confidence_*model_*.json")))
    if not confs:
        raise FileNotFoundError(f"no Boltz confidence JSON in {pred_dir}")
    if name is None:
        name = re.sub(r"^confidence_(.+)_model_\d+\.json$", r"\1",
                      os.path.basename(confs[0]))
    ens = {}
    for f in confs:
        i = int(re.search(r"model_(\d+)", f).group(1))
        ens[i] = json.load(open(f))
    best = max(ens, key=lambda i: ens[i].get("confidence_score", 0))
    bc = ens[best]

    plddt = _load_npz(os.path.join(pred_dir, f"plddt_{name}_model_{best}.npz"))
    pae = _load_npz(os.path.join(pred_dir, f"pae_{name}_model_{best}.npz"))
    if float(plddt.max()) <= 1.5:            # Boltz emits 0-1; AF bands are 0-100
        plddt = plddt * 100.0
    pdb_text = open(os.path.join(pred_dir, f"{name}_model_{best}.pdb")).read()
    chains = _parse_pdb_chains(pdb_text)

    aff = {}
    af = os.path.join(pred_dir, f"affinity_{name}.json")
    if os.path.exists(af):
        aff = json.load(open(af))

    # symmetric inter-chain ipTM (Boltz stores it row-keyed and asymmetric)
    pci = bc.get("pair_chains_iptm", {})
    keys = sorted(pci.keys(), key=int) if pci else []
    pair = {}
    for a in keys:
        for b in keys:
            v = max(pci.get(a, {}).get(b, 0.0), pci.get(b, {}).get(a, 0.0))
            pair[f"{a},{b}"] = round(v, 3)

    pae_ds, factor = _downsample(pae)
    chain_meta = []
    for c in chains:
        sl = plddt[c["start"]:c["start"] + c["n"]]
        chain_meta.append({
            "id": c["id"], "kind": c["kind"], "start": c["start"], "n": c["n"],
            "plddt_mean": round(float(sl.mean()), 1) if len(sl) else 0.0,
            "ptm": round(bc.get("chains_ptm", {}).get(str(len(chain_meta)), 0.0), 3),
        })

    ens_scores = sorted((ens[i].get("confidence_score", 0.0) for i in ens),
                        reverse=True)
    protein = chains[0]["n"] if chains else len(plddt)
    band_counts = [sum(1 for v in plddt[:protein] if lo <= v < hi or
                       (hi == 100 and v >= 90))
                   for lo, hi, _, _ in PLDDT_BANDS]
    return {
        "name": name, "best_model": best, "n_models": len(ens),
        "metrics": {k: round(float(bc[k]), 4) for k in (
            "confidence_score", "ptm", "iptm", "ligand_iptm", "complex_plddt",
            "complex_iplddt", "complex_pde", "complex_ipde") if k in bc},
        "chains": chain_meta, "pair_iptm": pair, "chain_keys": keys,
        "plddt": [round(float(x), 1) for x in plddt.tolist()],
        "protein_len": protein,
        "plddt_bands": [{"label": lab, "color": col, "count": n}
                        for (lo, hi, lab, col), n in zip(PLDDT_BANDS, band_counts)],
        "pae": [[round(float(x), 1) for x in row] for row in pae_ds.tolist()],
        "pae_factor": factor, "pae_max": round(float(pae.max()), 1),
        "chain_bounds": [c["start"] for c in chains] + [len(plddt)],
        "affinity": {k: round(float(v), 4) for k, v in aff.items()
                     if isinstance(v, (int, float))},
        "ensemble": [round(float(x), 4) for x in ens_scores],
        "pdb": pdb_text,
    }


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _band(v: float, good: float, ok: float) -> str:
    return "#1D9E75" if v >= good else ("#BA7517" if v >= ok else "#C0392B")


def build_complex_report_html(*, target_id: str, stats: dict,
                              conditions: List[Tuple[str, str]],
                              generated: str,
                              ligand_names: Optional[List[str]] = None) -> str:
    import html as _html
    import json as _json
    g = _html.escape
    s = stats
    m = s["metrics"]
    # Ligand chains come from the structure (generic for ANY target); optional
    # human names are supplied by the caller in chain order (B, C, …). The Boltz
    # affinity binder is the first ligand. Nothing here is target-specific.
    lig_chains = [c for c in s["chains"] if c["kind"] == "ligand"]

    def _ln(i):
        return ligand_names[i] if ligand_names and i < len(ligand_names) else None
    binder_lbl = _ln(0) or ("chain " + (lig_chains[0]["id"] if lig_chains else "B"))
    _parts = []
    for i, c in enumerate(lig_chains):
        nm = _ln(i)
        role = "affinity binder" if i == 0 else "co-modelled"
        _parts.append("chain " + c["id"] + (f" ({nm})" if nm else "")
                      + (", " + role if len(lig_chains) > 1 else ""))
    lig_descr = "; ".join(_parts) if _parts else "the modelled ligand(s)"
    aff = s.get("affinity", {})
    av = aff.get("affinity_pred_value")
    # Boltz-2 affinity head: affinity_pred_value = log10(IC50 / µM)
    ic50 = round(10 ** av, 2) if av is not None else None          # µM
    dg = round((av - 6) * 1.364, 1) if av is not None else None    # kcal/mol
    pbind = aff.get("affinity_probability_binary")

    def card(lab, val, sub, color="#111"):
        return (f'<div class="card"><div class="lab">{g(lab)}</div>'
                f'<div class="num" style="color:{color}">{g(val)}</div>'
                f'<div class="sub2">{g(sub)}</div></div>')

    cards = [
        card("confidence", f"{m['confidence_score']:.3f}", "0.8·pLDDT+0.2·ipTM",
             _band(m["confidence_score"], 0.8, 0.6)),
        card("ipTM", f"{m['iptm']:.3f}", "interface (>0.8 good)",
             _band(m["iptm"], 0.8, 0.6)),
        card("ligand ipTM", f"{m['ligand_iptm']:.3f}", "protein-ligand iface",
             _band(m["ligand_iptm"], 0.8, 0.6)),
        card("pTM", f"{m['ptm']:.3f}", "global fold", _band(m["ptm"], 0.8, 0.5)),
        card("complex pLDDT", f"{m['complex_plddt'] * 100:.1f}", "mean local conf.",
             _band(m["complex_plddt"], 0.9, 0.7)),
    ]
    if pbind is not None:
        cards.append(card("P(binder)", f"{pbind:.2f}", f"{binder_lbl}, hit-vs-decoy",
                          _band(pbind, 0.5, 0.3)))
    if ic50 is not None:
        cards.append(card("pred. IC50", f"{ic50:g} µM", f"ΔG≈{dg} kcal/mol"))
    cards.append(card("ensemble", f"{s['n_models']}", "diffusion samples"))
    cards_html = "".join(cards)

    band_legend = "".join(
        f'<span class="lg"><span class="sw" style="background:{b["color"]}"></span>'
        f'{g(b["label"])} ({b["count"]})</span>' for b in s["plddt_bands"])
    chain_rows = "".join(
        f'<tr><td>{g(c["id"])}</td><td>{g(c["kind"])}</td><td>{c["n"]}</td>'
        f'<td>{c["plddt_mean"]:.1f}</td><td>{c["ptm"]:.3f}</td></tr>'
        for c in s["chains"])
    cond_rows = "".join(f"<tr><td class=ck>{g(k)}</td><td>{g(str(v))}</td></tr>"
                        for k, v in conditions)
    blob = _json.dumps({k: s[k] for k in (
        "plddt", "pae", "pae_max", "chain_bounds", "pair_iptm", "chain_keys",
        "chains", "ensemble", "protein_len", "plddt_bands", "metrics")},
        separators=(",", ":"))

    return (_COMPLEX_TEMPLATE
            .replace("%%TITLE%%", g(f"{target_id} — complex prediction (Boltz-2)"))
            .replace("%%TARGET%%", g(target_id))
            .replace("%%GENERATED%%", g(generated))
            .replace("%%BESTMODEL%%", str(s["best_model"]))
            .replace("%%CARDS%%", cards_html)
            .replace("%%BANDLEGEND%%", band_legend)
            .replace("%%CHAINROWS%%", chain_rows)
            .replace("%%AFFVAL%%", "—" if av is None else f"{av:.3f}")
            .replace("%%AFFIC50%%", "—" if ic50 is None else f"{ic50:g}")
            .replace("%%AFFDG%%", "—" if dg is None else f"{dg}")
            .replace("%%AFFPBIND%%", "—" if pbind is None else f"{pbind:.2f}")
            .replace("%%CONDROWS%%", cond_rows)
            .replace("%%BINDER%%", g(binder_lbl))
            .replace("%%LIGDESCR%%", g(lig_descr))
            .replace("%%NMODELS%%", str(s["n_models"]))
            .replace("%%TARGETID%%", g(target_id))
            .replace("%%BLOB%%", blob)
            .replace("%%PDB%%", s["pdb"]))


def write_complex_report(path, **kwargs) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_complex_report_html(**kwargs), encoding="utf-8")


_COMPLEX_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
:root{--bg:#fff;--fg:#1c1c1a;--mut:#5f5e5a;--line:#e7e5df;--surf:#f7f6f2}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a18;--fg:#ece9e3;--mut:#a8a69e;--line:#33322e;--surf:#232220}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:2rem 1.25rem 3rem}
h1{font-size:22px;font-weight:500;margin:0 0 2px}
h2{font-size:18px;font-weight:500;margin:2.2rem 0 .5rem}
.sub{color:var(--mut);font-size:14px;margin:0 0 1.5rem}
.note{font-size:13px;color:var(--mut);margin:.3rem 0 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:0 0 1rem}
.card{background:var(--surf);border-radius:8px;padding:.7rem .85rem}
.lab{font-size:12px;color:var(--mut)}.num{font-size:22px;font-weight:500;margin-top:1px}.sub2{font-size:11px;color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:.5px solid var(--line)}
th{color:var(--mut);font-weight:500;font-size:13px}
td.ck{color:var(--mut);width:42%}
.legend{display:flex;flex-wrap:wrap;gap:13px;font-size:13px;color:var(--mut);margin:.2rem 0 .6rem}
.lg{display:flex;align-items:center;gap:5px}.sw{width:11px;height:11px;border-radius:2px;display:inline-block}
.chartbox{position:relative;width:100%;height:230px}
#viewer{width:100%;height:460px;position:relative;border:.5px solid var(--line);border-radius:6px;background:var(--surf)}
.vbtns{display:flex;gap:8px;margin:.5rem 0}
.btn{font:inherit;font-size:13px;color:var(--fg);background:var(--surf);border:.5px solid var(--line);border-radius:6px;padding:4px 12px;cursor:pointer}
.btn:hover{background:var(--line)}
.vctrl{display:flex;flex-wrap:wrap;gap:6px 16px;margin:.55rem 0 .35rem}
.cg{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.cgl{color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;margin-right:1px}
.vb{font:inherit;font-size:12px;color:var(--fg);background:var(--surf);border:.5px solid var(--line);border-radius:5px;padding:3px 8px;cursor:pointer;line-height:1.2}
.vb:hover{background:var(--line)}
.vb.on{background:#378ADD;color:#fff;border-color:#378ADD}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;align-items:start}
@media(max-width:680px){.row2{grid-template-columns:1fr}}
#paebox{position:relative}
#pae{width:100%;max-width:420px;image-rendering:pixelated;border:.5px solid var(--line);border-radius:4px;display:block}
.gauge{height:14px;border-radius:7px;background:linear-gradient(90deg,#C0392B,#FFDB13,#1D9E75);position:relative;margin:.4rem 0}
.gauge .pin{position:absolute;top:-4px;width:3px;height:22px;background:var(--fg);border-radius:2px}
.gauge .mid{position:absolute;left:50%;top:0;width:1px;height:14px;background:rgba(128,128,128,.6)}
.warn{font-size:12.5px;color:var(--mut);background:var(--surf);border-left:3px solid #BA7517;padding:.5rem .7rem;border-radius:4px;margin:.5rem 0}
footer{margin-top:2.5rem;border-top:.5px solid var(--line);padding-top:1rem}
footer p{font-size:13.5px}.cite{font-size:12px;color:var(--mut);line-height:1.7}
</style></head>
<body><div class="wrap">
<h1>%%TITLE%%</h1>
<p class="sub">target %%TARGET%% · Boltz-2 co-folded complex · best = model_%%BESTMODEL%% of the diffusion ensemble · generated %%GENERATED%%</p>

<div class="cards">%%CARDS%%</div>

<h2>3D structure — predicted complex</h2>
<div class="legend">pLDDT: %%BANDLEGEND%%</div>
<div class="vctrl">
 <div class="cg"><span class="cgl">color</span>
  <button class="vb on" onclick="setColor('plddt',this)">pLDDT</button>
  <button class="vb" onclick="setColor('chain',this)">chain</button>
  <button class="vb" onclick="setColor('spectrum',this)">rainbow</button>
  <button class="vb" onclick="setColor('hydro',this)">hydrophobicity</button>
  <button class="vb" onclick="setColor('ss',this)">2° structure</button></div>
 <div class="cg"><span class="cgl">protein</span>
  <button class="vb on" onclick="setRep('cartoon',this)">cartoon</button>
  <button class="vb" onclick="setRep('stick',this)">sticks</button>
  <button class="vb" onclick="setRep('line',this)">lines</button>
  <button class="vb" onclick="setRep('trace',this)">trace</button>
  <button class="vb" onclick="setRep('putty',this)">putty</button>
  <button class="vb" onclick="toggleSurf(this)">surface</button></div>
 <div class="cg"><span class="cgl">ligands</span>
  <button class="vb on" onclick="setLig('stick',this)">sticks</button>
  <button class="vb" onclick="setLig('ballstick',this)">ball+stick</button>
  <button class="vb" onclick="setLig('sphere',this)">spheres</button>
  <button class="vb" onclick="toggleLig(this)">hide</button></div>
 <div class="cg"><span class="cgl">pocket</span>
  <button class="vb" onclick="toggleIsolate(this)">isolate</button>
  <button class="vb" onclick="togglePsurf(this)">surface</button>
  <button class="vb" onclick="togglePocket(this)">residues</button>
  <button class="vb" onclick="toggleContacts(this)">H-bonds</button>
  <button class="vb" onclick="toggleLabels(this)">labels</button></div>
 <div class="cg"><span class="cgl">view</span>
  <button class="vb" onclick="viewWhole()">reset</button>
  <button class="vb" onclick="viewPocket()">pocket</button>
  <button class="vb" onclick="spin(this)">spin</button>
  <button class="vb" onclick="toggleBg(this)">dark</button>
  <button class="vb" onclick="fullscreen()">⛶ full</button>
  <button class="vb" onclick="snap()">⤓ PNG</button></div>
</div>
<div id="viewer"><div style="padding:1rem;color:var(--mut);font-size:13px">loading 3D viewer…</div></div>
<p class="note">Protein cartoon coloured by per-residue pLDDT (AlphaFold scheme: blue
= very high → orange = very low); ligands as sticks / spheres (carbons
highlighted). Drag to rotate, scroll to zoom. 3Dmol.js (Rego &amp; Koes 2015).</p>

<div class="row2">
<div>
<h2>Per-residue pLDDT</h2>
<div class="chartbox" style="height:220px"><canvas id="plddtChart"></canvas></div>
<p class="note">Local confidence (pLDDT 0–100) per token: protein residues first,
then the ligand atoms. Line coloured by band (&gt;90 very high,
70–90 confident, 50–70 low, &lt;50 very low).</p>
</div>
<div>
<h2>PAE — predicted aligned error</h2>
<div id="paebox"><canvas id="pae"></canvas></div>
<p class="note">PAE(i,j) in Å (dark green = confident, light = ≥30 Å). White
lines split the chains: dark protein↔ligand off-diagonal blocks mean the
ligand(s) are confidently placed relative to the protein. After
AlphaFold2 (Jumper 2021).</p>
</div>
</div>

<h2>Inter-chain interface (ipTM) &amp; binding affinity</h2>
<div class="row2">
<div>
<table><thead><tr><th>chain</th><th>type</th><th>tokens</th><th>pLDDT</th><th>pTM</th></tr></thead>
<tbody>%%CHAINROWS%%</tbody></table>
<div class="chartbox" style="height:150px;margin-top:.6rem"><canvas id="iptmChart"></canvas></div>
<p class="note">Pairwise interface ipTM (high = confident interface).</p>
</div>
<div>
<p style="margin:.2rem 0"><b>%%BINDER%% binding (Boltz-2 affinity head)</b></p>
<p class="note" style="margin:.1rem 0">P(binder) — hit vs decoy</p>
<div class="gauge"><div class="mid"></div><div class="pin" id="pbindpin"></div></div>
<p style="font-size:13px;margin:.1rem 0">P(binder) = <b>%%AFFPBIND%%</b> · pred. IC50 ≈ <b>%%AFFIC50%% µM</b>
 · ΔG ≈ <b>%%AFFDG%% kcal/mol</b> <span class="note">(raw log10IC50 = %%AFFVAL%%)</span></p>
<div class="warn">Boltz-2 affinity is a <b>ranking/triage signal, not a measured K<sub>d</sub></b>
(~40% false-positive rate on predicted hits). IC50 = 10<sup>value</sup> µM,
ΔG = (value−6)·1.364 kcal/mol; lower value = stronger. Experimental validation
required.</div>
</div>
</div>

<h2>Diffusion ensemble — model_%%BESTMODEL%% selected</h2>
<div class="chartbox" style="height:180px"><canvas id="ensChart"></canvas></div>
<p class="note">confidence_score across all diffusion samples (sorted). A tight
high cluster = a robust prediction; a wide/bimodal spread = competing poses.</p>

<footer>
<h2>Methods</h2>
<p><b>Complex prediction.</b> The wild-type protein–ligand complex was predicted
with <b>Boltz-2</b> (Wohlwend et al. 2024), an AlphaFold3-class (Abramson et al.
2024) diffusion co-folding model, from the target sequence, the integrated
multi-track MSA, and the ligand(s) (%%LIGDESCR%%). <b>%%NMODELS%% diffusion
samples</b> were drawn; models are ranked
by confidence_score = 0.8·complex_pLDDT + 0.2·ipTM (Boltz convention) and the
top-ranked (model_0) is shown. Confidence is reported as per-residue pLDDT
(0–100; bands &gt;90 / 70–90 / 50–70 / &lt;50, AlphaFold scheme), the predicted
aligned error matrix (PAE, 0–30 Å), and pTM / ipTM / ligand-ipTM. pLDDT and PAE
arrays are length (protein residues + ligand atoms); indices past the protein
length are ligand atoms.</p>
<p><b>Binding affinity.</b> Boltz-2's affinity head outputs affinity_pred_value
(= log10 IC50 in µM, a non-standard pIC50) and affinity_probability_binary
(P that the ligand is a binder vs a decoy). We report IC50 = 10^value µM and
ΔG = (value−6)·1.364 kcal/mol, treating affinity as a ranking signal for triage,
not a measured constant. <b>Visualisation</b>: interactive 3D via 3Dmol.js (Rego
&amp; Koes 2015), protein cartoon coloured by pLDDT B-factor.</p>
<h2>Analysis conditions</h2>
<table><tbody>%%CONDROWS%%</tbody></table>
<h2>References</h2>
<p class="cite">
Jumper et al. (2021) <i>Nature</i> 596:583 — AlphaFold2 (pLDDT/PAE).
Abramson et al. (2024) <i>Nature</i> 630:493 — AlphaFold3.
Wohlwend et al. (2024) — Boltz-1/Boltz-2.
Mirdita et al. (2022) <i>Nat. Methods</i> 19:679 — ColabFold.
Rego &amp; Koes (2015) <i>Bioinformatics</i> 31:1322 — 3Dmol.js.
Sehnal et al. (2021) <i>NAR</i> 49:W431 — Mol*. Rose et al. (2018) — NGL.
</p>
</footer>

<script id="pdbdata" type="text/plain">%%PDB%%</script>
</div>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
var R=%%BLOB%%;
function cssv(n){return getComputedStyle(document.body).getPropertyValue(n).trim()||'#888';}
function af(b){return b>90?'#0053D6':b>70?'#65CBF3':b>50?'#FFDB13':'#FF7D45';}
var V=null,MDL=null,_surf=null,_psurf=null,_spin=false,_dark=false,_pocket=false,_isolate=false,_labels=false,_lig=true,_contacts=[],_hbl=[];
var curColor='plddt',curRep='cartoon',ligStyle='stick',bScale=1,resiMax=384;
var CHAINCOL={A:'#6aa7df',B:'#1D9E75',C:'#BA7517',D:'#9b59b6',E:'#C0392B'};
var HYDRO={ALA:1.8,ARG:-4.5,ASN:-3.5,ASP:-3.5,CYS:2.5,GLN:-3.5,GLU:-3.5,GLY:-0.4,HIS:-3.2,ILE:4.5,LEU:3.8,LYS:-3.9,MET:1.9,PHE:2.8,PRO:-1.6,SER:-0.8,THR:-0.7,TRP:-0.9,TYR:-1.3,VAL:4.2};
function hydroCol(aa){var h=HYDRO[aa];if(h===undefined)return '#888';var t=Math.max(0,Math.min(1,(h+4.5)/9));return 'rgb('+Math.round(60+195*t)+',135,'+Math.round(60+195*(1-t))+')';}
function ssCol(a){return a.ss==='h'?'#E0501E':a.ss==='s'?'#F1C40F':'#cfcdc6';}
function specCol(a){var t=Math.max(0,Math.min(1,((a.resi||1)-1)/Math.max(1,resiMax-1)));return 'hsl('+Math.round(240-240*t)+',70%,50%)';}
var POCKET={within:{distance:5,sel:{hetflag:true}},byres:true,hetflag:false};
function colorFn(){
  if(curColor==='hydro')return function(a){return hydroCol(a.resn);};
  if(curColor==='chain')return function(a){return CHAINCOL[a.chain]||'#888';};
  if(curColor==='spectrum')return specCol;
  if(curColor==='ss')return ssCol;
  return function(a){return af(a.b*bScale);};
}
function ligSpec(){
  if(ligStyle==='sphere')return {sphere:{scale:0.3}};
  if(ligStyle==='ballstick')return {stick:{radius:0.12,colorscheme:'cyanCarbon'},sphere:{scale:0.25}};
  return {stick:{radius:0.2,colorscheme:'cyanCarbon'}};
}
function initViewer(){
  if(!window.$3Dmol){document.getElementById('viewer').innerHTML='<p style="padding:1rem;color:#C0392B;font-size:13px">3Dmol.js failed to load (offline?). The confidence plots below still work.</p>';return;}
  V=$3Dmol.createViewer('viewer',{backgroundColor:'white'});
  MDL=V.addModel(document.getElementById('pdbdata').textContent,'pdb');
  var bmax=0,rmax=0;MDL.selectedAtoms({hetflag:false}).forEach(function(a){if(a.b>bmax)bmax=a.b;if(a.resi>rmax)rmax=a.resi;});
  bScale=bmax<=1.5?100:1;resiMax=rmax||384;
  V.setHoverable({},true,
    function(a){if(!a._lab){a._lab=V.addLabel((a.resn||'')+(a.resi||'')+' · '+a.atom+(a.b?' · pLDDT '+(a.b*bScale).toFixed(0):''),{position:a,backgroundColor:'black',backgroundOpacity:0.75,fontSize:11});V.render();}},
    function(a){if(a._lab){V.removeLabel(a._lab);a._lab=null;V.render();}});
  applyStyle();V.zoomTo();V.render();
}
function applyStyle(){
  if(!V)return;_isolate=false;
  V.setStyle({},{});
  MDL.setColorByFunction({hetflag:false},colorFn());
  if(curRep==='putty'){
    [[0,55,0.15],[55,70,0.35],[70,85,0.65],[85,201,1.0]].forEach(function(bn){
      V.setStyle({hetflag:false,b:{gte:bn[0]/bScale,lt:bn[1]/bScale}},{cartoon:{style:'trace',thickness:bn[2]}});});
  }else{
    var ps={};
    if(curRep==='cartoon')ps.cartoon={};
    else if(curRep==='stick')ps.stick={radius:0.15};
    else if(curRep==='line')ps.line={};
    else if(curRep==='trace')ps.cartoon={style:'trace',thickness:0.35};
    V.setStyle({hetflag:false},ps);
  }
  if(_lig)V.setStyle({hetflag:true},ligSpec());
  if(_pocket)V.addStyle(POCKET,{stick:{radius:0.13,colorscheme:'whiteCarbon'}});
  V.render();
}
function mark(b){if(b&&b.parentNode){b.parentNode.querySelectorAll('.vb').forEach(function(x){x.classList.remove('on');});b.classList.add('on');}}
function setColor(c,b){curColor=c;mark(b);if(_isolate)isolate(true);else applyStyle();}
function setRep(r,b){curRep=r;mark(b);applyStyle();}
function setLig(s,b){ligStyle=s;_lig=true;mark(b);if(_isolate)isolate(true);else applyStyle();}
function toggleLig(b){_lig=!_lig;if(b)b.classList.toggle('on');applyStyle();}
function toggleSurf(b){if(!V)return;
  if(_surf!=null){if(_surf!=='p')V.removeSurface(_surf);_surf=null;if(b)b.classList.remove('on');V.render();}
  else{if(b)b.classList.add('on');_surf='p';Promise.resolve(V.addSurface($3Dmol.SurfaceType.VDW,{opacity:0.6,colorscheme:'whiteCarbon'},{hetflag:false})).then(function(r){var id=(r&&r.surfid!==undefined)?r.surfid:r;if(_surf==='p')_surf=id;else if(id!=null)V.removeSurface(id);V.render();});}}
function togglePsurf(b){if(!V)return;
  if(_psurf!=null){if(_psurf!=='p')V.removeSurface(_psurf);_psurf=null;if(b)b.classList.remove('on');V.render();}
  else{if(b)b.classList.add('on');_psurf='p';Promise.resolve(V.addSurface($3Dmol.SurfaceType.SES,{opacity:0.65,colorscheme:'whiteCarbon'},POCKET,{})).then(function(r){var id=(r&&r.surfid!==undefined)?r.surfid:r;if(_psurf==='p')_psurf=id;else if(id!=null)V.removeSurface(id);V.render();});}}
function isolate(keep){
  V.setStyle({},{});
  V.setStyle({hetflag:false},{cartoon:{color:'#d8d6cf',opacity:0.22}});
  MDL.setColorByFunction(POCKET,colorFn());
  V.addStyle(POCKET,{stick:{radius:0.17}});
  V.setStyle({hetflag:true},{stick:{radius:0.22,colorscheme:'cyanCarbon'}});
  if(!keep){V.zoomTo({hetflag:true});V.zoom(1.4);}
  V.render();
}
function toggleIsolate(b){if(!V)return;_isolate=!_isolate;if(b)b.classList.toggle('on');
  if(_isolate)isolate(false);else{applyStyle();V.zoomTo();V.render();}}
function togglePocket(b){_pocket=!_pocket;if(b)b.classList.toggle('on');applyStyle();if(_pocket)viewPocket();}
function toggleLabels(b){if(!V)return;_labels=!_labels;if(b)b.classList.toggle('on');
  if(_labels)MDL.selectedAtoms({and:[{hetflag:false},{atom:'CA'},{within:{distance:5,sel:{hetflag:true}}}]}).forEach(function(a){a._rl=V.addLabel(a.resn+a.resi,{position:a,fontSize:10,backgroundColor:'black',backgroundOpacity:0.6});});
  else MDL.selectedAtoms({}).forEach(function(a){if(a._rl){V.removeLabel(a._rl);a._rl=null;}});
  V.render();}
function toggleContacts(b){if(!V)return;
  if(_contacts.length||_hbl.length){_contacts.forEach(function(s){V.removeShape(s);});_hbl.forEach(function(l){V.removeLabel(l);});_contacts=[];_hbl=[];if(b)b.classList.remove('on');V.render();return;}
  var lig=MDL.selectedAtoms({hetflag:true}),pro=MDL.selectedAtoms({hetflag:false});
  lig.forEach(function(l){if(l.elem!=='N'&&l.elem!=='O')return;
    pro.forEach(function(p){if(p.elem!=='N'&&p.elem!=='O')return;
      var dx=p.x-l.x,dy=p.y-l.y,dz=p.z-l.z,d2=dx*dx+dy*dy+dz*dz;if(d2<12.96){
        _contacts.push(V.addCylinder({start:{x:p.x,y:p.y,z:p.z},end:{x:l.x,y:l.y,z:l.z},radius:0.04,dashed:true,color:'#f1c40f'}));
        _hbl.push(V.addLabel(Math.sqrt(d2).toFixed(1),{position:{x:(p.x+l.x)/2,y:(p.y+l.y)/2,z:(p.z+l.z)/2},fontSize:9,backgroundColor:'black',backgroundOpacity:0.45,fontColor:'#f1c40f'}));}});});
  if(b)b.classList.add('on');V.render();}
function viewWhole(){if(V){_isolate=false;applyStyle();V.zoomTo();V.render();}}
function viewPocket(){if(V){V.zoomTo({hetflag:true});V.zoom(1.7);V.render();}}
function spin(b){if(!V)return;_spin=!_spin;V.spin(_spin?'y':false);if(b)b.classList.toggle('on');}
function toggleBg(b){if(!V)return;_dark=!_dark;V.setBackgroundColor(_dark?'#15140f':'white');if(b)b.classList.toggle('on');V.render();}
function fullscreen(){var e=document.getElementById('viewer');if(e.requestFullscreen)e.requestFullscreen();else if(e.webkitRequestFullscreen)e.webkitRequestFullscreen();}
function snap(){if(V){var a=document.createElement('a');a.href=V.pngURI();a.download='%%TARGETID%%_complex.png';a.click();}}

var MUT=cssv('--mut'),GRID=cssv('--line');
function drawPae(){
  var c=document.getElementById('pae');if(!c)return;var x=c.getContext('2d');
  var P=R.pae,n=P.length,mx=30;c.width=n;c.height=n;
  for(var i=0;i<n;i++)for(var j=0;j<n;j++){var t=Math.min(1,P[i][j]/mx);
    // dark green (#00441b) -> white
    var r=Math.round(0+255*t),g=Math.round(68+187*t),b=Math.round(27+228*t);
    x.fillStyle='rgb('+r+','+g+','+b+')';x.fillRect(j,i,1,1);}
  var f=R.chain_bounds[R.chain_bounds.length-1]/n;
  x.strokeStyle='#fff';x.lineWidth=Math.max(0.5,n/300);
  for(var k=1;k<R.chain_bounds.length-1;k++){var p=R.chain_bounds[k]/f;
    x.beginPath();x.moveTo(p,0);x.lineTo(p,n);x.moveTo(0,p);x.lineTo(n,p);x.stroke();}
}
function mkPlddt(){
  var pl=R.plddt,lab=pl.map(function(_,i){return i+1;}),pcut=R.protein_len;
  new Chart(document.getElementById('plddtChart'),{type:'line',
    data:{labels:lab,datasets:[{data:pl,pointRadius:0,borderWidth:1.4,
      segment:{borderColor:function(ctx){return af(ctx.p1.parsed.y);}},borderColor:'#65CBF3'}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
     scales:{y:{min:0,max:100,title:{display:true,text:'pLDDT',color:MUT},ticks:{color:MUT},grid:{color:GRID}},
      x:{title:{display:true,text:'token (residues 1-'+pcut+', then ligand atoms)',color:MUT},
       ticks:{color:MUT,maxTicksLimit:10},grid:{display:false}}}}});
}
function mkIptm(){
  var ks=R.chain_keys,lbl=R.chains.map(function(c){return c.id+' ('+c.kind[0]+')';});
  var data=[];for(var i=0;i<ks.length;i++)for(var j=0;j<ks.length;j++){
    data.push({x:lbl[j],y:lbl[i],v:R.pair_iptm[ks[i]+','+ks[j]]||0});}
  // simple grouped bars: protein-vs-each
  var dss=ks.map(function(k,i){return {label:lbl[i],data:ks.map(function(k2,j){return R.pair_iptm[k+','+k2]||0;}),
    backgroundColor:['#7F77DD','#1D9E75','#BA7517'][i%3]};});
  new Chart(document.getElementById('iptmChart'),{type:'bar',
    data:{labels:lbl,datasets:dss},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:true,labels:{color:MUT,boxWidth:10,font:{size:10}}}},
     scales:{y:{min:0,max:1,title:{display:true,text:'ipTM',color:MUT},ticks:{color:MUT},grid:{color:GRID}},
      x:{ticks:{color:MUT},grid:{display:false}}}}});
}
function mkEns(){
  var e=R.ensemble;
  new Chart(document.getElementById('ensChart'),{type:'bar',
    data:{labels:e.map(function(_,i){return i+1;}),datasets:[{data:e,backgroundColor:'#378ADD'}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
     scales:{y:{title:{display:true,text:'confidence_score',color:MUT},ticks:{color:MUT},grid:{color:GRID},
       min:Math.max(0,Math.min.apply(null,e)-0.05)},
      x:{title:{display:true,text:'diffusion sample (ranked)',color:MUT},ticks:{color:MUT,maxTicksLimit:10},grid:{display:false}}}}});
}
function pinGauge(){var p=document.getElementById('pbindpin');if(p)p.style.left=(%%AFFPBIND%%*100)+'%';}
function start(){drawPae();pinGauge();if(window.Chart){mkPlddt();mkIptm();mkEns();}
  setTimeout(initViewer,150);}
if(document.readyState!=='loading')start();else document.addEventListener('DOMContentLoaded',start);
</script></body></html>
"""
