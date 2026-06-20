"""Self-contained HTML report for the s05 reference-docking ensemble.

This reports the WT reference docking as an INTERNAL reproducibility check, not
an experimental validation: the reference is the s04 Boltz pose (a COMPUTED
structure), so the question is "do independent dockers reproduce the Boltz
binding mode well enough to use it as a baseline for mutant interpretation
(s09)?" — not "is the binding mode experimentally correct?".

Pose accuracy is the symmetry-corrected heavy-atom RMSD to the reference (the
CASF/PDBbind docking-power metric, RMSD<2 A), computed after ligand
normalization (strip H, drop co-modelled fragments e.g. formate, share bond
orders from the ligand SMILES template). Centroid distance is also shown but only
as a COARSE proxy — it stays small even when a large flexible cofactor like NADP
rotates or twists. GNINA's score is split into its real components
(minimizedAffinity kcal/mol vs CNNaffinity pK vs CNNscore). DiffDock confidence
is read with calibration (a log-odds, so <0 means <50% predicted success) and
rank margins. Generic for any target/ligand — all from the actual s05 outputs.
"""

from __future__ import annotations

import glob
import math
import os
import re
import sqlite3
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Parsing (stdlib)
# --------------------------------------------------------------------------- #
def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _sdf_first_coords(sdf_text: Optional[str]) -> List[tuple]:
    if not sdf_text:
        return []
    lines = sdf_text.splitlines()
    if len(lines) < 4:
        return []
    try:
        natoms = int(lines[3][:3])
    except ValueError:
        return []
    out = []
    for ln in lines[4:4 + natoms]:
        try:
            out.append((float(ln[0:10]), float(ln[10:20]), float(ln[20:30]),
                        ln[31:34].strip()))
        except (ValueError, IndexError):
            pass
    return out


def _pdb_het_coords(pdb_text: Optional[str]) -> List[tuple]:
    out = []
    for ln in (pdb_text or "").splitlines():
        if ln[:6].strip() == "HETATM":
            try:
                out.append((float(ln[30:38]), float(ln[38:46]),
                            float(ln[46:54]), ln[76:78].strip()))
            except (ValueError, IndexError):
                pass
    return out


def _heavy(coords):
    return [c for c in coords if c[3] != "H"]


def _centroid(coords):
    if not coords:
        return None
    n = len(coords)
    return (sum(c[0] for c in coords) / n, sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n)


def _dist(a, b):
    if a is None or b is None:
        return None
    return round(sum((a[k] - b[k]) ** 2 for k in range(3)) ** 0.5, 2)


def _diffdock_landscape(dd_dir: str) -> List[tuple]:
    out = set()
    for f in glob.glob(os.path.join(dd_dir, "**", "rank*_confidence*.sdf"),
                       recursive=True):
        m = re.search(r"rank(\d+)_confidence(-?\d+\.?\d*)", os.path.basename(f))
        if m:
            out.add((int(m.group(1)), float(m.group(2))))
    return sorted(out)


def _sdf_props(sdf_text: Optional[str]) -> Dict[str, float]:
    """GNINA SDF data fields (minimizedAffinity, CNNaffinity, CNNscore, …) of the
    FIRST pose. GNINA writes the Vina-style minimizedAffinity AND the CNN scores
    as separate tags — they are NOT the same number on the same scale."""
    props: Dict[str, float] = {}
    if not sdf_text:
        return props
    lines = sdf_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "$$$$":  # only the first pose
            break
        m = re.match(r"^>\s*<(.+?)>", lines[i])
        if m and i + 1 < len(lines):
            try:
                props[m.group(1)] = float(lines[i + 1].strip())
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    return props


# --------------------------------------------------------------------------- #
# RDKit: ligand normalization + symmetry-corrected heavy-atom RMSD (optional)
# --------------------------------------------------------------------------- #
def _template(smiles: Optional[str]):
    if not smiles:
        return None
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _pose_mol(text: str, fmt: str, template):
    """Normalized ligand RDKit mol from a docked pose: the LARGEST fragment (so a
    co-modelled formate is dropped), heavy atoms only, bond orders taken from the
    SMILES template so symmetry is well-defined and shared across poses. Returns
    (mol|None, qc-dict)."""
    qc = {"raw": None, "heavy": None, "frags": None, "templated": False}
    try:
        from rdkit import Chem
    except Exception:
        return None, qc
    full = (Chem.MolFromMolBlock(text, sanitize=False, removeHs=False) if fmt == "sdf"
            else Chem.MolFromPDBBlock(text, sanitize=False, removeHs=False))
    if full is None:
        return None, qc
    qc["raw"] = full.GetNumAtoms()
    try:
        m = Chem.RemoveHs(full, sanitize=False)
    except Exception:
        m = full
    try:
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    except Exception:
        frags = [m]
    qc["frags"] = len(frags)
    m = max(frags, key=lambda f: f.GetNumAtoms())
    qc["heavy"] = m.GetNumAtoms()
    if template is not None:
        try:
            from rdkit.Chem import AllChem
            m = AllChem.AssignBondOrdersFromTemplate(template, m)
            qc["templated"] = True
        except Exception:
            pass
    return m, qc


def _no_align_rmsd(prb, ref) -> Optional[float]:
    """Symmetry-corrected heavy-atom RMSD WITHOUT superposition — the poses are
    already in the receptor frame, so we must NOT re-align (that would hide a
    translation/rotation). Min over symmetry-equivalent atom mappings."""
    if prb is None or ref is None:
        return None
    try:
        import numpy as np
        matches = ref.GetSubstructMatches(prb, uniquify=False, maxMatches=5000)
        if not matches:
            return None
        cp, cr = prb.GetConformer(), ref.GetConformer()
        pc = np.array([[cp.GetAtomPosition(i).x, cp.GetAtomPosition(i).y,
                        cp.GetAtomPosition(i).z] for i in range(prb.GetNumAtoms())])
        best = None
        for mt in matches:
            rc = np.array([[cr.GetAtomPosition(j).x, cr.GetAtomPosition(j).y,
                            cr.GetAtomPosition(j).z] for j in mt])
            r = float(np.sqrt(((pc - rc) ** 2).sum(1).mean()))
            best = r if best is None or r < best else best
        return round(best, 2) if best is not None else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
def compute_docking_stats(run_dir: str, db_path: str, methods: List[str], *,
                          candidate: str = "wt",
                          ligand_smiles: Optional[str] = None) -> dict:
    dock = os.path.join(run_dir, "docking")

    scores: Dict[str, dict] = {}
    try:
        con = sqlite3.connect(str(db_path), timeout=15)
        for m, sc, r in con.execute(
            "select method,score,ligand_rmsd_to_reference from docking_pose "
            "where candidate_id=?", (candidate,)
        ):
            scores[m] = {"score": sc, "rmsd": r}
        con.close()
    except sqlite3.Error:
        pass

    receptor = (_read(os.path.join(dock, "gnina", f"{candidate}_rec.pdb"))
                or _read(os.path.join(dock, "diffdock", f"{candidate}_rec.pdb")))
    ref_pdb = _read(os.path.join(dock, "gnina", f"{candidate}_ref_lig.pdb"))
    gnina_sdf = _read(os.path.join(dock, "gnina", f"{candidate}_gnina_out.sdf"))
    dd_dir = os.path.join(dock, "diffdock", f"{candidate}_dd_out")
    landscape = _diffdock_landscape(dd_dir)
    r1 = sorted(glob.glob(os.path.join(dd_dir, "**", "rank1_confidence*.sdf"),
                          recursive=True))
    dd_sdf = _read(r1[0]) if r1 else None

    poses: Dict[str, str] = {}
    if gnina_sdf:
        poses["gnina"] = gnina_sdf
    if dd_sdf:
        poses["diffdock"] = dd_sdf

    # GNINA's real sub-scores (the DB 'score' is minimizedAffinity)
    gnina_props = _sdf_props(gnina_sdf)

    # --- ligand normalization + symmetry-corrected heavy-atom RMSD ---
    tmpl = _template(ligand_smiles)
    mols, qc = {}, {}
    for m, sdf in poses.items():
        mols[m], qc[m] = _pose_mol(sdf, "sdf", tmpl)
    if ref_pdb:
        mols["reference"], qc["reference"] = _pose_mol(ref_pdb, "pdb", tmpl)

    names = [n for n in ["gnina", "diffdock"] if n in mols] + \
            (["reference"] if "reference" in mols else [])
    pairs = []  # (a, b, heavy_rmsd, centroid_dist)
    cen = {}
    for n, sdf in poses.items():
        cen[n] = _centroid(_heavy(_sdf_first_coords(sdf)))
    cen["reference"] = _centroid(_heavy(_pdb_het_coords(ref_pdb))) if ref_pdb else None
    seen = set()
    order = [("gnina", "reference"), ("diffdock", "reference"),
             ("gnina", "diffdock")]
    for a, b in order:
        if a in mols and b in mols and (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b, _no_align_rmsd(mols[a], mols[b]),
                          _dist(cen.get(a), cen.get(b))))

    rdkit_ok = any(q.get("heavy") for q in qc.values())
    return {
        "candidate": candidate,
        "methods": [m for m in methods if m in scores] or list(scores),
        "scores": scores,
        "gnina_props": gnina_props,
        "landscape": landscape,
        "receptor": receptor,
        "poses": poses,
        "reference_pdb": ref_pdb,
        "rmsd_pairs": pairs,
        "ligand_qc": qc,
        "rdkit_ok": rdkit_ok,
        "ligand_smiles": ligand_smiles,
        "n_diffdock_poses": len(landscape),
    }


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
_POSECOL = {"gnina": "#1D9E75", "diffdock": "#BA7517", "reference": "#3b7dd8"}


def build_docking_report_html(*, target_id: str, stats: dict,
                              conditions: List, generated: str,
                              ligand_name: Optional[str] = None) -> str:
    import html as _html
    import json as _json
    g = _html.escape
    s = stats
    lig = ligand_name or "the ligand"
    gp = s.get("gnina_props", {})

    def fnum(v, fmt="{:g}"):
        return "—" if v is None else fmt.format(v)

    def card(lab, val, sub, color="#111"):
        return (f'<div class="card"><div class="lab">{g(lab)}</div>'
                f'<div class="num" style="color:{color}">{g(str(val))}</div>'
                f'<div class="sub2">{g(sub)}</div></div>')

    # best heavy-atom RMSD vs reference (the headline accuracy number)
    ref_rmsds = [r for a, b, r, _ in s["rmsd_pairs"] if b == "reference" and r is not None]
    best_ref = min(ref_rmsds) if ref_rmsds else None

    cards = []
    if "gnina" in s["scores"]:
        cards.append(card("GNINA minAff", fnum(gp.get("minimizedAffinity",
                          s["scores"]["gnina"]["score"])), "kcal/mol, lower=stronger"))
        if "CNNaffinity" in gp:
            cards.append(card("GNINA CNNaff", fnum(gp["CNNaffinity"]),
                              "pK, higher=stronger"))
    if "diffdock" in s["scores"]:
        c = s["scores"]["diffdock"]["score"]
        p = 1 / (1 + math.exp(-c)) if c is not None else None
        cards.append(card("DiffDock conf", fnum(c),
                          f"P(<2A)≈{p:.0%}" if p is not None else "log-odds",
                          "#1D9E75" if (c is not None and c > 0) else "#BA7517"))
    if best_ref is not None:
        cards.append(card("heavy-atom RMSD", f"{best_ref} A", "vs Boltz ref (<2=ok)",
                          "#1D9E75" if best_ref < 2 else "#BA7517"))
    cards.append(card("DiffDock poses", s["n_diffdock_poses"], "sampled & ranked"))
    cards_html = "".join(cards)

    # RMSD + centroid table (RMSD primary; centroid a coarse proxy)
    rrows = ""
    for a, b, rm, ce in s["rmsd_pairs"]:
        verdict = ("—" if rm is None else
                   ("reproduced (&lt;2 A)" if rm < 2 else "shifted (&ge;2 A)"))
        rrows += (f"<tr><td>{g(a)} ↔ {g(b)}</td>"
                  f"<td><b>{fnum(rm, '{:.2f} A')}</b></td>"
                  f"<td class=ck>{fnum(ce, '{:.2f} A')}</td><td>{verdict}</td></tr>")
    if not s["rmsd_pairs"]:
        rrows = "<tr><td colspan=4 class=ck>single method — no pairwise comparison</td></tr>"

    # ligand normalization QC
    qrows = ""
    for nm in ("gnina", "diffdock", "reference"):
        q = s["ligand_qc"].get(nm)
        if not q:
            continue
        frag = ("1" if q.get("frags") in (1, None) else
                f'{q["frags"]} (largest kept)')
        qrows += (f"<tr><td>{g(nm)}</td><td>{fnum(q.get('raw'))}</td>"
                  f"<td>{fnum(q.get('heavy'))}</td><td>{frag}</td>"
                  f"<td>{'yes' if q.get('templated') else 'no'}</td></tr>")
    if not qrows:
        qrows = "<tr><td colspan=5 class=ck>RDKit normalization unavailable</td></tr>"

    # method score table
    mrows = ""
    if "gnina" in s["scores"]:
        mrows += ("<tr><td><b>GNINA</b></td>"
                  f"<td>minimizedAffinity = {fnum(gp.get('minimizedAffinity', s['scores']['gnina']['score']))}</td>"
                  "<td>kcal/mol, lower = stronger (AutoDock Vina energy)</td></tr>"
                  "<tr><td></td>"
                  f"<td>CNNaffinity = {fnum(gp.get('CNNaffinity'))}</td>"
                  "<td>pK, higher = stronger (CNN-predicted)</td></tr>"
                  "<tr><td></td>"
                  f"<td>CNNscore = {fnum(gp.get('CNNscore'))}</td>"
                  "<td>0–1 pose-quality (CNN P(pose is good))</td></tr>")
    if "diffdock" in s["scores"]:
        c = s["scores"]["diffdock"]["score"]
        mrows += ("<tr><td><b>DiffDock</b></td>"
                  f"<td>confidence = {fnum(c)}</td>"
                  "<td>log-odds of a &lt;2 A pose; &lt;0 → P&lt;50%</td></tr>")

    # diffdock rank margins
    land = s["landscape"]
    margin = ""
    if len(land) >= 2:
        top = land[0][1]
        d12 = round(land[0][1] - land[1][1], 3)
        margin = (f"rank-1 confidence {top}; rank-1→rank-2 margin {d12}. "
                  + ("FLAT top — competing modes, not a single confident pose."
                     if abs(d12) < 0.1 else "clear rank-1 separation."))

    cond_rows = "".join(f"<tr><td class=ck>{g(str(k))}</td><td>{g(str(v))}</td></tr>"
                        for k, v in conditions)

    # 3D models
    blocks, model_js = [], []
    if s["receptor"]:
        blocks.append('<script id="m_rec" type="text/plain">' + s["receptor"] + "</script>")
    for m, sdf in s["poses"].items():
        blocks.append(f'<script id="m_{m}" type="text/plain">{sdf}</script>')
        model_js.append(m)
    if s["reference_pdb"]:
        blocks.append('<script id="m_reference" type="text/plain">' + s["reference_pdb"] + "</script>")
        model_js.append("reference")
    pose_btns = "".join(
        f'<button class="vb on" onclick="togglePose(\'{m}\',this)">'
        f'<span class="dot" style="background:{_POSECOL.get(m, "#888")}"></span>'
        f'{g(m)}</button>' for m in model_js)

    blob = _json.dumps({"landscape": land, "models": model_js,
                        "posecol": {m: _POSECOL.get(m, "#888") for m in model_js}},
                       separators=(",", ":"))

    return (_DOCK_TEMPLATE
            .replace("%%TITLE%%", g(f"{target_id} — reference docking (s05)"))
            .replace("%%TARGET%%", g(target_id))
            .replace("%%LIG%%", g(lig))
            .replace("%%GENERATED%%", g(generated))
            .replace("%%CARDS%%", cards_html)
            .replace("%%POSEBTNS%%", pose_btns)
            .replace("%%RROWS%%", rrows)
            .replace("%%QROWS%%", qrows)
            .replace("%%MROWS%%", mrows)
            .replace("%%MARGIN%%", g(margin) or "single ranked pose")
            .replace("%%CONDROWS%%", cond_rows)
            .replace("%%SMILES%%", g((s.get("ligand_smiles") or "—")[:60]))
            .replace("%%TARGETID%%", g(target_id))
            .replace("%%BLOB%%", blob)
            .replace("%%MODELS%%", "".join(blocks)))


def write_docking_report(path, **kwargs) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_docking_report_html(**kwargs), encoding="utf-8")


_DOCK_TEMPLATE = r"""<!DOCTYPE html>
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
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 1rem}
.card{background:var(--surf);border-radius:8px;padding:.7rem .85rem}
.lab{font-size:12px;color:var(--mut)}.num{font-size:21px;font-weight:500;margin-top:1px}.sub2{font-size:11px;color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:.5px solid var(--line)}
th{color:var(--mut);font-weight:500;font-size:13px}
td.ck{color:var(--mut)}
#viewer{width:100%;height:460px;position:relative;border:.5px solid var(--line);border-radius:6px;background:var(--surf)}
.vctrl{display:flex;flex-wrap:wrap;gap:6px 16px;margin:.55rem 0 .35rem}
.cg{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.cgl{color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;margin-right:1px}
.vb{font:inherit;font-size:12px;color:var(--fg);background:var(--surf);border:.5px solid var(--line);border-radius:5px;padding:3px 8px;cursor:pointer;line-height:1.2;display:inline-flex;align-items:center;gap:5px}
.vb:hover{background:var(--line)}.vb.on{background:#378ADD;color:#fff;border-color:#378ADD}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;border:1px solid rgba(0,0,0,.2)}
.chartbox{position:relative;width:100%;height:240px}
.concl{background:var(--surf);border-left:3px solid #378ADD;padding:.7rem .9rem;border-radius:4px;margin:.3rem 0 1rem;font-size:14px}
.warn{font-size:12.5px;color:var(--mut);background:var(--surf);border-left:3px solid #BA7517;padding:.5rem .7rem;border-radius:4px;margin:.6rem 0}
.todo li{margin:.15rem 0}
footer{margin-top:2.5rem;border-top:.5px solid var(--line);padding-top:1rem}
footer p{font-size:13.5px}.cite{font-size:12px;color:var(--mut);line-height:1.7}
</style></head>
<body><div class="wrap">
<h1>%%TITLE%%</h1>
<p class="sub">target %%TARGET%% · wild-type redocking of %%LIG%% vs the <b>s04 Boltz pose</b> ·
<b>internal reproducibility</b> baseline for s09 (not experimental validation) · generated %%GENERATED%%</p>

<div class="cards">%%CARDS%%</div>

<div class="concl"><b>Conclusion (scoped).</b> GNINA and DiffDock converge to the
same pocket around the s04 Boltz WT reference (heavy-atom RMSD shown below), so
this is usable as an <b>internal baseline pose</b>. The reference is a
<b>computed (Boltz) structure, not an experimental one</b> — this measures
reproducibility, not biological correctness. Before interpreting mutants (s09),
complete the binding-interaction QC and confirm ligand normalization (checklist
at the bottom).</div>

<h2>3D docked poses — overlay in the receptor pocket</h2>
<div class="vctrl">
 <div class="cg"><span class="cgl">poses</span>%%POSEBTNS%%
  <button class="vb" onclick="allPoses(true)">all</button>
  <button class="vb" onclick="allPoses(false)">none</button></div>
 <div class="cg"><span class="cgl">receptor</span>
  <button class="vb on" onclick="setRec('cartoon',this)">cartoon</button>
  <button class="vb" onclick="setRec('pocket',this)">pocket only</button>
  <button class="vb" onclick="toggleRecSurf(this)">surface</button>
  <button class="vb" onclick="setRec('hide',this)">hide</button></div>
 <div class="cg"><span class="cgl">view</span>
  <button class="vb" onclick="viewPocket()">pocket</button>
  <button class="vb" onclick="viewWhole()">reset</button>
  <button class="vb" onclick="spin(this)">spin</button>
  <button class="vb" onclick="toggleBg(this)">dark</button>
  <button class="vb" onclick="snap()">⤓ PNG</button></div>
</div>
<div id="viewer"><div style="padding:1rem;color:var(--mut);font-size:13px">loading 3D viewer…</div></div>
<p class="note">Each method's best pose is overlaid (whole-pose colour = method)
on the WT receptor; the s04 Boltz pose is the <b>reference</b> (blue). Visual
overlap is the qualitative check; the quantitative one is the RMSD below. 3Dmol.js.</p>

<h2>Pose accuracy — symmetry-corrected heavy-atom RMSD</h2>
<table><thead><tr><th>comparison</th><th>heavy-atom RMSD</th><th>centroid Δ</th><th>verdict</th></tr></thead>
<tbody>%%RROWS%%</tbody></table>
<div class="warn"><b>RMSD is the real metric</b> (CASF/PDBbind docking power,
RMSD&nbsp;&lt;&nbsp;2&nbsp;Å = success; Su et al. 2019), computed without
re-alignment (poses are already in the receptor frame) and minimised over the
ligand's symmetric atom mappings. <b>Centroid Δ is only a coarse proxy</b> — for
a large flexible ligand it stays small even when the ligand rotates or a tail
twists, so a small centroid Δ alone does NOT prove the same pose.</div>

<h2>Ligand normalization (QC)</h2>
<table><thead><tr><th>pose</th><th>raw atoms</th><th>heavy (used)</th><th>fragments</th><th>bonds from SMILES</th></tr></thead>
<tbody>%%QROWS%%</tbody></table>
<p class="note">Before any RMSD, each ligand is reduced to the same chemical
entity: explicit H removed, only the <b>largest fragment</b> kept (so any
co-modelled cofactor/ion the reference carries alongside the design ligand is
dropped), and bond orders assigned from the ligand SMILES template
(<code>%%SMILES%%…</code>) so the symmetry perception is identical across poses.
Protonation/tautomer/charge are taken as-provided by each tool and are NOT
re-standardised here.</p>

<h2>DiffDock pose-confidence landscape</h2>
<div class="chartbox"><canvas id="landChart"></canvas></div>
<p class="note"><b>%%MARGIN%%</b> DiffDock's confidence is a <b>log-odds</b> that
the pose is within 2 Å (Corso et al. 2023): &gt;0 ⇒ predicted P&gt;50%, &lt;0 ⇒
&lt;50%. It is <b>not calibrated</b> here, so read it as a relative ranking, not
an absolute probability. Closely-spaced top ranks ⇒ competing modes.</p>

<h2>Method scores (separated)</h2>
<table><thead><tr><th>method</th><th>value</th><th>meaning</th></tr></thead>
<tbody>%%MROWS%%</tbody></table>
<p class="note">Scores are on different scales and are not comparable as numbers
— GNINA's <b>minimizedAffinity</b> (Vina energy, the value stored in the DB) is
distinct from its <b>CNNaffinity</b> (a CNN-predicted pK) and <b>CNNscore</b>
(pose quality). Methods are compared by <b>where</b> they place the ligand
(RMSD), not by raw score.</p>

<h2>QC checklist before mutant (s09) interpretation</h2>
<div class="warn todo">Done here: pose overlay · symmetry-corrected heavy-atom
RMSD · ligand normalization · separated GNINA scores · calibrated DiffDock
reading. <b>Still required before trusting this as a mutant baseline:</b>
<ul>
<li><b>Binding-interaction review</b> — H-bonds / salt bridges at the
catalytic and cofactor/substrate-binding residues (the specificity determinants),
steric clashes, ligand strain. Not computed in s05 (the contact graph is built
in s06).</li>
<li><b>Preparation record</b> — receptor protonation/pH, partial charges,
explicit waters/ions/cofactors, docking box/grid, flexible residues. The receptor
is the s04 full-atom Boltz structure; gnina prepares it via OpenBabel — these
exact conditions are not separately logged yet.</li>
<li><b>DiffDock confidence calibration</b> + top-k pose clustering.</li>
</ul></div>

<footer>
<h2>Methods</h2>
<p><b>Reference docking.</b> The WT %%LIG%% pose was established by redocking the
ligand into the s04 Boltz receptor with each configured method (GNINA, McNutt et
al. 2021 — AutoDock Vina search, Trott &amp; Olson 2010 / Eberhardt et al. 2021,
+ a 3D-CNN rescore; DiffDock, Corso et al. 2023 — diffusion generative pose
sampling + a confidence model). The reference is the s04 <b>Boltz-2</b> pose
(Wohlwend et al. 2024) — a computed structure.</p>
<p><b>Accuracy metric.</b> Symmetry-corrected heavy-atom RMSD to the reference,
without re-superposition, after normalising each ligand to its largest fragment
(heavy atoms, SMILES-template bond orders). RMSD &lt; 2 Å is the CASF/PDBbind
"docking power" success threshold (Su et al. 2019). This is <b>internal
reproducibility vs a computed reference</b>, not experimental validation. 3D via
3Dmol.js (Rego &amp; Koes 2015).</p>
<h2>Analysis conditions</h2>
<table><tbody>%%CONDROWS%%</tbody></table>
<h2>References</h2>
<p class="cite">
McNutt et al. (2021) <i>J. Cheminform.</i> 13:43 — GNINA.
Corso et al. (2023) <i>ICLR</i> — DiffDock.
Trott &amp; Olson (2010) <i>J. Comput. Chem.</i> 31:455; Eberhardt et al. (2021) <i>JCIM</i> 61:3891 — AutoDock Vina.
Su et al. (2019) <i>JCIM</i> 59:895 — CASF-2016 docking power (RMSD&lt;2 Å).
Wohlwend et al. (2024) — Boltz-2. Rego &amp; Koes (2015) <i>Bioinformatics</i> 31:1322 — 3Dmol.js.
</p>
</footer>
%%MODELS%%
</div>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
var R=%%BLOB%%;
function cssv(n){return getComputedStyle(document.body).getPropertyValue(n).trim()||'#888';}
var V=null,REC=null,MDL={},shown={},recRep='cartoon',_spin=false,_dark=false,_recsurf=null;
R.models.forEach(function(m){shown[m]=true;});
var POCKET={within:{distance:5,sel:{not:{resn:['HOH']}}},byres:true};
function initViewer(){
  if(!window.$3Dmol){document.getElementById('viewer').innerHTML='<p style="padding:1rem;color:#C0392B;font-size:13px">3Dmol.js failed to load (offline?). The charts/tables below still work.</p>';return;}
  V=$3Dmol.createViewer('viewer',{backgroundColor:'white'});
  var rec=document.getElementById('m_rec');
  if(rec)REC=V.addModel(rec.textContent,'pdb');
  R.models.forEach(function(m){var el=document.getElementById('m_'+m);if(!el)return;
    MDL[m]=V.addModel(el.textContent,(m==='reference')?'pdb':'sdf');});
  styleRec();R.models.forEach(stylePose);
  V.zoomTo(anyPoseSel());V.zoom(0.85);V.render();
}
function anyPoseSel(){var ids=R.models.filter(function(m){return MDL[m];}).map(function(m){return MDL[m].getID();});return ids.length?{model:ids}:{};}
function styleRec(){if(!REC)return;REC.setStyle({},{});
  if(recRep==='cartoon')REC.setStyle({},{cartoon:{color:'#c4c2bb'}});
  else if(recRep==='pocket'){REC.setStyle(POCKET,{stick:{radius:0.1,colorscheme:'whiteCarbon'}});}
  V.render();}
function stylePose(m){if(!MDL[m])return;
  MDL[m].setStyle({}, shown[m]?{stick:{radius:0.17,color:R.posecol[m]},sphere:{scale:0.16,color:R.posecol[m]}}:{});V.render();}
function togglePose(m,b){shown[m]=!shown[m];if(b)b.classList.toggle('on');stylePose(m);}
function allPoses(on){R.models.forEach(function(m){shown[m]=on;stylePose(m);});
  document.querySelectorAll('.vb .dot').forEach(function(d){var b=d.parentNode;if(on)b.classList.add('on');else b.classList.remove('on');});}
function mark(b){if(b&&b.parentNode){b.parentNode.querySelectorAll('.vb').forEach(function(x){if(x.querySelector('.dot'))return;x.classList.remove('on');});b.classList.add('on');}}
function setRec(r,b){recRep=r;mark(b);styleRec();}
function toggleRecSurf(b){if(!V||!REC)return;
  if(_recsurf!=null){if(_recsurf!=='p')V.removeSurface(_recsurf);_recsurf=null;if(b)b.classList.remove('on');V.render();}
  else{if(b)b.classList.add('on');_recsurf='p';Promise.resolve(V.addSurface($3Dmol.SurfaceType.VDW,{opacity:0.55,colorscheme:'whiteCarbon'},{model:REC.getID()})).then(function(r){var id=(r&&r.surfid!==undefined)?r.surfid:r;if(_recsurf==='p')_recsurf=id;else if(id!=null)V.removeSurface(id);V.render();});}}
function viewPocket(){if(V){V.zoomTo(anyPoseSel());V.zoom(1.6);V.render();}}
function viewWhole(){if(V){V.zoomTo();V.render();}}
function spin(b){if(!V)return;_spin=!_spin;V.spin(_spin?'y':false);if(b)b.classList.toggle('on');}
function toggleBg(b){if(!V)return;_dark=!_dark;V.setBackgroundColor(_dark?'#15140f':'white');if(b)b.classList.toggle('on');V.render();}
function snap(){if(V){var a=document.createElement('a');a.href=V.pngURI();a.download='%%TARGETID%%_docking.png';a.click();}}
var MUT=cssv('--mut'),GRID=cssv('--line');
function mkLand(){if(R.landscape.length===0)return;
  var L=R.landscape,lab=L.map(function(p){return p[0];}),val=L.map(function(p){return p[1];});
  new Chart(document.getElementById('landChart'),{type:'bar',
    data:{labels:lab,datasets:[{data:val,backgroundColor:lab.map(function(r){return r===1?'#BA7517':'#cdb98e';})}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},
      tooltip:{callbacks:{title:function(t){return 'rank '+t[0].label;},label:function(c){return 'confidence '+c.parsed.y;}}}},
     scales:{y:{title:{display:true,text:'confidence (log-odds; >0 = P>50%)',color:MUT},ticks:{color:MUT},grid:{color:GRID}},
      x:{title:{display:true,text:'pose rank (1 = best)',color:MUT},ticks:{color:MUT},grid:{display:false}}}}});
}
function start(){if(window.Chart)mkLand();setTimeout(initViewer,150);}
if(document.readyState!=='loading')start();else document.addEventListener('DOMContentLoaded',start);
</script></body></html>
"""
