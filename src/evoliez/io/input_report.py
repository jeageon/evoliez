"""Self-contained HTML report for s01 input & ligand preprocessing.

Documents the INPUT PROVENANCE the way a methods section must: where the target
sequence came from (and what is missing), its composition + an integrity hash,
the asserted catalytic / fixed / binding residues with a numbering-verification
warning, and the full ligand chemistry (formula, formal charge, H-bond
donors/acceptors, rotatable bonds, stereocentres, canonical atom IDs) computed
from the actual SMILES via RDKit — for the design-target ligand AND any
co-modelled cofactors/substrates. Generic for any target/ligand: everything is
read from the config + the parsed ligand, nothing is hard-coded.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

_AA3 = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
    "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
    "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
    "Y": "Tyr", "V": "Val", "X": "Xaa",
}
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"


def _ligand_chem(smiles: str) -> dict:
    """RDKit physicochemical summary of a ligand SMILES (empty dict if RDKit /
    parsing is unavailable — the report then shows the raw SMILES only)."""
    out: dict = {"smiles": smiles}
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return out
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return out
    try:
        centers = Chem.FindMolChiralCenters(
            m, useLegacyImplementation=False, includeUnassigned=True)
        unassigned = sum(1 for _, c in centers if c == "?")
    except Exception:
        centers, unassigned = [], 0
    out.update({
        "canonical": Chem.MolToSmiles(m),
        "formula": rdMolDescriptors.CalcMolFormula(m),
        "mw": round(Descriptors.MolWt(m), 1),
        "charge": Chem.GetFormalCharge(m),
        "heavy": m.GetNumHeavyAtoms(),
        "hbd": Lipinski.NumHDonors(m),
        "hba": Lipinski.NumHAcceptors(m),
        "rot": Lipinski.NumRotatableBonds(m),
        "rings": rdMolDescriptors.CalcNumRings(m),
        "stereo": len(centers),
        "stereo_unassigned": unassigned,
    })
    return out


def compute_input_stats(*, target_id: str, sequence: str,
                        sequence_source: str,
                        ligand: Tuple[str, str, str],
                        extra_ligands: List[Tuple[str, str, str]],
                        residues: Dict[str, List[str]],
                        organism: Optional[str], ec_number: Optional[str],
                        target_ph: Optional[float],
                        ligand_atom_ids: Optional[List[str]] = None) -> dict:
    """All the s01 report needs. `ligand` / `extra_ligands` are (id, type,
    smiles); `residues` maps catalytic|fixed|binding -> raw tokens."""
    seq = sequence.upper()
    comp = {a: seq.count(a) for a in _AA_ORDER}
    # integrity hash over the IDENTITY-defining inputs (same blob the run
    # fingerprint uses): sequence + ligand type + ligand value.
    blob = f"{seq}|{ligand[1]}|{ligand[2]}"
    input_hash = hashlib.sha1(blob.encode()).hexdigest()[:16]

    lid, ltype, lsmiles = ligand
    primary = {"id": lid, "type": ltype, **_ligand_chem(lsmiles),
               "n_atom_ids": len(ligand_atom_ids or [])}
    extras = []
    for eid, etype, esm in extra_ligands:
        extras.append({"id": eid, "type": etype, **_ligand_chem(esm)})

    return {
        "target_id": target_id,
        "sequence_source": sequence_source,
        "length": len(seq),
        "composition": comp,
        "input_hash": input_hash,
        "organism": organism,
        "ec_number": ec_number,
        "target_ph": target_ph,
        "residues": residues,
        "ligand": primary,
        "extra_ligands": extras,
        "x_count": comp.get("X", 0),
    }


# --------------------------------------------------------------------------- #
def build_input_report_html(*, stats: dict, generated: str,
                            provenance: str = "") -> str:
    import html as _html
    import json as _json
    g = _html.escape
    s = stats
    # Provenance stamp replaces the minimal "generated <date>" subtitle so
    # same-day re-runs cannot be confused (run name + config fingerprint + git +
    # per-ligand net charge). Falls back to the bare date when not supplied.
    subtitle = provenance or f"generated {g(generated)}"

    def card(lab, val, sub):
        return (f'<div class="card"><div class="lab">{g(lab)}</div>'
                f'<div class="num">{g(str(val))}</div>'
                f'<div class="sub2">{g(sub)}</div></div>')

    lg = s["ligand"]
    cards = "".join([
        card("length", f"{s['length']} aa", "target sequence"),
        card("ligand", lg["id"], lg.get("formula", lg.get("smiles", "—"))),
        card("ligand charge",
             "—" if "charge" not in lg else f"{lg['charge']:+d}",
             "formal charge"),
        card("organism", s["organism"] or "not provided", "provenance"),
        card("EC", s["ec_number"] or "not provided", "annotation"),
        card("input hash", s["input_hash"], "sha1(seq|ligand)"),
    ])

    # provenance table (with explicit "not provided" gaps the methods need)
    prov = [
        ("target id", s["target_id"]),
        ("sequence source", s["sequence_source"]),
        ("length", f"{s['length']} aa" + (f" ({s['x_count']} non-standard → X)"
                                          if s["x_count"] else "")),
        ("organism", s["organism"] or "⚠ not provided"),
        ("EC number", s["ec_number"] or "⚠ not provided"),
        ("UniProt / PDB accession", "⚠ not provided — add for provenance"),
        ("isoform / residue numbering", "⚠ as-provided — verify vs reference DB"),
        ("target pH", "—" if s["target_ph"] is None else str(s["target_ph"])),
        ("integrity hash", s["input_hash"] + "  (sha1 of sequence|ligand)"),
    ]
    prov_rows = "".join(f"<tr><td class=ck>{g(k)}</td><td>{g(str(v))}</td></tr>"
                        for k, v in prov)

    # residue annotations
    res_rows = ""
    for kind in ("catalytic", "fixed", "binding"):
        toks = s["residues"].get(kind) or []
        res_rows += (f"<tr><td>{g(kind)}</td>"
                     f"<td>{g(', '.join(map(str, toks)) if toks else '— none set')}</td>"
                     f"<td class=ck>{len(toks)}</td></tr>")

    def lig_table(lg, role):
        def f(k, fmt="{}"):
            return "—" if lg.get(k) is None else fmt.format(lg[k])
        rows = [
            ("role", role),
            ("id", lg["id"]),
            ("input type", lg["type"]),
            ("canonical SMILES", lg.get("canonical", lg.get("smiles", "—"))),
            ("formula", f("formula")),
            ("MW", f("mw", "{} g/mol")),
            ("formal charge", "—" if lg.get("charge") is None else f"{lg['charge']:+d}"),
            ("heavy atoms", f("heavy")),
            ("H-bond donors / acceptors", f"{f('hbd')} / {f('hba')}"),
            ("rotatable bonds", f("rot")),
            ("rings", f("rings")),
            ("stereocentres (unassigned)",
             f"{f('stereo')} ({f('stereo_unassigned')})"),
        ]
        if lg.get("n_atom_ids"):
            rows.append(("canonical atom IDs", f"{lg['n_atom_ids']} (atom-lock reference)"))
        return "".join(f"<tr><td class=ck>{g(k)}</td><td>{g(str(v))}</td></tr>"
                       for k, v in rows)

    primary_tbl = lig_table(lg, "design target (affinity binder)")
    extra_html = ""
    for e in s["extra_ligands"]:
        extra_html += (f'<h3 style="font-size:15px;margin:1rem 0 .3rem">'
                       f'co-modelled: {g(e["id"])}</h3><table><tbody>'
                       + lig_table(e, "structural context (not the optimisation target)")
                       + "</tbody></table>")
    if not s["extra_ligands"]:
        extra_html = '<p class="note">No co-modelled cofactors/substrates configured.</p>'

    comp_blob = _json.dumps({"order": list(_AA_ORDER),
                             "counts": [s["composition"][a] for a in _AA_ORDER]},
                            separators=(",", ":"))

    has_unassigned = (lg.get("stereo_unassigned") or 0) > 0
    return (_INPUT_TEMPLATE
            .replace("%%TITLE%%", g(f"{s['target_id']} — input & ligand QC (s01)"))
            .replace("%%TARGET%%", g(s["target_id"]))
            .replace("%%PROVSUB%%", subtitle)
            .replace("%%CARDS%%", cards)
            .replace("%%PROVROWS%%", prov_rows)
            .replace("%%RESROWS%%", res_rows)
            .replace("%%PRIMARY%%", primary_tbl)
            .replace("%%EXTRA%%", extra_html)
            .replace("%%STEREONOTE%%",
                     ("⚠ unassigned stereocentres — fix the SMILES stereochemistry "
                      "before a real run." if has_unassigned else
                      "All stereocentres assigned."))
            .replace("%%COMPBLOB%%", comp_blob))


def write_input_report(path, **kwargs) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_input_report_html(**kwargs), encoding="utf-8")


_INPUT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
:root{--bg:#fff;--fg:#1c1c1a;--mut:#5f5e5a;--line:#e7e5df;--surf:#f7f6f2}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a18;--fg:#ece9e3;--mut:#a8a69e;--line:#33322e;--surf:#232220}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:2rem 1.25rem 3rem}
h1{font-size:22px;font-weight:500;margin:0 0 2px}
h2{font-size:18px;font-weight:500;margin:2.2rem 0 .5rem}
.sub{color:var(--mut);font-size:14px;margin:0 0 1.5rem}
.note{font-size:13px;color:var(--mut);margin:.3rem 0 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:0 0 1rem}
.card{background:var(--surf);border-radius:8px;padding:.7rem .85rem}
.lab{font-size:12px;color:var(--mut)}.num{font-size:19px;font-weight:500;margin-top:1px;word-break:break-word}.sub2{font-size:11px;color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:.5px solid var(--line);vertical-align:top}
th{color:var(--mut);font-weight:500;font-size:13px}
td.ck{color:var(--mut);width:38%}
.chartbox{position:relative;width:100%;height:190px}
.warn{font-size:12.5px;color:var(--mut);background:var(--surf);border-left:3px solid #BA7517;padding:.5rem .7rem;border-radius:4px;margin:.6rem 0}
.todo li{margin:.15rem 0}
footer{margin-top:2.5rem;border-top:.5px solid var(--line);padding-top:1rem}
footer p{font-size:13.5px}.cite{font-size:12px;color:var(--mut);line-height:1.7}
code{font-size:12.5px;word-break:break-all}
</style></head>
<body><div class="wrap">
<h1>%%TITLE%%</h1>
<p class="sub">target %%TARGET%% · input provenance + ligand chemistry QC · %%PROVSUB%%</p>

<div class="cards">%%CARDS%%</div>

<h2>Provenance</h2>
<table><tbody>%%PROVROWS%%</tbody></table>
<div class="warn">Items marked ⚠ are <b>required for a methods section but were
not supplied</b>. Residue numbering is taken AS-PROVIDED; the asserted residues
below must be checked against the same reference (UniProt/PDB) the numbering
came from — a 1-residue offset mis-places the whole active site.</div>

<h2>Sequence composition</h2>
<div class="chartbox"><canvas id="compChart"></canvas></div>
<p class="note">Amino-acid counts over the %%TARGET%% sequence (X = non-standard
residues, replaced before downstream use). Integrity hash above pins this exact
sequence + ligand for reproducibility.</p>

<h2>Asserted residues (numbering as-provided)</h2>
<table><thead><tr><th>role</th><th>positions</th><th>n</th></tr></thead>
<tbody>%%RESROWS%%</tbody></table>
<p class="note">Catalytic = held fixed + geometry-checked; fixed = never mutated;
binding = known site. Empty roles are not an error — they are simply unset, in
which case design proceeds without those constraints.</p>

<h2>Design-target ligand</h2>
<table><tbody>%%PRIMARY%%</tbody></table>
<p class="note">%%STEREONOTE%% Charge / protonation / tautomer are taken
AS-WRITTEN in the input SMILES — for a real run they should reflect the dominant
microspecies at the target pH (e.g. phosphate deprotonation), which is NOT
auto-standardised here.</p>

<h2>Co-modelled ligands</h2>
%%EXTRA%%
<p class="note">Co-modelled cofactors/substrates are folded into the s04 complex
as their OWN chains to complete the active site, but are <b>structural context,
not the optimisation target</b> — the design objective acts on the design-target
ligand only.</p>

<footer>
<h2>QC checklist (methods-grade input)</h2>
<div class="warn todo">Captured here: sequence + length + composition + integrity
hash · asserted residues with numbering-mismatch warnings (s01 validates the
WT-letter of each token) · per-ligand formula/charge/heavy/donors/acceptors/
rotatable/stereo from RDKit. <b>Add before publication:</b>
<ul>
<li>UniProt/PDB <b>accession + isoform</b> and the residue-numbering reference.</li>
<li>Ligand <b>protonation/charge/tautomer state at the target pH</b> (and the
reason for co-modelling each cofactor/substrate).</li>
<li>Confirm catalytic/binding residue numbers against the cited structure.</li>
</ul></div>
<h2>Methods</h2>
<p><b>Input preprocessing.</b> The target sequence is validated (non-standard
residues → X), the ligand(s) are parsed with <b>RDKit</b> (Landrum) — a real run
HARD-FAILS if RDKit cannot parse a ligand, so no downstream chemistry is ever
synthetic — and catalytic/fixed/binding residue tokens are checked for
in-range positions and WT-letter consistency with the sequence. Physicochemical
descriptors are RDKit defaults (Lipinski donor/acceptor counts, formal charge,
canonical SMILES).</p>
<p class="cite">RDKit — Open-source cheminformatics, https://www.rdkit.org.</p>
</footer>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
var C=%%COMPBLOB%%;
function cssv(n){return getComputedStyle(document.body).getPropertyValue(n).trim()||'#888';}
function start(){
  if(!window.Chart)return;var MUT=cssv('--mut'),GRID=cssv('--line');
  new Chart(document.getElementById('compChart'),{type:'bar',
    data:{labels:C.order,datasets:[{data:C.counts,backgroundColor:C.order.map(function(a){return a==='X'?'#C0392B':'#7F77DD';})}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
     scales:{y:{title:{display:true,text:'count',color:MUT},ticks:{color:MUT},grid:{color:GRID}},
      x:{ticks:{color:MUT},grid:{display:false}}}}});
}
if(document.readyState!=='loading')start();else document.addEventListener('DOMContentLoaded',start);
</script></body></html>
"""
