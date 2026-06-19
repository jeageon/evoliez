"""Self-contained HTML report for the s03 integrated-MSA analysis.

Presents the final multi-source MSA the way structure/evolution papers do:
an AlphaFold/ColabFold-style coverage map, a per-position conservation +
information-content track, a sequence-identity-to-query distribution, a
per-track contribution breakdown, and a sequence logo for the most conserved
window. Everything is derived from the ACTUAL alignment + per-position features,
so it works unchanged for any target. The render layer lives in this module too;
this top section is the viz-agnostic DATA layer.
"""

from __future__ import annotations

from collections import Counter
from math import exp, log, log2
from typing import Dict, List, Optional, Sequence, Tuple

_AA = "ACDEFGHIKLMNPQRSTVWY"

# per-track colours (match the homolog report) — the 5 independent retrievers.
_TRACK_COLORS = {
    "colabfold": "#378ADD", "foldseek": "#1D9E75", "mmseqs": "#7F77DD",
    "hhblits": "#993C1D", "jackhmmer": "#BA7517", "other": "#888780",
}


def _track_of(seq_id: str) -> str:
    """Recover which retrieval track a row came from, from its id prefix
    (UniRef* = ColabFold remote a3m; mm_/fs_/hh_/jh_ = the local tracks)."""
    if seq_id.startswith("UniRef"):
        return "colabfold"
    return {"mm": "mmseqs", "fs": "foldseek", "hh": "hhblits",
            "jh": "jackhmmer"}.get(seq_id.split("_")[0], "other")


def _identity_to_query(row: str, target: str) -> Tuple[float, int]:
    """Fraction of this row's ALIGNED columns that match the query, plus the
    aligned-column count (= columns where the row is not a gap; the query is
    gap-free because the MSA is target-anchored)."""
    aligned = matches = 0
    for r, t in zip(row, target):
        if r != "-":
            aligned += 1
            if r == t:
                matches += 1
    return (matches / aligned if aligned else 0.0), aligned


def _col_info_bits(column_counts: Counter) -> Tuple[float, str]:
    """Shannon information content (bits, gaps excluded) of a column and its
    consensus residue. 0 bits = uniform over 20 aa, log2(20)≈4.32 = invariant."""
    tot = sum(column_counts.values())
    if not tot:
        return 0.0, "-"
    h = -sum((c / tot) * log2(c / tot) for c in column_counts.values())
    consensus = max(column_counts, key=column_counts.get)
    return log2(20) - h, consensus


def compute_msa_stats(ids: Sequence[str], seqs: Sequence[str],
                      conservation: Optional[Dict] = None,
                      n_row_bins: int = 300) -> dict:
    """All numbers the report needs, derived from the target-anchored MSA
    (ids[0]/seqs[0] = query). `conservation` is the optional s03 conservation
    .json (per target-position conservation/entropy/gap); recomputed if absent."""
    n, L = len(seqs), len(seqs[0]) if seqs else 0
    target = seqs[0]

    # ---- per-column residue counts (gaps excluded) -> info content + logo ----
    col_counts: List[Counter] = [Counter() for _ in range(L)]
    coverage = [0] * L
    for s in seqs:
        for j, ch in enumerate(s):
            if ch == "-":
                continue
            coverage[j] += 1
            if ch in _AA:
                col_counts[j][ch] += 1
    info_bits: List[float] = []
    consensus: List[str] = []
    col_entropy_nats: List[float] = []
    for j in range(L):
        bits, cons = _col_info_bits(col_counts[j])
        info_bits.append(round(bits, 3))
        consensus.append(cons)
        tot = sum(col_counts[j].values())
        h = (-sum((c / tot) * log(c / tot) for c in col_counts[j].values())
             if tot else 0.0)
        col_entropy_nats.append(h)

    # ---- per-row identity to query + track ----
    rows = []  # (identity, track, row_index)
    for i in range(1, n):
        id, _ = _identity_to_query(seqs[i], target)
        rows.append((idd := round(id, 4), _track_of(ids[i]), i))
    order = sorted(range(len(rows)), key=lambda k: rows[k][0], reverse=True)

    # ---- identity histogram (overall + per track), 0.0..1.0 in 0.05 bins ----
    edges = [round(0.05 * k, 2) for k in range(21)]
    labels = [f"{edges[k]:.2f}" for k in range(20)]
    tracks = sorted({t for _, t, _ in rows}, key=lambda t: -sum(
        1 for _, tt, _ in rows if tt == t))
    id_hist: Dict[str, List[int]] = {t: [0] * 20 for t in tracks}
    overall_hist = [0] * 20
    for idv, t, _ in rows:
        b = min(19, int(idv / 0.05))
        id_hist[t][b] += 1
        overall_hist[b] += 1
    track_summary = {
        t: {"count": sum(1 for _, tt, _ in rows if tt == t),
            "color": _TRACK_COLORS.get(t, _TRACK_COLORS["other"]),
            "median_id": _median([idv for idv, tt, _ in rows if tt == t])}
        for t in tracks
    }

    # ---- coverage map (ColabFold-style): rows sorted by identity, binned to
    #      n_row_bins; per bin = mean identity, dominant track, per-column
    #      coverage fraction encoded as a 0-9 digit string (canvas heatmap) ----
    nb = min(n_row_bins, len(order)) or 1
    cov_rows = []
    for b in range(nb):
        lo = b * len(order) // nb
        hi = max(lo + 1, (b + 1) * len(order) // nb)
        members = [rows[order[k]] for k in range(lo, hi)]
        idx = [m[2] for m in members]
        digits = []
        for j in range(L):
            c = sum(1 for ri in idx if seqs[ri][j] != "-")
            digits.append(str(min(9, int(9 * c / len(idx)))))
        cov_rows.append({
            "id": round(sum(m[0] for m in members) / len(members), 3),
            "track": Counter(m[1] for m in members).most_common(1)[0][0],
            "cov": "".join(digits),
        })

    # ---- conservation track (prefer the s03 features; else from columns) ----
    if conservation:
        cons_track = [conservation.get(str(p), {}).get("conservation", 0.0)
                      for p in range(1, L + 1)]
        gap_freq = [conservation.get(str(p), {}).get("gap_frequency", 0.0)
                    for p in range(1, L + 1)]
    else:
        cons_track = [round(max(c.values()) / sum(c.values()), 3) if sum(c.values())
                      else 0.0 for c in col_counts]
        gap_freq = [round(1 - coverage[j] / n, 3) for j in range(L)]

    # ---- Neff (HHsuite-style: exp of mean per-column entropy, nats) ----
    neff = round(exp(sum(col_entropy_nats) / L), 1) if L else 0.0

    # ---- sequence logo for the most-conserved contiguous window (<=40 cols) ----
    logo = _logo_window(col_counts, info_bits, coverage, consensus)

    well = sum(1 for c in coverage if c > 0.5 * n)
    return {
        "n_seqs": n, "aln_len": L, "neff": neff,
        "mean_info": round(sum(info_bits) / L, 3) if L else 0.0,
        "mean_cons": round(sum(cons_track) / L, 3) if L else 0.0,
        "mean_cov_frac": round(sum(coverage) / (n * L), 3) if n * L else 0.0,
        "well_aligned": well, "n_highly_conserved": sum(1 for b in info_bits if b > 2),
        "positions": list(range(1, L + 1)),
        "coverage": coverage, "info_bits": info_bits, "consensus": "".join(consensus),
        "conservation": [round(x, 3) for x in cons_track],
        "gap_freq": [round(x, 3) for x in gap_freq],
        "id_labels": labels, "overall_hist": overall_hist, "id_hist": id_hist,
        "tracks": tracks, "track_summary": track_summary,
        "cov_rows": cov_rows, "logo": logo,
    }


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return round(s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2, 3)


def _logo_window(col_counts, info_bits, coverage, consensus, width: int = 40):
    """Pick the highest-summed-information window of `width` columns and return
    per-column residue stacks (residue, height-in-bits) for a sequence logo."""
    L = len(info_bits)
    if L == 0:
        return {"start": 1, "stacks": []}
    w = min(width, L)
    best_s, best_v = 0, -1.0
    run = sum(info_bits[:w])
    best_v, best_s = run, 0
    for s in range(1, L - w + 1):
        run += info_bits[s + w - 1] - info_bits[s - 1]
        if run > best_v:
            best_v, best_s = run, s
    stacks = []
    for j in range(best_s, best_s + w):
        tot = sum(col_counts[j].values())
        ic = info_bits[j]
        items = sorted(col_counts[j].items(), key=lambda kv: kv[1], reverse=True)
        stacks.append([[aa, round(cnt / tot * ic, 3)] for aa, cnt in items
                       if cnt / tot * ic > 0.01] if tot else [])
    return {"start": best_s + 1, "stacks": stacks, "consensus":
            consensus[best_s:best_s + w]}


def effective_neff(seqs: Sequence[str], theta: float = 0.8,
                   sample: int = 160) -> Optional[float]:
    """AlphaFold2/DCA-style effective sequence count: Neff = Σ_k 1/m_k, where
    m_k = #sequences with >= theta identity to k (identity measured over columns
    non-gap in EITHER sequence, AF2 convention). Estimated from `sample` evenly
    spaced query rows scored against ALL sequences and scaled by N/sample (an
    unbiased estimator of the full sum). Needs numpy; None if unavailable."""
    try:
        import numpy as np
    except Exception:
        return None
    n = len(seqs)
    if n == 0:
        return None
    code = {c: i + 1 for i, c in enumerate(_AA)}
    m = np.array([[code.get(c, 0) for c in s] for s in seqs], dtype=np.int8)
    nong = m != 0
    idx = list(range(0, n, max(1, n // min(sample, n))))
    acc = 0.0
    for k in idx:
        rk, nk = m[k], nong[k]
        both = nong & nk
        either = nong | nk
        idkn = ((m == rk) & both).sum(1) / np.maximum(either.sum(1), 1)
        acc += 1.0 / max(1, int((idkn >= theta).sum()))
    return round((n / len(idx)) * acc, 1)


def build_msa_report_html(*, target_id: str, target_len: int, stats: dict,
                          conditions: List[Tuple[str, str]], generated: str) -> str:
    import html as _html
    import json as _json
    g = _html.escape
    s = stats
    neff80 = s.get("neff80")
    cards = [
        ("MSA depth", f"{s['n_seqs']:,}", "sequences"),
        ("columns", f"{s['aln_len']:,}", "target positions"),
        ("Neff (eff., 80% id)", f"{neff80:,.0f}" if neff80 else "—", "AF2 convention"),
        ("Neff / L", f"{neff80 / s['aln_len']:.1f}" if neff80 else "—", "per residue"),
        ("Neff (HH-suite)", f"{s['neff']}", "column diversity"),
        ("mean conservation", f"{s['mean_cons']:.2f}", "1 − H/Hₘₐₓ"),
        ("well-aligned cols", f"{s['well_aligned']}", ">50% occupancy"),
        ("highly conserved", f"{s['n_highly_conserved']}", ">2 bits"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="lab">{g(lab)}</div>'
        f'<div class="num">{g(val)}</div><div class="sub2">{g(sub)}</div></div>'
        for lab, val, sub in cards
    )
    legend = "".join(
        f'<span class="lg"><span class="sw" style="background:{ts["color"]}">'
        f'</span>{g(t)} ({ts["count"]:,}, med {ts["median_id"]:.0%})</span>'
        for t, ts in ((t, s["track_summary"][t]) for t in s["tracks"])
    )
    cond_rows = "".join(
        f"<tr><td class=ck>{g(k)}</td><td>{g(str(v))}</td></tr>"
        for k, v in conditions
    )
    track_colors = {t: s["track_summary"][t]["color"] for t in s["tracks"]}
    blob = _json.dumps({
        "n_seqs": s["n_seqs"], "aln_len": s["aln_len"],
        "coverage": s["coverage"], "conservation": s["conservation"],
        "info_bits": s["info_bits"], "gap_freq": s["gap_freq"],
        "id_labels": s["id_labels"], "id_hist": s["id_hist"],
        "tracks": s["tracks"], "track_colors": track_colors,
        "cov_rows": s["cov_rows"], "logo": s["logo"],
    }, separators=(",", ":"))
    return (_MSA_TEMPLATE
            .replace("%%TITLE%%", g(f"{target_id} — integrated MSA analysis"))
            .replace("%%TARGET%%", g(target_id))
            .replace("%%TLEN%%", str(target_len))
            .replace("%%GENERATED%%", g(generated))
            .replace("%%DEPTH%%", f"{s['n_seqs']:,}")
            .replace("%%CARDS%%", cards_html)
            .replace("%%LEGEND%%", legend)
            .replace("%%LOGOSTART%%", str(s["logo"]["start"]))
            .replace("%%LOGOEND%%", str(s["logo"]["start"] + len(s["logo"]["stacks"]) - 1))
            .replace("%%COND_ROWS%%", cond_rows)
            .replace("%%BLOB%%", blob))


def write_msa_report(path, **kwargs) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_msa_report_html(**kwargs), encoding="utf-8")


_MSA_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
:root{--bg:#fff;--fg:#1c1c1a;--mut:#5f5e5a;--line:#e7e5df;--surf:#f7f6f2}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a18;--fg:#ece9e3;--mut:#a8a69e;--line:#33322e;--surf:#232220}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:940px;margin:0 auto;padding:2rem 1.25rem 3rem}
h1{font-size:22px;font-weight:500;margin:0 0 2px}
h2{font-size:18px;font-weight:500;margin:2.2rem 0 .5rem}
.sub{color:var(--mut);font-size:14px;margin:0 0 1.5rem}
.note{font-size:13px;color:var(--mut);margin:.3rem 0 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:0 0 1rem}
.card{background:var(--surf);border-radius:8px;padding:.7rem .85rem}
.lab{font-size:12px;color:var(--mut)}
.num{font-size:22px;font-weight:500;margin-top:1px}
.sub2{font-size:11px;color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:7px 10px;border-bottom:.5px solid var(--line)}
th{color:var(--mut);font-weight:500;font-size:13px}
td.ck{color:var(--mut);width:42%}
.legend{display:flex;flex-wrap:wrap;gap:13px;font-size:13px;color:var(--mut);margin:.2rem 0 .6rem}
.lg{display:flex;align-items:center;gap:5px}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block}
.chartbox{position:relative;width:100%;height:230px}
#covmap{width:100%;height:340px;image-rendering:pixelated;image-rendering:crisp-edges;border:.5px solid var(--line);border-radius:4px;display:block}
.cbar{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--mut);margin:.4rem 0 .2rem}
.grad{height:10px;width:160px;border-radius:3px;background:linear-gradient(90deg,hsl(240,70%,48%),hsl(180,70%,48%),hsl(120,70%,48%),hsl(60,70%,48%),hsl(0,70%,48%))}
.btn{font:inherit;font-size:12px;color:var(--fg);background:var(--surf);border:.5px solid var(--line);border-radius:5px;padding:3px 9px;cursor:pointer}
.logobox{width:100%;overflow-x:auto;border:.5px solid var(--line);border-radius:4px;background:var(--surf)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;color:var(--mut);word-break:break-all}
footer{margin-top:2.5rem;border-top:.5px solid var(--line);padding-top:1rem}
footer p{font-size:13.5px;color:var(--fg)}
.cite{font-size:12px;color:var(--mut);line-height:1.7}
</style></head>
<body><div class="wrap">
<h1>%%TITLE%%</h1>
<p class="sub">target %%TARGET%% · %%TLEN%% aa · %%DEPTH%% sequences · generated %%GENERATED%%</p>

<div class="cards">%%CARDS%%</div>

<h2>MSA coverage map</h2>
<div class="legend">%%LEGEND%%</div>
<div class="cbar" id="cbId"><span>sequence identity to query</span>
 <span>low</span><div class="grad"></div><span>high</span>
 <button class="btn" id="cbtn" onclick="toggleColor()">color: identity ⇄ track</button></div>
<div class="cbar" id="cbTrack" style="display:none"><span>colored by retrieval track</span>
 <button class="btn" onclick="toggleColor()">color: track ⇄ identity</button></div>
<canvas id="covmap" role="img" aria-label="MSA coverage map: sequences (rows, sorted by identity to query) by alignment position (columns); colored cells are aligned residues, the dark line is per-position coverage."></canvas>
<p class="note">Rows = the %%DEPTH%% sequences, sorted by identity to the query (most
similar at top); columns = the 384 target positions. A coloured cell = that
sequence is aligned (non-gap) there. The dark overlaid line is the per-position
coverage (number of aligned sequences, 0…depth) — dips mark weakly supported
regions (termini, insertions). After AlphaFold2 (Jumper 2021) / ColabFold (Mirdita 2022).</p>

<h2>Per-position coverage &amp; conservation</h2>
<div class="chartbox"><canvas id="occChart"></canvas></div>
<p class="note">Occupancy = fraction of sequences aligned at each column (1 − gap
fraction); dashed line = 50% threshold below which a column is gap-dominated.</p>
<div class="chartbox" style="margin-top:1rem"><canvas id="consChart"></canvas></div>
<p class="note">Conservation = 1 − H/H<sub>max</sub> from per-column Shannon entropy
H = −Σ p<sub>a</sub> log₂ p<sub>a</sub> (gaps excluded); information content in bits
(right axis, max log₂20 ≈ 4.32). Conserved peaks = catalytic/structural cores.</p>

<h2>Sequence identity to query, by track</h2>
<div class="chartbox"><canvas id="idChart"></canvas></div>
<p class="note">Distribution of each sequence's identity to the query, stacked by
retrieval track. The structure track (Foldseek) populates the low-identity
"twilight zone" the sequence tracks cannot reach — the tracks are complementary,
which is the rationale for the 5-track design.</p>

<h2>Sequence logo — most conserved window (positions %%LOGOSTART%%–%%LOGOEND%%)</h2>
<div class="logobox"><div id="logo"></div></div>
<p class="note">Information-content logo (bits): stack height = column information
content R = log₂20 − H; letter height ∝ residue frequency. Colour = chemistry
(green polar, blue basic, red acidic, dark hydrophobic). After WebLogo (Crooks 2004).</p>

<footer>
<h2>Methods</h2>
<p><b>MSA construction.</b> A target-anchored multiple sequence alignment was built
for the query (%%TLEN%% aa) by merging five independent homolog-retrieval tracks,
each run separately so tools with different sensitivity profiles contribute
partly-distinct homolog pools: (i) MMseqs2 GPU iterative profile search vs UniRef30;
(ii) ColabFold remote MSA (MMseqs2 vs UniRef30 + the environmental ColabFoldDB);
(iii) jackhmmer (HMMER 3.4, 3 iterations) vs UniRef30; (iv) HHblits (HH-suite3,
2 iterations) vs UniRef30; (v) Foldseek structure search (ProstT5 sequence→3Di) vs
the PDB. Each track's hits were placed into the query coordinate frame using their
native alignment — sequence tracks via their a3m/Stockholm query columns, the
structure track via the Foldseek backbone alignment — rather than re-aligning, and
the merged rows were de-duplicated. Hits were retained in the 20–95% identity band
and clustered into subfamilies by k-mer Jaccard, yielding a final alignment of
%%DEPTH%% sequences over %%TLEN%% columns.</p>
<p><b>Conservation &amp; information content.</b> Per-column Shannon entropy
H(j) = −Σ<sub>a</sub> p<sub>a</sub> log₂ p<sub>a</sub> was computed over observed
residues (gaps excluded), reported as conservation (1 − H/log₂20) and information
content R(j) = log₂20 − H(j) bits; per-column occupancy (non-gap fraction) is shown
alongside so conservation is not confounded with coverage. Sequence logos use
information-content (bits) scaling.</p>
<p><b>Depth &amp; diversity.</b> Alignment depth is the raw sequence count; the
effective number of sequences (Neff) is reported both as the AlphaFold2 reweighted
count at 80% identity (measured over columns non-gap in either sequence) with its
length-normalised Neff/L, and as the HH-suite per-column diversity exp(mean column
entropy). The MSA and its per-position conservation/coevolution features feed the
downstream structure-prediction and evolutionary-analysis stages.</p>
<h2>Analysis conditions</h2>
<table><tbody>%%COND_ROWS%%</tbody></table>
<h2>References</h2>
<p class="cite">
Jumper et al. (2021) <i>Nature</i> 596:583 — AlphaFold2 (MSA depth/Neff convention).
Mirdita et al. (2022) <i>Nat. Methods</i> 19:679 — ColabFold (coverage plot).
Steinegger &amp; Söding (2017) <i>Nat. Biotechnol.</i> 35:1026 — MMseqs2.
Eddy (2011) <i>PLoS Comput. Biol.</i> 7:e1002195 — HMMER/jackhmmer.
Steinegger et al. (2019) <i>BMC Bioinformatics</i> 20:473 — HH-suite3.
van Kempen et al. (2024) <i>Nat. Biotechnol.</i> 42:243 — Foldseek.
Crooks et al. (2004) <i>Genome Res.</i> 14:1188 — WebLogo.
Henikoff &amp; Henikoff (1994) <i>J. Mol. Biol.</i> 243:574 — sequence weighting.
Shannon (1948) <i>Bell Syst. Tech. J.</i> 27:379 — entropy.
</p>
</footer>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
var R=%%BLOB%%;
function cssv(n){return getComputedStyle(document.body).getPropertyValue(n).trim()||'#888';}
function idCol(v){return 'hsl('+Math.round((1-v)*240)+',70%,48%)';}
function idColA(v,a){return 'hsla('+Math.round((1-v)*240)+',70%,48%,'+a+')';}
var COLOR_BY='id';
function drawCov(){
  var c=document.getElementById('covmap');if(!c)return;var x=c.getContext('2d');
  var rows=R.cov_rows,L=R.aln_len,H=rows.length;c.width=L;c.height=H;x.clearRect(0,0,L,H);
  for(var b=0;b<H;b++){var r=rows[b];
    for(var j=0;j<L;j++){var d=r.cov.charCodeAt(j)-48;if(d<=0)continue;var a=0.2+0.8*d/9;
      if(COLOR_BY==='id'){x.fillStyle=idColA(r.id,a);}
      else{x.globalAlpha=a;x.fillStyle=R.track_colors[r.track]||'#888';}
      x.fillRect(j,b,1,1);x.globalAlpha=1;}}
  var cv=R.coverage,N=R.n_seqs;x.strokeStyle=cssv('--fg');x.lineWidth=Math.max(0.5,H/500);
  x.globalAlpha=0.85;x.beginPath();
  for(var j=0;j<L;j++){var y=H*(1-cv[j]/N);j?x.lineTo(j+0.5,y):x.moveTo(j+0.5,y);}
  x.stroke();x.globalAlpha=1;
}
function toggleColor(){COLOR_BY=COLOR_BY==='id'?'track':'id';drawCov();
  document.getElementById('cbId').style.display=COLOR_BY==='id'?'':'none';
  document.getElementById('cbTrack').style.display=COLOR_BY==='track'?'':'none';}

var GRID=cssv('--line'),MUT=cssv('--mut');
function baseOpts(y1,y2){var o={responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false}},
  scales:{x:{title:{display:true,text:'target position',color:MUT},
    ticks:{color:MUT,maxTicksLimit:12},grid:{display:false}},
   y:{title:{display:true,text:y1,color:MUT},ticks:{color:MUT},grid:{color:GRID}}}};
  if(y2){o.scales.y1={position:'right',title:{display:true,text:y2,color:MUT},
    ticks:{color:MUT},grid:{display:false},min:0,max:4.32};}
  return o;}
function mkLine(){
  var pos=R.coverage.map(function(_,i){return i+1;});
  var occ=R.coverage.map(function(c){return +(c/R.n_seqs).toFixed(3);});
  new Chart(document.getElementById('occChart'),{type:'line',
    data:{labels:pos,datasets:[
      {label:'occupancy',data:occ,borderColor:'#378ADD',backgroundColor:'rgba(55,138,221,.12)',
       fill:true,pointRadius:0,borderWidth:1.2},
      {label:'50%',data:pos.map(function(){return 0.5;}),borderColor:MUT,borderDash:[5,4],
       pointRadius:0,borderWidth:1}]},
    options:Object.assign(baseOpts('occupancy (fraction aligned)'),{scales:Object.assign(baseOpts('occupancy (fraction aligned)').scales,{y:{min:0,max:1,title:{display:true,text:'occupancy',color:MUT},ticks:{color:MUT},grid:{color:GRID}}})})});
  new Chart(document.getElementById('consChart'),{type:'line',
    data:{labels:pos,datasets:[
      {label:'conservation',data:R.conservation,borderColor:'#1D9E75',
       backgroundColor:'rgba(29,158,117,.12)',fill:true,pointRadius:0,borderWidth:1.2,yAxisID:'y'},
      {label:'information (bits)',data:R.info_bits,borderColor:'#BA7517',
       pointRadius:0,borderWidth:1,yAxisID:'y1'}]},
    options:Object.assign(baseOpts('conservation (1 − H/Hmax)','information (bits)'),
      {scales:Object.assign(baseOpts('conservation','bits').scales,
       {y:{min:0,max:1,title:{display:true,text:'conservation',color:MUT},ticks:{color:MUT},grid:{color:GRID}}})})});
}
function mkId(){
  var ds=R.tracks.map(function(t){return {label:t,data:R.id_hist[t],
    backgroundColor:R.track_colors[t]};});
  new Chart(document.getElementById('idChart'),{type:'bar',
    data:{labels:R.id_labels,datasets:ds},
    options:{responsive:true,maintainAspectRatio:false,
     plugins:{legend:{display:true,labels:{color:MUT,boxWidth:12}}},
     scales:{x:{stacked:true,title:{display:true,text:'identity to query',color:MUT},
       ticks:{color:MUT,maxTicksLimit:11},grid:{display:false}},
      y:{stacked:true,title:{display:true,text:'sequences',color:MUT},
       ticks:{color:MUT},grid:{color:GRID}}}}});
}
function genLogo(){
  var lg=R.logo,st=lg.stacks;if(!st||!st.length)return;
  var n=st.length,W=Math.max(640,n*22),Hh=210,pad=30,base=Hh-22;
  var pxb=(Hh-30)/4.322;
  var CHEM={};'GSTYCQN'.split('').forEach(function(a){CHEM[a]='#1D9E75';});
  'KRH'.split('').forEach(function(a){CHEM[a]='#2C6FBB';});
  'DE'.split('').forEach(function(a){CHEM[a]='#C0392B';});
  function col(a){return CHEM[a]||'#3a382f';}
  var colW=(W-pad-6)/n;
  var s='<svg viewBox="0 0 '+W+' '+Hh+'" width="'+W+'" height="'+Hh+'" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,Menlo,monospace">';
  s+='<line x1="'+pad+'" y1="'+base+'" x2="'+W+'" y2="'+base+'" stroke="'+GRID+'"/>';
  s+='<line x1="'+pad+'" y1="8" x2="'+pad+'" y2="'+base+'" stroke="'+GRID+'"/>';
  for(var b=0;b<=4;b++){var y=base-b*pxb;s+='<text x="'+(pad-5)+'" y="'+(y+3)+'" font-size="9" fill="'+MUT+'" text-anchor="end">'+b+'</text>';
    s+='<line x1="'+(pad-2)+'" y1="'+y+'" x2="'+pad+'" y2="'+y+'" stroke="'+GRID+'"/>';}
  for(var j=0;j<n;j++){var stack=st[j].slice().sort(function(p,q){return p[1]-q[1];});
    var x0=pad+j*colW,y=base;
    for(var i=0;i<stack.length;i++){var aa=stack[i][0],segH=stack[i][1]*pxb;if(segH<0.7)continue;
      var sx=(colW-1)/9.6,sy=segH/11.2;
      s+='<text transform="translate('+x0.toFixed(2)+','+y.toFixed(2)+') scale('+sx.toFixed(3)+','+sy.toFixed(3)+')" font-size="16" font-weight="700" fill="'+col(aa)+'">'+aa+'</text>';
      y-=segH;}
    if(j%5===0){s+='<text x="'+(x0+colW/2)+'" y="'+(Hh-7)+'" font-size="8.5" fill="'+MUT+'" text-anchor="middle">'+(lg.start+j)+'</text>';}}
  s+='<text x="9" y="'+(Hh/2)+'" font-size="10" fill="'+MUT+'" transform="rotate(-90,9,'+(Hh/2)+')" text-anchor="middle">bits</text></svg>';
  document.getElementById('logo').innerHTML=s;
}
function start(){drawCov();genLogo();if(window.Chart){mkLine();mkId();}}
if(document.readyState!=='loading')start();else document.addEventListener('DOMContentLoaded',start);
window.addEventListener('resize',function(){drawCov();});
</script></body></html>
"""
