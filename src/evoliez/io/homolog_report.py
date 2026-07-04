"""Self-contained HTML report for the s02 integrated-homolog analysis.

Auto-generated when s02 finishes. Everything (identity bins, cluster buckets,
axis ranges, per-source colours, the summary table, the methods footer) is
derived from the ACTUAL data + config, so it works unchanged for any target
protein. Charts use Chart.js from a CDN; the summary table is the always-visible
fallback when offline.
"""

from __future__ import annotations

import html
import json
import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple

# stable per-annotation colours; unknown annotations cycle through the tail.
_COLORS = {
    "colabfold": "#378ADD", "foldseek": "#1D9E75", "mmseqs": "#7F77DD",
    "blastp": "#D4537E", "jackhmmer": "#BA7517", "synthetic": "#888780",
    "synthetic_structure": "#B4B2A9",
}
_CYCLE = ["#534AB7", "#0F6E56", "#993C1D", "#185FA5", "#854F0B", "#A32D2D"]


def _color_for(ann: str, i: int) -> str:
    return _COLORS.get(ann, _CYCLE[i % len(_CYCLE)])


def _nice_step(span: float, target_bins: int) -> float:
    """A round-ish bin width (...0.01,0.02,0.05,0.1...) for ~target_bins bins."""
    if span <= 0:
        return 0.05
    raw = span / max(1, target_bins)
    import math
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def _identity_histogram(homologs, groups) -> Tuple[List[str], Dict[str, List[int]]]:
    ids = [h.identity for h in homologs if h.identity is not None]
    if not ids:
        return [], {g: [] for g in groups}
    lo, hi = min(ids), max(ids)
    step = _nice_step(hi - lo, 14)
    start = step * (int(lo / step))
    nb = max(1, int((hi - start) / step) + 1)
    edges = [round(start + step * i, 4) for i in range(nb + 1)]
    labels = [f"{edges[i]:.2f}" for i in range(nb)]
    hist = {g: [0] * nb for g in groups}
    for h in homologs:
        if h.identity is None:
            continue
        k = min(nb - 1, max(0, int((h.identity - start) / step)))
        g = _group_of(h)
        if g in hist:
            hist[g][k] += 1
    return labels, hist


def _cluster_buckets(homologs) -> Tuple[List[str], List[int], int, int]:
    sizes = list(Counter(h.cluster_id for h in homologs).values())
    if not sizes:
        return [], [], 0, 0
    largest = max(sizes)
    n_singletons = sum(1 for s in sizes if s == 1)
    ranges = [(1, 1), (2, 2), (3, 5), (6, 15), (16, 50), (51, 200),
              (201, 1000), (1001, 10 ** 12)]
    labels, counts = [], []
    for a, b in ranges:
        c = sum(1 for s in sizes if a <= s <= b)
        if not c:
            continue
        labels.append(str(a) if a == b else (f"{a}+" if b >= 10 ** 12 else f"{a}-{b}"))
        counts.append(c)
    return labels, counts, largest, n_singletons


def _group_of(h) -> str:
    return getattr(h, "annotation", "") or getattr(h, "source", "sequence")


def _chart_data(homologs, msa_depth) -> dict:
    groups = sorted({_group_of(h) for h in homologs})
    labels, idh = _identity_histogram(homologs, groups)
    cl_labels, cl_counts, largest, n_singletons = _cluster_buckets(homologs)
    n_clusters = len(set(h.cluster_id for h in homologs))
    return {
        "n": len(homologs),
        "msa_depth": msa_depth,
        "groups": groups,
        "colors": {g: _color_for(g, i) for i, g in enumerate(groups)},
        "counts": {g: sum(1 for h in homologs if _group_of(h) == g) for g in groups},
        "id_labels": labels,
        "id_hist": idh,
        "cluster_labels": cl_labels,
        "cluster_counts": cl_counts,
        "n_clusters": n_clusters,
        "largest_cluster": largest,
        "n_singletons": n_singletons,
    }


def _summary_rows(homologs, data) -> List[List[str]]:
    rows = []
    for g in data["groups"]:
        ids = [h.identity for h in homologs
               if _group_of(h) == g and h.identity is not None]
        n = data["counts"][g]
        pct = 100.0 * n / max(1, data["n"])
        if ids:
            mn, md, mx = min(ids), statistics.median(ids), max(ids)
            rng = f"{mn:.2f} / {md:.2f} / {mx:.2f}"
        else:
            rng = "-"
        rows.append([g, f"{n:,}", f"{pct:.1f}%", rng])
    return rows


def build_homolog_report_html(
    *, target_id: str, target_len: int, homologs: Sequence,
    msa_depth, conditions: List[Tuple[str, str]], generated: str,
    provenance: str = "",
) -> str:
    data = _chart_data(list(homologs), msa_depth)
    g_esc = html.escape
    subtitle = provenance or f"generated {g_esc(generated)}"

    cards = [
        ("homologs", f"{data['n']:,}", "#111"),
        ("sequence (diversity) clusters", f"{data['n_clusters']:,}", "#111"),
        ("largest cluster", f"{data['largest_cluster']:,}", "#111"),
        ("MSA depth", f"{data['msa_depth']:,}" if data['msa_depth'] else "-", "#111"),
    ]
    for g in data["groups"]:
        cards.insert(1, (g_esc(g), f"{data['counts'][g]:,}", data["colors"][g]))
    cards_html = "".join(
        f'<div class="card"><div class="lab">{lab}</div>'
        f'<div class="num" style="color:{col}">{val}</div></div>'
        for lab, val, col in cards
    )

    srows = "".join(
        "<tr>" + "".join(f"<td>{g_esc(c)}</td>" for c in row) + "</tr>"
        for row in _summary_rows(list(homologs), data)
    )
    legend = "".join(
        f'<span class="lg"><span class="sw" style="background:{data["colors"][g]}">'
        f'</span>{g_esc(g)} ({data["counts"][g]:,})</span>'
        for g in data["groups"]
    )
    cond_rows = "".join(
        f"<tr><td class=ck>{g_esc(k)}</td><td>{g_esc(str(v))}</td></tr>"
        for k, v in conditions
    )
    blob = json.dumps({k: data[k] for k in
                       ("groups", "colors", "id_labels", "id_hist",
                        "cluster_labels", "cluster_counts")})

    nclu, nsing = data["n_clusters"], data.get("n_singletons", 0)
    spct = round(100 * nsing / nclu) if nclu else 0
    caveat = (
        f"{nsing:,} of {nclu:,} sequence-diversity clusters are SINGLETONS "
        f"({spct}%) while "
        f"the largest holds {data['largest_cluster']:,}. This is a singleton-"
        "DOMINATED distribution (one big bucket + a long singleton tail), NOT a "
        "balanced family decomposition — so 'n clusters' over-states the effective "
        "sequence diversity, and s06b takes the LARGEST clusters first. These are "
        "sequence-identity (diversity) clusters, NOT validated biological "
        "subfamilies. "
        "Pool note: this homolog count = deduplicated RETRIEVED sequences across "
        "the 5 tracks; the s03 “MSA depth” additionally includes the "
        "ColabFold remote-MSA base rows, so the two counts are not the same set.")

    return _TEMPLATE.format(
        title=g_esc(f"{target_id} — integrated homolog analysis"),
        target=g_esc(target_id), tlen=target_len, provsub=subtitle,
        cards=cards_html, summary_rows=srows, legend=legend,
        cond_rows=cond_rows, blob=blob, cluster_caveat=g_esc(caveat),
    )


def write_homolog_report(path, **kwargs) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    from evoliez.io._report_kit import assert_html_clean
    p.write_text(assert_html_clean(build_homolog_report_html(**kwargs), title="homolog"),
                 encoding="utf-8")


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#fff;--fg:#1c1c1a;--mut:#5f5e5a;--line:#e7e5df;--surf:#f7f6f2}}
@media(prefers-color-scheme:dark){{:root{{--bg:#1a1a18;--fg:#ece9e3;--mut:#a8a69e;--line:#33322e;--surf:#232220}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
.wrap{{max-width:920px;margin:0 auto;padding:2rem 1.25rem 3rem}}
h1{{font-size:22px;font-weight:500;margin:0 0 2px}}
h2{{font-size:18px;font-weight:500;margin:2rem 0 .6rem}}
.sub{{color:var(--mut);font-size:14px;margin:0 0 1.5rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:0 0 1rem}}
.card{{background:var(--surf);border-radius:8px;padding:.85rem 1rem}}
.lab{{font-size:13px;color:var(--mut)}}
.num{{font-size:24px;font-weight:500;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:.5px solid var(--line)}}
th{{color:var(--mut);font-weight:500;font-size:13px}}
td.ck{{color:var(--mut);width:38%}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--mut);margin:.2rem 0 .6rem}}
.lg{{display:flex;align-items:center;gap:5px}}
.sw{{width:11px;height:11px;border-radius:2px;display:inline-block}}
.chartbox{{position:relative;width:100%;height:300px}}
.note{{font-size:13px;color:var(--mut)}}
footer{{margin-top:2.5rem;border-top:.5px solid var(--line);padding-top:1rem}}
</style></head>
<body><div class="wrap">
<h1>{title}</h1>
<p class="sub">target {target} · {tlen} aa · {provsub}</p>

<div class="cards">{cards}</div>

<h2>Summary by source</h2>
<table><thead><tr><th>source</th><th>homologs</th><th>share</th>
<th>identity min / median / max</th></tr></thead><tbody>{summary_rows}</tbody></table>

<h2>Identity to target</h2>
<div class="legend">{legend}</div>
<div class="chartbox"><canvas id="idChart" role="img"
 aria-label="Identity distribution of homologs by source (log count)."></canvas></div>
<p class="note">log count; bins auto-scaled to the observed identity range.</p>

<h2>Sequence (diversity) cluster sizes</h2>
<div class="chartbox" style="height:260px"><canvas id="clChart" role="img"
 aria-label="Distribution of sequence (diversity) cluster sizes (log count)."></canvas></div>
<p class="note">how many clusters contain N members (log count). A single very
large bucket means that source was not sub-clustered.</p>
<div class="note" style="border-left:3px solid #BA7517;padding:.5rem .75rem;background:var(--surf);border-radius:4px;margin:.6rem 0">{cluster_caveat}</div>

<footer>
<h2>Analysis conditions</h2>
<table><tbody>{cond_rows}</tbody></table>
</footer>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
var R={blob};
function mk(id,labels,datasets,xlab){{
  var el=document.getElementById(id);if(!el||!window.Chart)return;
  new Chart(el,{{type:'bar',data:{{labels:labels,datasets:datasets}},
   options:{{responsive:true,maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}}},
    scales:{{y:{{type:'logarithmic',title:{{display:true,text:'count'}}}},
     x:{{ticks:{{autoSkip:false,maxRotation:45}},title:{{display:true,text:xlab}}}}}}}}}});
}}
mk('idChart',R.id_labels,R.groups.map(function(g){{
  return {{label:g,data:R.id_hist[g],backgroundColor:R.colors[g]}};}}),'identity to target');
mk('clChart',R.cluster_labels,[{{label:'clusters',data:R.cluster_counts,
  backgroundColor:'#888780'}}],'cluster size (members)');
</script></body></html>
"""
