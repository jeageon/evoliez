"""Funnel-provenance reporting for the paper (design-funnel attrition + audit).

The pipeline narrows a large generated candidate pool down to a small ranked
shortlist through a sequence of increasingly expensive filters:

    generate (s07) -> fast rerank (s08) -> real mutant Boltz (s08b)
        -> redock / stability validate (s09) -> short MD (s10) -> final rank (s11)

The per-stage meta COUNTS are persisted as the run proceeds (``ctx.persist_meta``),
but a methods section also needs a DURABLE, per-candidate audit trail: which
generator proposed each variant, how far down the funnel it survived, and WHAT
KIND of evidence backs its final score (a real per-mutant Boltz re-prediction, a
real redock, a stability measurement that failed, an actual MD run, or just the
cheap proxy). This module turns the final ranked candidates + the persisted meta
counts into three small, generic tables (JSON + CSV, plus a compact HTML view):

  (a) ATTRITION FUNNEL    - count surviving at each funnel stage;
  (b) PER-GENERATOR survival - #generated vs #in the final ranked top-N;
  (c) PER-CANDIDATE evidence - rank, generator, survival depth, evidence class.

Everything is read from the candidate objects + the meta dict; nothing about the
target / ligand / chemistry is hard-coded, so it is identical for any protein.
"""

from __future__ import annotations

import csv
import html as _html
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from evoliez.types import Candidate

# Ordered funnel stages: (key in the output, human label, meta key the producing
# stage persisted). The meta key is None for stages whose survivor count we read
# straight from the candidate objects rather than from a persisted counter.
_FUNNEL: List[tuple] = [
    ("generated", "candidates generated (s07)", "n_candidates_generated"),
    ("reranked", "kept after fast rerank (s08)", "n_after_rerank"),
    ("boltz_reevaluated", "real mutant Boltz re-eval (s08b)",
     "n_mutant_boltz_evaluated"),
    ("validated", "passed non-MD validation (s09)", "n_after_nonmd"),
    ("selected_for_md", "advanced to MD (s09)", "n_for_md"),
    ("md_ran", "MD actually executed (s10)", "n_md_real_ran"),
    ("md_passed", "MD passed (s10)", "n_md_passed"),
    ("ranked", "final ranked (s11)", "n_ranked"),
]

# Per-candidate columns persisted in the evidence table (machine-readable record).
_EVIDENCE_COLUMNS = [
    "rank", "candidate_id", "mutations", "generator", "n_mutations",
    "final_score", "ml_score", "evidence_class", "survived_to",
    "boltz_delta_source", "n_redock_methods", "stability_unavailable",
    "md_ran",
]


def evidence_class(cand: Candidate, md_ids: set) -> str:
    """Strongest line of evidence backing this candidate's final score.

    Ordered most-informative first: a REAL per-mutant Boltz re-prediction (s08b)
    > a REAL redock against the reference pose (s09) > an MD run that actually
    executed (s10) > a stability measurement that FAILED (surfaced, not hidden) >
    the cheap proxy that every candidate at least carries. The order is a
    deliberate ranking of how much a result can be trusted, not a category set.
    """
    if cand.details.get("boltz_delta_source") == "real":
        return "real_boltz"
    if cand.details.get("redock"):
        return "real_docking"
    if cand.candidate_id in md_ids:
        return "md_ran"
    if cand.details.get("stability_unavailable"):
        return "stability_unavailable"
    return "proxy"


def _survived_to(cand: Candidate, md_ids: set) -> str:
    """Deepest funnel stage this candidate reached, inferred from the evidence it
    carries (the per-candidate analogue of the attrition counts). A candidate is
    always at least 'ranked' if it made the final table."""
    if cand.candidate_id in md_ids:
        return "md"
    if cand.details.get("redock") or "ddg_fold" in cand.scores:
        return "validated"
    if cand.details.get("boltz_delta_source") in ("real", "mock", "dry-run"):
        return "boltz_reevaluated"
    if "ml_score" in cand.scores:
        return "reranked"
    return "generated"


def build_funnel_report(
    *,
    ranked: Sequence[Candidate],
    meta: Callable[[str, Any], Any],
    md_candidate_ids: Optional[Sequence[str]] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the three funnel tables from the final ranked candidates and the
    persisted meta counts.

    ``meta`` is ``ctx.meta`` (key, default) -> value. ``md_candidate_ids`` are the
    ids of candidates handed to MD (``ctx.get('md_candidates')``); ``top_n``
    scopes the per-generator "survived to the shortlist" count (defaults to all
    ranked). Returns a JSON-safe dict with ``attrition``, ``per_generator``,
    ``per_candidate`` and a small ``summary``.
    """
    md_ids = set(md_candidate_ids or [])
    ranked = list(ranked)
    n_ranked = len(ranked)
    top_n = n_ranked if top_n is None else min(top_n, n_ranked)
    shortlist_ids = {c.candidate_id for c in ranked[:top_n]}

    # (a) attrition funnel ------------------------------------------------- #
    # Pull each stage's survivor count from its persisted meta key; 'ranked'
    # comes from the live candidate list so it is correct even if s11's own
    # n_ranked is written after this report. Stages that did not run leave their
    # meta key unset -> reported as None (distinct from a real 0).
    attrition: List[Dict[str, Any]] = []
    n_generated = meta("n_candidates_generated", None)
    for key, label, meta_key in _FUNNEL:
        if key == "ranked":
            count: Optional[int] = n_ranked
        else:
            count = meta(meta_key, None)
        frac = (round(count / n_generated, 4)
                if count is not None and n_generated else None)
        attrition.append({
            "stage": key, "label": label, "count": count,
            "fraction_of_generated": frac,
        })

    # (b) per-generator survival ------------------------------------------ #
    # #generated per generator comes from the durable generated-provenance table
    # written by s07 (it has the FULL pre-attrition pool); the ranked objects only
    # carry the survivors. We fall back to ranked-only counts if that table is
    # absent. Filled in by the caller via ``generated_by_generator`` when known.
    gens = sorted({c.generator for c in ranked})
    in_ranked: Dict[str, int] = {g: 0 for g in gens}
    in_shortlist: Dict[str, int] = {g: 0 for g in gens}
    for c in ranked:
        in_ranked[c.generator] += 1
        if c.candidate_id in shortlist_ids:
            in_shortlist[c.generator] += 1
    per_generator = [
        {
            "generator": g,
            "n_in_ranked": in_ranked[g],
            "n_in_shortlist": in_shortlist[g],
        }
        for g in gens
    ]

    # (c) per-candidate evidence ------------------------------------------ #
    per_candidate: List[Dict[str, Any]] = []
    klass_counts: Dict[str, int] = {}
    for i, c in enumerate(ranked, 1):
        klass = evidence_class(c, md_ids)
        klass_counts[klass] = klass_counts.get(klass, 0) + 1
        redock = c.details.get("redock") or {}
        per_candidate.append({
            "rank": c.details.get("rank", i),
            "candidate_id": c.candidate_id,
            "mutations": c.mutation_str,
            "generator": c.generator,
            "n_mutations": len(c.mutations),
            "final_score": c.scores.get("final_score"),
            "ml_score": c.scores.get("ml_score"),
            "evidence_class": klass,
            "survived_to": _survived_to(c, md_ids),
            "boltz_delta_source": c.details.get("boltz_delta_source"),
            "n_redock_methods": len(redock) if isinstance(redock, dict) else 0,
            "stability_unavailable": bool(
                c.details.get("stability_unavailable", False)),
            "md_ran": c.candidate_id in md_ids,
        })

    return {
        "attrition": attrition,
        "per_generator": per_generator,
        "per_candidate": per_candidate,
        "summary": {
            "n_generated": n_generated,
            "n_ranked": n_ranked,
            "n_shortlist": top_n,
            "n_md": len(md_ids),
            "evidence_class_counts": klass_counts,
        },
    }


def attach_generated_counts(
    report: Dict[str, Any], generated_provenance_path: Path
) -> Dict[str, Any]:
    """Enrich the per-generator table with the TRUE #generated per generator read
    from s07's durable generated-provenance JSON (the full pre-attrition pool),
    plus a survival fraction. A no-op (leaves ``n_generated`` None) if that file
    is missing so the report still builds on a partial run."""
    counts: Dict[str, int] = {}
    try:
        rows = json.loads(Path(generated_provenance_path).read_text())
        if isinstance(rows, dict):
            rows = rows.get("candidates", [])
        for r in rows:
            g = r.get("generator")
            if g:
                counts[g] = counts.get(g, 0) + 1
    except (OSError, json.JSONDecodeError, AttributeError):
        counts = {}
    for row in report["per_generator"]:
        g = row["generator"]
        n_gen = counts.get(g)
        row["n_generated"] = n_gen
        row["survival_fraction"] = (
            round(row["n_in_ranked"] / n_gen, 4)
            if n_gen else None
        )
    return report


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def write_funnel_report(out_dir: Path, report: Dict[str, Any],
                        *, write_html: bool = True) -> List[Path]:
    """Persist the funnel report: one JSON (the full structure) + three CSVs
    (attrition, per-generator, per-candidate) + an optional compact HTML view.
    Returns the paths written. Generic + small; clearly named under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    pj = out_dir / "funnel_provenance.json"
    pj.write_text(json.dumps(report, indent=2, default=str))
    written.append(pj)

    pa = out_dir / "attrition_funnel.csv"
    with pa.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "label", "count", "fraction_of_generated"])
        for r in report["attrition"]:
            w.writerow([r["stage"], r["label"], _blank(r["count"]),
                        _blank(r["fraction_of_generated"])])
    written.append(pa)

    pg = out_dir / "generator_survival.csv"
    with pg.open("w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["generator", "n_generated", "n_in_ranked", "n_in_shortlist",
                "survival_fraction"]
        w.writerow(cols)
        for r in report["per_generator"]:
            w.writerow([_blank(r.get(c)) for c in cols])
    written.append(pg)

    pc = out_dir / "candidate_provenance.csv"
    with pc.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_EVIDENCE_COLUMNS)
        for r in report["per_candidate"]:
            w.writerow([_blank(r.get(c)) for c in _EVIDENCE_COLUMNS])
    written.append(pc)

    if write_html:
        ph = out_dir / "funnel_provenance.html"
        ph.write_text(_render_html(report), encoding="utf-8")
        written.append(ph)
    return written


def _blank(v: Any) -> Any:
    """Missing -> empty cell (paper-grade: an unrun stage is blank, not 0)."""
    return "" if v is None else v


# --------------------------------------------------------------------------- #
# compact HTML view (same minimal CSS vocabulary as the other io/*.py reports)
# --------------------------------------------------------------------------- #
def _render_html(report: Dict[str, Any]) -> str:
    g = _html.escape
    summ = report["summary"]

    def card(lab, val, sub):
        return (f'<div class="card"><div class="lab">{g(lab)}</div>'
                f'<div class="num">{g(str(val))}</div>'
                f'<div class="sub2">{g(sub)}</div></div>')

    cards = "".join([
        card("generated", _blank(summ["n_generated"]) or "—", "s07 pool"),
        card("ranked", summ["n_ranked"], "final table"),
        card("to MD", summ["n_md"], "s10 input"),
        card("real Boltz",
             summ["evidence_class_counts"].get("real_boltz", 0),
             "s08b re-eval"),
    ])

    # attrition: count + a proportional bar (width = fraction of generated).
    a_rows = ""
    for r in report["attrition"]:
        frac = r["fraction_of_generated"]
        pct = f"{100 * frac:.0f}%" if frac is not None else ""
        width = max(1.0, 100 * frac) if frac is not None else 0.0
        a_rows += (
            f"<tr><td>{g(r['label'])}</td>"
            f"<td class=ck>{g(str(_blank(r['count'])))}</td>"
            f"<td><div class='bar' style='width:{width:.1f}%'></div>"
            f"<span class=pct>{g(pct)}</span></td></tr>"
        )

    has_gen = any(r.get("n_generated") is not None
                  for r in report["per_generator"])
    if has_gen:
        gen_head = ("<tr><th>generator</th><th>generated</th><th>in ranked</th>"
                    "<th>in shortlist</th><th>survival</th></tr>")
    else:
        gen_head = ("<tr><th>generator</th><th>in ranked</th>"
                    "<th>in shortlist</th></tr>")
    g_rows = ""
    for r in report["per_generator"]:
        sf = r.get("survival_fraction")
        if has_gen:
            g_rows += (
                f"<tr><td class=ck>{g(r['generator'])}</td>"
                f"<td>{g(str(_blank(r.get('n_generated'))))}</td>"
                f"<td>{r['n_in_ranked']}</td>"
                f"<td>{r['n_in_shortlist']}</td>"
                f"<td>{g('' if sf is None else f'{100*sf:.0f}%')}</td></tr>"
            )
        else:
            g_rows += (
                f"<tr><td class=ck>{g(r['generator'])}</td>"
                f"<td>{r['n_in_ranked']}</td>"
                f"<td>{r['n_in_shortlist']}</td></tr>"
            )

    # evidence-class legend (counts across the ranked table)
    kc = summ["evidence_class_counts"]
    legend = " · ".join(f"{g(k)}: {v}" for k, v in sorted(kc.items())) or "—"

    # per-candidate table, capped for the HTML view (CSV has the full set)
    c_rows = ""
    for r in report["per_candidate"][:200]:
        c_rows += (
            f"<tr><td class=ck>{g(str(r['rank']))}</td>"
            f"<td>{g(r['mutations'])}</td>"
            f"<td>{g(r['generator'])}</td>"
            f"<td>{g(str(_blank(r['final_score'])))}</td>"
            f"<td><span class='tag t-{g(r['evidence_class'])}'>"
            f"{g(r['evidence_class'])}</span></td>"
            f"<td class=ck>{g(r['survived_to'])}</td></tr>"
        )

    return (_FUNNEL_TEMPLATE
            .replace("%%CARDS%%", cards)
            .replace("%%ATTRITION%%", a_rows)
            .replace("%%GENHEAD%%", gen_head)
            .replace("%%GENROWS%%", g_rows)
            .replace("%%LEGEND%%", legend)
            .replace("%%CANDROWS%%", c_rows)
            .replace("%%NSHORT%%", str(summ["n_shortlist"])))


_FUNNEL_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvoLiEZ — design-funnel provenance</title>
<style>
:root{--bg:#fff;--fg:#1c1c1a;--mut:#5f5e5a;--line:#e7e5df;--surf:#f7f6f2;--bar:#7F77DD}
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
th,td{text-align:left;padding:7px 10px;border-bottom:.5px solid var(--line);vertical-align:middle}
th{color:var(--mut);font-weight:500;font-size:13px}
td.ck{color:var(--mut)}
.bar{display:inline-block;height:10px;background:var(--bar);border-radius:3px;vertical-align:middle;min-width:2px}
.pct{font-size:12px;color:var(--mut);margin-left:6px}
.tag{font-size:11.5px;padding:1px 7px;border-radius:10px;background:var(--surf);color:var(--mut)}
.t-real_boltz{background:#2E7D32;color:#fff}.t-real_docking{background:#1565C0;color:#fff}
.t-md_ran{background:#6A1B9A;color:#fff}.t-stability_unavailable{background:#BA7517;color:#fff}
footer{margin-top:2.5rem;border-top:.5px solid var(--line);padding-top:1rem}
footer p{font-size:13.5px}.cite{font-size:12px;color:var(--mut);line-height:1.7}
</style></head>
<body><div class="wrap">
<h1>Design-funnel provenance</h1>
<p class="sub">per-candidate audit trail + attrition through the generate → rerank
→ Boltz → validate → MD → rank funnel</p>

<div class="cards">%%CARDS%%</div>

<h2>Attrition funnel</h2>
<table><thead><tr><th>stage</th><th>surviving</th><th>fraction of generated</th></tr></thead>
<tbody>%%ATTRITION%%</tbody></table>
<p class="note">Counts are the durable per-stage meta counters persisted as the
run proceeds. A blank cell means that stage did not run in this configuration —
which is distinct from a real count of 0.</p>

<h2>Per-generator survival</h2>
<table><thead>%%GENHEAD%%</thead><tbody>%%GENROWS%%</tbody></table>
<p class="note">How each generation strategy fared through the funnel: how many
candidates it produced (from the s07 generated-provenance table) versus how many
reached the final ranked table and the top-%%NSHORT%% shortlist.</p>

<h2>Per-candidate evidence</h2>
<p class="note">Evidence class across the ranked table — %%LEGEND%%</p>
<table><thead><tr><th>rank</th><th>mutations</th><th>generator</th>
<th>final score</th><th>evidence</th><th>survived to</th></tr></thead>
<tbody>%%CANDROWS%%</tbody></table>
<p class="note"><b>real_boltz</b> = score backed by a real per-mutant Boltz
re-prediction (s08b); <b>real_docking</b> = backed by a real redock vs the
reference pose (s09); <b>md_ran</b> = an MD simulation actually executed (s10);
<b>stability_unavailable</b> = the ΔΔG tool produced no parseable value (surfaced,
never silently treated as stable); <b>proxy</b> = the cheap proxy every candidate
carries. The full per-candidate table is in <code>candidate_provenance.csv</code>.</p>

<footer>
<h2>Methods</h2>
<p><b>Funnel provenance.</b> A large generated candidate pool is narrowed by
increasingly expensive filters (fast ML rerank → real per-mutant Boltz
re-prediction → redocking + stability validation → short MD), each emitting a
durable survivor count. This table reconstructs the attrition and tags every
ranked candidate with the strongest line of computational evidence behind its
score, so the shortlist can be read without over-trusting proxy-only entries.</p>
<p class="cite">Computational predictions are testable hypotheses, not guarantees
of activity.</p>
</footer>
</div></body></html>
"""
