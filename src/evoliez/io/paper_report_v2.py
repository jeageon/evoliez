"""v2 evidence-class paper report (ROADMAP_V2 Phase E output).

Renders the paper-grade view of a v2 run from disk — NO DB, NO stage re-run. The final claim
is an EVIDENCE CLASS, not a scalar rank: candidates grouped by gate-stack verdict, the
paper-grade set (anchored validation + reference_like pose + a verdict), the ML functional-
state re-evaluation (ml enriches binding-validity, not catalysis), and the MSA QC (depth is
not diversity). Reads:
  reports/provenance/{md_candidates,evidence_classes}.json  +  _state.json meta.
Self-contained HTML via _report_kit (the s08/s09 house style).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from evoliez.io import _report_kit as kit

_VERDICT_COLOR = {
    "confirmed_computational": "#16a34a",
    "candidate_improved": "#2563eb",
    "candidate_no_gain": "#6b7280",
    "alternative_pose": "#d97706",
    "rejected": "#dc2626",
}


def _f(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def build_paper_report_v2(run_dir) -> str:
    RD = Path(run_dir)
    PROV = RD / "reports" / "provenance"
    mc = json.loads((PROV / "md_candidates.json").read_text())
    recs = mc.get("candidates", [])
    ev = {}
    if (PROV / "evidence_classes.json").exists():
        ev = json.loads((PROV / "evidence_classes.json").read_text())
    meta = {}
    if (RD / "_state.json").exists():
        meta = json.loads((RD / "_state.json").read_text()).get("meta") or {}
    qc = meta.get("msa_qc", {}) or {}
    mle = meta.get("ml_functional_eval", {}) or {}
    counts = ev.get("counts", {}) or {}
    paper = set(ev.get("paper_grade", []) or [])
    pareto = set(ev.get("pareto", []) or [])

    # --- summary cards ---
    cards = [
        kit.card("Candidates", str(len(recs)), "validated in MD"),
        kit.card("Paper-grade", str(len(paper)),
                 "wt_anchored + reference_like + verdict",
                 "#16a34a" if paper else "#dc2626"),
        kit.card("Pareto front", str(len(pareto)), "ΔNAC↑ / instability↓ / ΔΔG↓"),
        kit.card("MSA Neff", str(qc.get("neff_clusters", "—")),
                 f"depth {qc.get('depth', '—')} · real {qc.get('real_fraction', '—')}"),
    ]

    # --- evidence classes ---
    ev_pills = "".join(
        kit.pill(f"{v}: {n}", _VERDICT_COLOR.get(v, "#6b7280"))
        for v, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    ev_section = (
        "<h2>Evidence classes (final claim = a class, not a rank)</h2>"
        f"<p>{ev_pills or 'none'}</p>"
        "<p class='muted'>A candidate is <b>paper-grade</b> only if it was validated "
        "WT-anchored, kept a reference_like design pose, and carries a gate-stack verdict — "
        "the per-mutant Boltz-pose path can never qualify (it is a pose-search hypothesis).</p>")

    # --- ML functional eval ---
    ml_section = (
        "<h2>ML re-evaluation (functional-state preservation)</h2>"
        f"<p>AUC(ml→functional) = <b>{mle.get('auc_ml_to_functional', '—')}</b> "
        f"vs AUC(ml→MD-pass) = <b>{mle.get('auc_ml_to_mdpass', '—')}</b>. "
        f"Low-ML control lane: <b>{mle.get('functional_in_control', '—')}</b> functional "
        f"winner(s) the ML cut would have dropped "
        f"({'⚠ ML false negatives present' if mle.get('ml_false_negative') else 'none'}).</p>"
        "<p class='muted'>ML enriches binding-validity, not catalysis — so it is a prior, "
        "never a hard filter; the multi-lane union + this control lane monitor its blind spots.</p>")

    # --- candidate table ---
    rows = []
    for r in recs:
        cid = r.get("candidate_id")
        gs = (r.get("gate_stack") or {}).get("verdict", "—")
        pg = ((r.get("pose_gate") or {}).get("design_ligand") or {}).get("status", "—")
        flags = []
        if cid in paper:
            flags.append(kit.pill("paper-grade", "#16a34a"))
        if cid in pareto:
            flags.append(kit.pill("pareto", "#2563eb"))
        rows.append(
            f"<tr><td>{kit.esc(r.get('mutation_string', cid))}</td>"
            f"<td>{kit.esc(r.get('validation_structure', '—'))}</td>"
            f"<td>{kit.esc(pg)}</td>"
            f"<td>{kit.pill(gs, _VERDICT_COLOR.get(gs, '#6b7280'))}</td>"
            f"<td>{_f(r.get('nac_occupancy'))}</td>"
            f"<td>{_f(r.get('nac_delta_vs_wt'))}</td>"
            f"<td>{_f(r.get('md_lite_score'))}</td>"
            f"<td>{''.join(flags)}</td></tr>")
    table = (
        "<h2>Per-candidate evidence</h2>"
        "<table><thead><tr><th>mutation</th><th>validation</th><th>design pose</th>"
        "<th>verdict</th><th>NAC</th><th>ΔNAC</th><th>md_lite</th><th>flags</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    body = (
        f"<div class='cards'>{''.join(cards)}</div>"
        + ev_section + ml_section + table
        + kit.citations([
            "Functional-state-anchored validation: mutants built from the WT reference "
            "complex (reference cofactor/substrate pose + only the mutation), not a fresh "
            "per-mutant pose search — separates mutation effect from pose-search noise.",
        ]))
    return kit.page("EvoLiEZ v2 — evidence-class report", body)


def write_paper_report_v2(run_dir, out_name: str = "paper_report_v2.html") -> Path:
    RD = Path(run_dir)
    out = RD / "reports" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_paper_report_v2(RD))
    return out
