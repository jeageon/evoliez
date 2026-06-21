"""Self-contained HTML report for EvoLiEZ stage s07 (mutation generation).

Presents the multi-strategy mutant LIBRARY the way an enzyme-design paper would:
the generator break-down (how many candidates each strategy contributed —
contact-ensemble chemistry rules, gated MSA sampling, ligand-aware LigandMPNN,
FuncLib-style multipoint), the budget TIERS the library was split into
(single / multipoint / risky), a per-position DESIGN-SPACE map over the
designable residues (contact frequency / nearest-ligand distance / MSA
permissiveness / how many candidates touch each, and which positions are
PROTECTED — catalytic / fixed / switch — vs designable), the per-mutation
PROVENANCE table (generator + the structural distances + MSA conservation /
occupancy + risk flag + forbidden-check, the paper's reproducibility appendix),
and an interactive 3D view of the WT complex with the cofactor ligand, the
protected catalytic core in one colour and the s07 designable region in another,
so the reader sees the design region vs the protected core in the pocket.

Everything is derived from the ACTUAL on-disk artifacts the stage writes to
``<run>/reports/provenance/`` (``generated_candidates{,_single,_multipoint,
_risky}.json``) plus the design-context inputs (designable / catalytic / fixed
positions, the contact list, the per-position MSA features, the WT complex PDB),
so it works UNCHANGED for any protein / ligand — NOTHING here is specific to one
target (no FDH / NADP hardcoding; ligand ids and positions are read from the
run). This top section is the viz-agnostic DATA layer; the render layer
(template + Chart.js + 3Dmol, reused verbatim from the sibling reports) lives
below it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

# The four canonical s07 generators, in display order, each with a fixed colour
# so the break-down chart / legend / tier tables read the same for any target.
_GEN_ORDER = ["chemistry_rules", "msa_sampler", "ligandmpnn", "multipoint"]
_GEN_COLORS = {
    "chemistry_rules": "#BA7517",   # contact-ensemble chemistry rules
    "msa_sampler": "#378ADD",       # gated MSA family sampling
    "ligandmpnn": "#1D9E75",        # ligand-aware deep design
    "multipoint": "#9b59b6",        # FuncLib-style active-site library
}
_GEN_LABELS = {
    "chemistry_rules": "chemistry rules",
    "msa_sampler": "MSA sampler",
    "ligandmpnn": "LigandMPNN",
    "multipoint": "multipoint (FuncLib)",
}
# Budget tiers, in display order, each with a fixed colour (matches the on-disk
# generated_candidates_{single,multipoint,risky} files).
_TIER_ORDER = ["single", "multipoint", "risky"]
_TIER_COLORS = {"single": "#1D9E75", "multipoint": "#9b59b6", "risky": "#C0392B"}
_TIER_SUB = {
    "single": "one substitution — cheapest s08–s10 tier",
    "multipoint": "≥ 2 substitutions — active-site epistasis",
    "risky": "risk-flagged — exploratory, QC first",
}
# Protected-position categories, each with a fixed colour. These positions are
# NEVER designed by s07; they are shown distinct from the designable region.
_PROT_COLORS = {"catalytic": "#C0392B", "fixed": "#993C1D",
                "designable": "#1D9E75"}


def _num(v) -> Optional[float]:
    """Best-effort float (``None`` for blanks / non-numeric — a CSV/JSON cell may
    be '', a ';'-joined per-position list, or a real number)."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _g(c, k, default=None):
    """Read ``k`` from a dict OR an object (works for both the on-disk JSON rows
    and in-memory Candidate / feature objects)."""
    if isinstance(c, dict):
        return c.get(k, default)
    return getattr(c, k, default)


def _read_pdb_text(wt_pdb_path) -> Optional[str]:
    """Read the WT complex PDB for the 3D viewer (``None`` if unavailable).
    Accepts a path OR already-loaded text; never raises (the report degrades
    gracefully without the viewer)."""
    if wt_pdb_path is None:
        return None
    try:
        s = str(wt_pdb_path)
        if "\n" in s and ("ATOM" in s or "HETATM" in s):   # already PDB text
            return s
        if os.path.exists(s):
            with open(s, "r") as fh:
                return fh.read()
    except OSError:
        return None
    return None


def find_wt_complex_pdb(complexes_dir) -> Optional[str]:
    """Locate the WT complex model_0 PDB under a run's ``complexes/`` tree
    (generic glob; prefers the 'wt' one). Mirrors the s04/s06b locator so the
    standalone regenerator finds the same structure. Returns a path or None."""
    import glob

    pats = [
        os.path.join(str(complexes_dir), "boltz", "boltz_results_*",
                     "predictions", "*", "*_model_0.pdb"),
        os.path.join(str(complexes_dir), "**", "*_model_0.pdb"),
    ]
    found: List[str] = []
    for p in pats:
        found.extend(glob.glob(p, recursive=True))
        if found:
            break
    if not found:
        return None
    wt = [f for f in found if "wt" in os.path.basename(f).lower()]
    return sorted(wt or found)[0]


# --------------------------------------------------------------------------- #
# DATA layer
# --------------------------------------------------------------------------- #
def _generator_breakdown(rows: Sequence[dict]) -> dict:
    """Candidate counts per generator (canonical order first, then any extra a
    run produced). Returns {generators, labels, colors, counts, total}."""
    counts: Dict[str, int] = {}
    for r in rows:
        gen = str(_g(r, "generator", "—") or "—")
        counts[gen] = counts.get(gen, 0) + 1
    gens = [g for g in _GEN_ORDER if g in counts]
    gens += sorted(g for g in counts if g not in gens)
    return {
        "generators": gens,
        "labels": [_GEN_LABELS.get(g, g) for g in gens],
        "colors": [_GEN_COLORS.get(g, "#888780") for g in gens],
        "counts": [counts[g] for g in gens],
        "total": sum(counts.values()),
    }


def _tier_breakdown(tier_rows: Dict[str, List[dict]]) -> dict:
    """Candidate counts per budget tier (single / multipoint / risky), plus any
    extra tier present. Returns {tiers, colors, counts, subs, total}."""
    tiers = [t for t in _TIER_ORDER if t in tier_rows]
    tiers += sorted(t for t in tier_rows if t not in tiers)
    counts = [len(tier_rows.get(t, [])) for t in tiers]
    return {
        "tiers": tiers,
        "colors": [_TIER_COLORS.get(t, "#888780") for t in tiers],
        "counts": counts,
        "subs": [_TIER_SUB.get(t, "") for t in tiers],
        "total": sum(counts),
    }


def _candidate_touch_counts(rows: Sequence[dict]) -> Dict[int, int]:
    """{position: how many candidates mutate it}, parsed from each row's
    ``mutation_string`` (``W153R;A198G`` form — generic, works off the on-disk
    rows alone). A malformed token is skipped rather than fabricated."""
    import re

    touch: Dict[int, int] = {}
    for r in rows:
        ms = str(_g(r, "mutation_string", "") or "")
        seen: set = set()
        for tok in ms.split(";"):
            tok = tok.strip()
            m = re.match(r"^[A-Z](\d+)[A-Z]$", tok)
            if m:
                seen.add(int(m.group(1)))
        for p in seen:
            touch[p] = touch.get(p, 0) + 1
    return touch


def _design_space(designable, catalytic, fixed, contacts, position_features,
                  rows) -> dict:
    """Per-position design-space table over the union of (designable ∪ protected)
    positions: contact frequency / count to the ligand, nearest-ligand distance,
    MSA conservation + permissiveness (occupancy), how many candidates touch the
    position, and its PROTECTION class (catalytic / fixed / designable). Generic:
    contacts / features are read by attribute-or-key so this works both wired into
    the stage (objects) and standalone (reconstructed dicts).
    """
    designable = sorted(set(int(p) for p in (designable or [])))
    catalytic = set(int(p) for p in (catalytic or []))
    fixed = set(int(p) for p in (fixed or [])) - catalytic   # don't double-count
    touch = _candidate_touch_counts(rows)

    # per-position ligand contact stats from the contact list (any object/dict
    # exposing residue_index / distance / contact_probability).
    by_pos_dist: Dict[int, float] = {}
    by_pos_n: Dict[int, int] = {}
    by_pos_prob: Dict[int, float] = {}
    for c in (contacts or []):
        ri = _g(c, "residue_index")
        if ri is None:
            continue
        ri = int(ri)
        d = _num(_g(c, "distance"))
        if d is not None:
            by_pos_dist[ri] = min(by_pos_dist.get(ri, 1e9), d)
        by_pos_n[ri] = by_pos_n.get(ri, 0) + 1
        pr = _num(_g(c, "contact_probability"))
        if pr is not None:
            by_pos_prob[ri] = max(by_pos_prob.get(ri, 0.0), pr)

    # per-position MSA features (conservation + family permissiveness/occupancy).
    pf = {int(_g(f, "target_position")): f for f in (position_features or [])
          if _g(f, "target_position") is not None}

    def _row(pos: int, cls: str) -> dict:
        f = pf.get(pos)
        cons = _num(_g(f, "conservation_score")) if f is not None else None
        gap = _num(_g(f, "gap_frequency")) if f is not None else None
        # family permissiveness ~ 1 - conservation (high = the column tolerates
        # substitution); occupancy = 1 - gap_frequency.
        permissive = None if cons is None else round(1.0 - cons, 3)
        occupancy = None if gap is None else round(1.0 - gap, 3)
        return {
            "position": pos,
            "wt": (str(_g(f, "wt", "")) if f is not None else ""),
            "klass": cls,
            "min_dist": (round(by_pos_dist[pos], 2)
                         if pos in by_pos_dist else None),
            "n_contacts": by_pos_n.get(pos, 0),
            "contact_prob": (round(by_pos_prob[pos], 3)
                             if pos in by_pos_prob else None),
            "conservation": (round(cons, 3) if cons is not None else None),
            "permissive": permissive,
            "occupancy": occupancy,
            "n_candidates": touch.get(pos, 0),
        }

    table: List[dict] = []
    for pos in designable:
        table.append(_row(pos, "designable"))
    for pos in sorted(catalytic):
        table.append(_row(pos, "catalytic"))
    for pos in sorted(fixed):
        table.append(_row(pos, "fixed"))
    table.sort(key=lambda d: d["position"])

    # the designable rows sorted ligand-central first (closest contact), for the
    # heatmap (the active-site core the library actually edits).
    design_rows = sorted(
        (d for d in table if d["klass"] == "designable"),
        key=lambda d: (d["min_dist"] if d["min_dist"] is not None else 1e9))
    return {
        "table": table,
        "design_rows": design_rows,
        "n_designable": len(designable),
        "n_catalytic": len(catalytic),
        "n_fixed": len(fixed),
        "n_protected": len(catalytic) + len(fixed),
        "n_touched": sum(1 for d in design_rows if d["n_candidates"] > 0),
    }


# Generator-rationale + safety columns surfaced in the per-mutation provenance
# table (paper reproducibility). DISTANCE columns are dynamic (one per configured
# extra ligand, ``distance_to_<id>``) and discovered at render time; these are the
# fixed ones every run carries.
_PROV_FEATURE_COLS = (
    ("generator", "source"),
    ("n_mutations", "n mut"),
    ("distance_to_design_ligand", "d→ligand (Å)"),
    ("distance_to_nearest_catalytic", "d→catalytic (Å)"),
    ("conservation", "conserv."),
    ("msa_freq", "MSA freq"),
    ("msa_permissiveness", "permiss."),
    ("ligandmpnn_logp", "MPNN logp"),
    ("risk_flag", "risk"),
    ("disallowed_reason", "reason"),
)


def _dynamic_extra_ligand_cols(rows: Sequence[dict]) -> List[str]:
    """Ordered, de-duplicated ``distance_to_<extra_ligand_id>`` columns present
    across the rows (everything ``distance_to_*`` MINUS the design-ligand /
    nearest-catalytic / per-pos columns). Generic: the ids come from the run."""
    fixed = {
        "distance_to_design_ligand", "distance_to_design_ligand_per_pos",
        "distance_to_nearest_catalytic", "distance_to_nearest_catalytic_per_pos",
    }
    ids = sorted({
        k for r in rows for k in (r.keys() if isinstance(r, dict) else [])
        if k.startswith("distance_to_") and k not in fixed
        and not k.endswith("_per_pos")})
    return ids


def _provenance_rows(rows: Sequence[dict], extra_cols: Sequence[str]) -> List[dict]:
    """Condensed, render-ready per-candidate rows: id + mutation string + the
    fixed provenance columns + each dynamic extra-ligand distance, every value
    coerced to a display string (blank for missing). Sorted clean-first, then by
    candidate id, so the risk-flagged rows cluster at the bottom."""
    out: List[dict] = []
    for r in rows:
        rec = {
            "candidate_id": str(_g(r, "candidate_id", "—") or "—"),
            "mutation_string": str(_g(r, "mutation_string", "—") or "—"),
            "risk": bool(_g(r, "risk_flag", False)),
        }
        for key, _label in _PROV_FEATURE_COLS:
            rec[key] = _g(r, key)
        for key in extra_cols:
            rec[key] = _g(r, key)
        out.append(rec)
    out.sort(key=lambda d: (d["risk"], d["candidate_id"]))
    return out


def compute_mutation_stats(
    generated_rows: Sequence[dict],
    *,
    tier_rows: Optional[Dict[str, List[dict]]] = None,
    designable_positions: Optional[Sequence[int]] = None,
    catalytic_positions: Optional[Sequence[int]] = None,
    fixed_positions: Optional[Sequence[int]] = None,
    contacts: Optional[Sequence] = None,
    position_features: Optional[Sequence] = None,
    wt_pdb_path=None,
    ligand_ids: Optional[Sequence[str]] = None,
) -> dict:
    """Everything the s07 report needs, derived from the on-disk artifacts so it
    works BOTH wired into the stage and standalone.

    ``generated_rows`` = the FULL generated_candidates.json rows (per-candidate:
                    generator, n_mutations, the rationale features + the safety /
                    distance columns + risk flags the stage writes).
    ``tier_rows``   = {tier: rows} for the single/multipoint/risky budget files
                    (optional — a tier missing on disk is simply omitted).
    ``designable_positions`` / ``catalytic_positions`` / ``fixed_positions`` =
                    the s06/s01 position sets (ints).
    ``contacts``    = the residue↔ligand contact list (objects or dicts exposing
                    residue_index / distance / contact_probability).
    ``position_features`` = per-position MSA features (target_position /
                    conservation_score / gap_frequency).
    ``wt_pdb_path`` = path to (or text of) the WT complex model PDB for the 3D
                    viewer; optional — the viewer is omitted if absent.
    ``ligand_ids``  = display ids of the modelled ligand(s) (design ligand first,
                    then extras), read from the run; purely cosmetic labels.
    """
    rows = list(generated_rows or [])
    tier_rows = tier_rows or {}
    extra_cols = _dynamic_extra_ligand_cols(rows)

    gen = _generator_breakdown(rows)
    tier = _tier_breakdown(tier_rows)
    space = _design_space(
        designable_positions, catalytic_positions, fixed_positions,
        contacts, position_features, rows)
    prov = _provenance_rows(rows, extra_cols)

    # candidate-size split (single vs multipoint) over the FULL pool + risk count.
    n_single = sum(1 for r in rows
                   if int(_g(r, "n_mutations", 1) or 1) == 1)
    n_multi = sum(1 for r in rows if int(_g(r, "n_mutations", 1) or 1) >= 2)
    n_risky = sum(1 for r in rows if bool(_g(r, "risk_flag", False)))

    # mutated-position set (for the 3D highlight: the residues the library edits).
    designed_resis = sorted(_candidate_touch_counts(rows).keys())
    catalytic = sorted(set(int(p) for p in (catalytic_positions or [])))
    fixed = sorted(set(int(p) for p in (fixed_positions or []))
                   - set(catalytic))

    return {
        "n_candidates": len(rows),
        "n_single": n_single,
        "n_multi": n_multi,
        "n_risky": n_risky,
        "gen": gen,
        "tier": tier,
        "space": space,
        "provenance": prov,
        "extra_cols": list(extra_cols),
        "ligand_ids": [str(x) for x in (ligand_ids or [])],
        "designed_resis": designed_resis,
        "catalytic_resis": catalytic,
        "fixed_resis": fixed,
        "pdb": _read_pdb_text(wt_pdb_path),
    }


# --------------------------------------------------------------------------- #
# standalone loader (reconstruct the on-disk artifacts from a run dir)
# --------------------------------------------------------------------------- #
def load_generated_provenance(run_dir) -> dict:
    """Read the s07 provenance artifacts off a finished run dir. Returns
    ``{"generated": [...], "tiers": {tier: [...]}}`` — the combined generated pool
    plus whatever budget-tier files are present. Best-effort: a missing tier is
    omitted; a missing combined file yields an empty list (the caller errors)."""
    import json
    from pathlib import Path

    prov = Path(run_dir) / "reports" / "provenance"
    out: dict = {"generated": [], "tiers": {}}
    cf = prov / "generated_candidates.json"
    if cf.exists():
        out["generated"] = json.loads(cf.read_text())
    for tier in _TIER_ORDER:
        tf = prov / f"generated_candidates_{tier}.json"
        if tf.exists():
            out["tiers"][tier] = json.loads(tf.read_text())
    return out


# --------------------------------------------------------------------------- #
# Render layer
# --------------------------------------------------------------------------- #
def _fmt(v, suffix: str = "", nd: int = 2) -> str:
    """Display a numeric cell (``None`` / blank / non-numeric -> '—')."""
    f = _num(v)
    if f is None:
        # keep a short non-numeric token (e.g. a 'reason' string) verbatim
        s = "" if v is None else str(v)
        return s if s else "—"
    return f"{f:.{nd}f}{suffix}"


def _svg_design_heatmap(design_rows: Sequence[dict], *, width: int = 760,
                        cell: int = 26, pad_l: int = 64, pad_t: int = 58,
                        pad_b: int = 8, max_cols: int = 64) -> str:
    """Self-contained inline-SVG HEATMAP (no JS) of the designable positions ×
    four design-space metrics (ligand contact, nearest-ligand closeness, MSA
    permissiveness, candidates touching). One column per position (ligand-central
    first), one row per metric; each cell shaded by the metric's 0..1 value with
    the number overlaid. Matches the report's inline-SVG idiom (CSS-var theming,
    reused .legend / .lg / .sw). Returns '' when there is nothing to draw."""
    import html as _h

    rows = list(design_rows or [])[:max_cols]
    if not rows:
        return ""
    metrics = [
        ("contact freq.", "contact_prob", "#378ADD"),
        ("ligand closeness", "_closeness", "#1D9E75"),
        ("MSA permissive", "permissive", "#BA7517"),
        ("# candidates", "_cand_norm", "#9b59b6"),
    ]
    # derive the two normalised metrics (closeness from distance; candidate count
    # scaled to its own max) so every cell is on a 0..1 shade scale.
    maxc = max((d["n_candidates"] for d in rows), default=0) or 1
    for d in rows:
        md = d.get("min_dist")
        d["_closeness"] = (None if md is None
                           else max(0.0, min(1.0, 1.0 - (md / 8.0))))
        d["_cand_norm"] = d["n_candidates"] / maxc
    n = len(rows)
    # adaptive cell width: shrink columns so ALL designable positions fit the base
    # width instead of clipping the design space to a fixed column cap (the prior
    # max_cols=22 silently dropped positions 23+); floored so labels stay legible.
    cell = max(12, min(cell, (width - pad_l - 10) // max(1, n)))
    plot_w = n * cell
    total_w = max(width, pad_l + plot_w + 10)
    height = pad_t + len(metrics) * cell + pad_b

    parts: List[str] = [
        f'<svg viewBox="0 0 {total_w} {height}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'style="font:11px -apple-system,Segoe UI,Roboto,Arial,sans-serif">']
    # column headers: position numbers (rotated when crowded)
    for ci, d in enumerate(rows):
        x = pad_l + ci * cell + cell / 2.0
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t - 6}" text-anchor="start" '
            f'fill="var(--mut)" font-size="9.5" '
            f'transform="rotate(-55 {x:.1f} {pad_t - 6})">'
            f'{d["wt"]}{d["position"]}</text>')
    # rows
    for ri, (label, key, base) in enumerate(metrics):
        y = pad_t + ri * cell
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + cell / 2.0 + 3:.1f}" '
            f'text-anchor="end" fill="var(--fg)" font-size="10.5">'
            f'{_h.escape(label)}</text>')
        for ci, d in enumerate(rows):
            v = d.get(key)
            x = pad_l + ci * cell
            if v is None:
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell - 1}" '
                    f'height="{cell - 1}" fill="var(--surf)" '
                    f'stroke="var(--line)" stroke-width="0.5"/>'
                    f'<text x="{x + cell / 2.0:.1f}" y="{y + cell / 2.0 + 3:.1f}" '
                    f'text-anchor="middle" fill="var(--mut)" font-size="9">·'
                    f'</text>')
                continue
            t = max(0.0, min(1.0, float(v)))
            op = 0.12 + 0.85 * t
            disp = (str(d["n_candidates"]) if key == "_cand_norm"
                    else f"{t:.2f}"[1:] if t < 1 else "1")
            fg = "#fff" if t >= 0.55 else "var(--fg)"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 1}" height="{cell - 1}" '
                f'fill="{base}" fill-opacity="{op:.2f}">'
                f'<title>{d["wt"]}{d["position"]} · {_h.escape(label)}: '
                f'{t:.2f}</title></rect>'
                f'<text x="{x + cell / 2.0:.1f}" y="{y + cell / 2.0 + 3:.1f}" '
                f'text-anchor="middle" fill="{fg}" font-size="8.5">{disp}</text>')
    parts.append("</svg>")
    leg = "".join(
        f'<span class="lg"><span class="sw" style="background:{base}"></span>'
        f'{_h.escape(lab)}</span>' for lab, _k, base in metrics)
    return f'{"".join(parts)}<div class="legend">{leg}</div>'


def build_mutation_report_html(*, target_id: str, stats: dict,
                               conditions: List[Tuple[str, str]],
                               generated: str, provenance: str = "") -> str:
    import html as _html
    import json as _json
    g = _html.escape
    s = stats

    subtitle = provenance or f"generated {g(generated)}"
    sp = s["space"]
    gen = s["gen"]
    tier = s["tier"]

    # ---- summary cards ---------------------------------------------------- #
    def card(lab, val, sub, color="#111"):
        return (f'<div class="card"><div class="lab">{g(lab)}</div>'
                f'<div class="num" style="color:{color}">{g(str(val))}</div>'
                f'<div class="sub2">{g(sub)}</div></div>')

    n_chem = dict(zip(gen["generators"], gen["counts"]))
    cards = [
        card("candidates", f"{s['n_candidates']:,}", "total generated library"),
        card("single / multi", f"{s['n_single']:,} / {s['n_multi']:,}",
             "1-point vs ≥2-point"),
        card("risk-flagged", f"{s['n_risky']:,}",
             "exploratory tier", "#C0392B" if s["n_risky"] else "#111"),
        card("designable", f"{sp['n_designable']:,}", "positions the library edits"),
        card("protected", f"{sp['n_protected']:,}",
             f"{sp['n_catalytic']} catalytic + {sp['n_fixed']} fixed", "#C0392B"),
    ]
    for gname in gen["generators"]:
        cards.append(card(_GEN_LABELS.get(gname, gname),
                          f"{n_chem.get(gname, 0):,}", "candidates",
                          _GEN_COLORS.get(gname, "#111")))
    cards_html = "".join(cards)

    # ---- generator + tier legends ---------------------------------------- #
    gen_legend = "".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>'
        f'{g(lab)} ({n})</span>'
        for lab, c, n in zip(gen["labels"], gen["colors"], gen["counts"]))
    tier_legend = "".join(
        f'<span class="lg"><span class="sw" style="background:'
        f'{_TIER_COLORS.get(t, "#888780")}"></span>{g(t)} ({n})</span>'
        for t, n in zip(tier["tiers"], tier["counts"]))

    # ---- design-space table ---------------------------------------------- #
    def _cls_badge(klass: str) -> str:
        col = _PROT_COLORS.get(klass, "#888780")
        return (f'<span class="pill" style="background:{col}">{g(klass)}</span>')

    ds_rows = ""
    for d in sp["table"]:
        ds_rows += (
            f'<tr class="{g(d["klass"])}">'
            f'<td><b>{g(d["wt"])}{d["position"]}</b></td>'
            f'<td>{_cls_badge(d["klass"])}</td>'
            f'<td>{_fmt(d["min_dist"], " Å")}</td>'
            f'<td>{d["n_contacts"]}</td>'
            f'<td>{_fmt(d["conservation"], nd=3)}</td>'
            f'<td>{_fmt(d["permissive"], nd=3)}</td>'
            f'<td>{_fmt(d["occupancy"], nd=3)}</td>'
            f'<td><b>{d["n_candidates"]}</b></td></tr>')
    if not ds_rows:
        ds_rows = ("<tr><td colspan=8 class=ck>no positions — check the design "
                   "mask / contacts</td></tr>")
    heatmap_svg = _svg_design_heatmap(sp["design_rows"])

    # ---- per-mutation provenance table (+ dynamic extra-ligand columns) --- #
    extra_cols = s["extra_cols"]
    head_cells = ['<th>candidate</th>', '<th>mutations</th>']
    for key, label in _PROV_FEATURE_COLS:
        head_cells.append(f'<th>{g(label)}</th>')
    for key in extra_cols:
        lid = key[len("distance_to_"):]
        head_cells.append(f'<th>d→{g(lid)} (Å)</th>')
    prov_head = "".join(head_cells)

    def _prov_cell(rec, key) -> str:
        v = rec.get(key)
        if key == "generator":
            gg = str(v or "—")
            col = _GEN_COLORS.get(gg, "#888780")
            return (f'<td><span class="sw" style="background:{col}"></span>'
                    f'{g(_GEN_LABELS.get(gg, gg))}</td>')
        if key == "risk_flag":
            on = bool(v)
            return ('<td><span class="risk-y">flag</span></td>' if on
                    else '<td class=ck>—</td>')
        if key == "forbidden_check":
            ok = (str(v) == "ok")
            return ('<td class=ck>ok</td>' if ok
                    else f'<td><span class="risk-y">{g(str(v))}</span></td>')
        if key == "disallowed_reason":
            txt = str(v or "")
            return (f'<td class=ck title="{g(txt)}">{g(txt[:38])}'
                    f'{"…" if len(txt) > 38 else ""}</td>' if txt
                    else '<td class=ck>—</td>')
        if key in ("n_mutations",):
            return f'<td>{g(str(v if v is not None else "—"))}</td>'
        if key.startswith("distance_to_"):
            return f'<td>{_fmt(v, nd=2)}</td>'
        if key in ("conservation", "msa_freq", "msa_permissiveness",
                   "ligandmpnn_logp"):
            return f'<td>{_fmt(v, nd=3)}</td>'
        return f'<td>{g(str(v)) if v not in (None, "") else "—"}</td>'

    MAX_PROV = 60
    prov = s["provenance"]
    prov_body = ""
    for rec in prov[:MAX_PROV]:
        tr_cls = ' class="riskrow"' if rec["risk"] else ""
        cells = [f'<td><code>{g(rec["candidate_id"])}</code></td>',
                 f'<td class="mut"><b>{g(rec["mutation_string"])}</b></td>']
        for key, _label in _PROV_FEATURE_COLS:
            cells.append(_prov_cell(rec, key))
        for key in extra_cols:
            cells.append(_prov_cell(rec, key))
        prov_body += f"<tr{tr_cls}>{''.join(cells)}</tr>"
    if not prov_body:
        prov_body = ("<tr><td colspan=13 class=ck>no candidates generated</td>"
                     "</tr>")
    n_more = max(0, len(prov) - MAX_PROV)

    # ---- budget-tier table ----------------------------------------------- #
    tier_rows_html = ""
    for t, n, sub in zip(tier["tiers"], tier["counts"], tier["subs"]):
        col = _TIER_COLORS.get(t, "#888780")
        tier_rows_html += (
            f'<tr><td><span class="sw" style="background:{col}"></span>'
            f'<b>{g(t)}</b></td><td><b>{n:,}</b></td>'
            f'<td class=ck>{g(sub)}</td></tr>')
    if not tier_rows_html:
        tier_rows_html = ("<tr><td colspan=3 class=ck>no tier files on disk "
                          "(run wrote no budget tiers)</td></tr>")

    cond_rows = "".join(f"<tr><td class=ck>{g(k)}</td><td>{g(str(v))}</td></tr>"
                        for k, v in conditions)

    # ---- ligand description (generic, from the run) ----------------------- #
    lig_ids = s["ligand_ids"]
    if lig_ids:
        lig_descr = (lig_ids[0] + " (design ligand)"
                     + ("; " + ", ".join(lig_ids[1:]) + " co-modelled"
                        if len(lig_ids) > 1 else ""))
    else:
        lig_descr = "the modelled ligand(s)"

    has_pdb = bool(s.get("pdb"))
    pdb_block = ('<script id="pdbdata" type="text/plain">' + s["pdb"] + "</script>"
                 if has_pdb else "")

    # JSON blob for the client charts + 3D highlight
    blob = _json.dumps({
        "gen": {"labels": gen["labels"], "counts": gen["counts"],
                "colors": gen["colors"]},
        "tier": {"tiers": tier["tiers"], "counts": tier["counts"],
                 "colors": tier["colors"]},
        "designed_resis": s["designed_resis"],
        "catalytic_resis": s["catalytic_resis"],
        "fixed_resis": s["fixed_resis"],
        "has_pdb": has_pdb,
    }, separators=(",", ":"))

    return (_S07_TEMPLATE
            .replace("%%TITLE%%", g(f"{target_id} — mutation library (s07)"))
            .replace("%%TARGET%%", g(target_id))
            .replace("%%PROVSUB%%", subtitle)
            .replace("%%NCAND%%", f"{s['n_candidates']:,}")
            .replace("%%NSINGLE%%", f"{s['n_single']:,}")
            .replace("%%NMULTI%%", f"{s['n_multi']:,}")
            .replace("%%NRISKY%%", f"{s['n_risky']:,}")
            .replace("%%NDESIGN%%", f"{sp['n_designable']:,}")
            .replace("%%NPROT%%", f"{sp['n_protected']:,}")
            .replace("%%NCAT%%", f"{sp['n_catalytic']:,}")
            .replace("%%NFIXED%%", f"{sp['n_fixed']:,}")
            .replace("%%NTOUCHED%%", f"{sp['n_touched']:,}")
            .replace("%%CARDS%%", cards_html)
            .replace("%%GENLEGEND%%", gen_legend)
            .replace("%%TIERLEGEND%%", tier_legend)
            .replace("%%DSROWS%%", ds_rows)
            .replace("%%HEATMAP%%", heatmap_svg)
            .replace("%%PROVHEAD%%", prov_head)
            .replace("%%PROVROWS%%", prov_body)
            .replace("%%NMORE%%", str(n_more))
            .replace("%%TIERROWS%%", tier_rows_html)
            .replace("%%LIGDESCR%%", g(lig_descr))
            .replace("%%CONDROWS%%", cond_rows)
            .replace("%%PDBBLOCK%%", pdb_block)
            .replace("%%TARGETID%%", g(target_id))
            .replace("%%BLOB%%", blob))


def write_mutation_report(path, **kwargs) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_mutation_report_html(**kwargs), encoding="utf-8")


_S07_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
:root{--bg:#fff;--fg:#1c1c1a;--mut:#5f5e5a;--line:#e7e5df;--surf:#f7f6f2}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a18;--fg:#ece9e3;--mut:#a8a69e;--line:#33322e;--surf:#232220}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:2rem 1.25rem 3rem}
h1{font-size:22px;font-weight:500;margin:0 0 2px}
h2{font-size:18px;font-weight:500;margin:2.2rem 0 .5rem}
.sub{color:var(--mut);font-size:14px;margin:0 0 1.5rem}
.note{font-size:13px;color:var(--mut);margin:.3rem 0 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:0 0 1rem}
.card{background:var(--surf);border-radius:8px;padding:.7rem .85rem}
.lab{font-size:12px;color:var(--mut)}.num{font-size:22px;font-weight:500;margin-top:1px}.sub2{font-size:11px;color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:14px}
td.mut{max-width:168px;white-space:normal;overflow-wrap:anywhere;line-height:1.35}
th,td{text-align:left;padding:7px 10px;border-bottom:.5px solid var(--line)}
th{color:var(--mut);font-weight:500;font-size:13px}
td.ck{color:var(--mut)}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:middle}
.legend{display:flex;flex-wrap:wrap;gap:13px;font-size:13px;color:var(--mut);margin:.2rem 0 .6rem}
.lg{display:flex;align-items:center;gap:5px}
.chartbox{position:relative;width:100%;height:230px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;align-items:start}
@media(max-width:680px){.row2{grid-template-columns:1fr}}
.scroll{overflow-x:auto;margin:.4rem 0}
.pill{display:inline-block;color:#fff;font-size:11px;font-weight:600;border-radius:10px;padding:1px 9px;letter-spacing:.02em}
tr.catalytic td,tr.fixed td{background:rgba(192,57,43,.06)}
.riskrow td{background:rgba(192,57,43,.05)}
.risk-y{color:#C0392B;font-weight:600;font-size:12px}
#provtable table{font-size:12.5px}
#provtable td,#provtable th{padding:5px 8px;white-space:nowrap}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
#viewer{width:100%;height:460px;position:relative;border:.5px solid var(--line);border-radius:6px;background:var(--surf)}
.vctrl{display:flex;flex-wrap:wrap;gap:6px 16px;margin:.55rem 0 .35rem}
.cg{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.cgl{color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;margin-right:1px}
.vb{font:inherit;font-size:12px;color:var(--fg);background:var(--surf);border:.5px solid var(--line);border-radius:5px;padding:3px 8px;cursor:pointer;line-height:1.2}
.vb:hover{background:var(--line)}.vb.on{background:#378ADD;color:#fff;border-color:#378ADD}
.warn{font-size:12.5px;color:var(--mut);background:var(--surf);border-left:3px solid #BA7517;padding:.5rem .7rem;border-radius:4px;margin:.5rem 0}
footer{margin-top:2.5rem;border-top:.5px solid var(--line);padding-top:1rem}
footer p{font-size:13.5px}.cite{font-size:12px;color:var(--mut);line-height:1.7}
</style></head>
<body><div class="wrap">
<h1>%%TITLE%%</h1>
<p class="sub">target %%TARGET%% · multi-strategy mutation library · %%NCAND%% candidates over %%NDESIGN%% designable positions · %%NPROT%% protected (catalytic + fixed) · %%PROVSUB%%</p>

<div class="cards">%%CARDS%%</div>

<div class="warn" style="border-left-color:#378ADD">This is the <b>generated design
SPACE</b> — a layered, deduplicated candidate library, NOT a ranked list of
validated mutants. Candidates carry a <b>rationale</b> (which generator proposed
them + structural / evolutionary features) but <b>no activity, stability, or
binding score yet</b>; those are added downstream (s08 mutant folding, s09
re-docking, s10 MD, s11 ranking). Every candidate is guaranteed to touch <b>only
designable</b> positions — the catalytic / fixed core is protected and never
mutated (verified by an independent cross-generator assertion in s07).</div>

<h2>1 · Generation strategy &amp; budget tiers</h2>
<p class="note" style="margin:.1rem 0 .4rem">The library is built by <b>four
complementary generators</b> under per-generator quotas (so the first strategy
can't monopolise the budget), then split into <b>budget tiers</b> for the s08–s10
cost. Counts per generator and per tier:</p>
<div class="row2">
<div><div class="chartbox" style="height:210px"><canvas id="genChart"></canvas></div>
<div class="legend" style="margin-top:.4rem">%%GENLEGEND%%</div></div>
<div><div class="chartbox" style="height:210px"><canvas id="tierChart"></canvas></div>
<div class="legend" style="margin-top:.4rem">%%TIERLEGEND%%</div></div>
</div>
<h2 style="font-size:15px;margin-top:1.2rem">Budget tiers (s08–s10 shortlist)</h2>
<table><thead><tr><th>tier</th><th>candidates</th><th>meaning</th></tr></thead>
<tbody>%%TIERROWS%%</tbody></table>
<p class="note">Each tier is a <b>top-N view</b> ranked by the per-candidate score
(MSA permissiveness, then closeness to the design ligand); the full library is
unchanged. <b>single</b> = one substitution (cheapest to validate); <b>multipoint</b>
= ≥ 2 substitutions on ligand-central positions (where active-site epistasis lives,
FuncLib-style); <b>risky</b> = risk-flagged candidates (Gly/Pro/Cys introduction or
a radical 2-shell substitution near a catalytic residue) surfaced as an explicit
exploratory budget rather than hidden among the clean ones.</p>

<h2>2 · Design space — per-position map</h2>
<p class="note" style="margin:.1rem 0 .4rem">The active-site positions the library
edits, with the <b>protected core</b> shown distinct. <b>%%NDESIGN%%</b> designable
positions (<b>%%NTOUCHED%%</b> actually touched by ≥ 1 candidate); <b>%%NCAT%%</b>
catalytic + <b>%%NFIXED%%</b> fixed positions are PROTECTED (red) and never mutated.
Heatmap columns are the designable positions, ligand-central first:</p>
<div class="scroll">%%HEATMAP%%</div>
<p class="note">Per designable position: <b>contact freq.</b> = its strongest
ligand-contact probability; <b>ligand closeness</b> = 1 − (nearest-ligand distance /
8 Å); <b>MSA permissive</b> = 1 − conservation (high = the family column tolerates
substitution); <b># candidates</b> = how many library members mutate it. Tall warm
cells on a central position = a permissive, ligand-proximal hotspot the library
exploits. Inline SVG, no external library.</p>
<div class="scroll">
<table><thead><tr><th>position</th><th>class</th><th>nearest ligand</th>
<th>contacts</th><th>conservation</th><th>permissiveness</th><th>occupancy</th>
<th># candidates</th></tr></thead>
<tbody>%%DSROWS%%</tbody></table>
</div>
<p class="note"><b>class</b>: <span style="color:#1D9E75;font-weight:600">designable</span>
(the library may mutate it) vs the PROTECTED core —
<span style="color:#C0392B;font-weight:600">catalytic</span> (reaction chemistry) /
<span style="color:#993C1D;font-weight:600">fixed</span> (user-locked or
over-conserved / fold-critical; this is where a cofactor-<b>switch</b> lock would
appear). Protected rows are tinted and excluded from design. <b>permissiveness</b> =
1 − MSA conservation; <b>occupancy</b> = 1 − gap frequency (how well the column
aligns). Nearest-ligand distance and contacts come from the s06 interaction graph.</p>

<h2>3 · Per-mutation provenance <span style="font-size:13px;color:var(--mut)">(methods / reproducibility)</span></h2>
<p class="note" style="margin:.1rem 0 .4rem">The reproducibility appendix: every
generated candidate with the generator that proposed it, the structural distances
(to the design ligand, each co-modelled ligand, and the nearest catalytic residue),
its MSA conservation / occupancy / LigandMPNN log-probability where the generator
recorded them, and its <b>safety status</b> (risk flag + reason, forbidden-position
check). Risk-flagged rows are tinted and sorted last.</p>
<div class="scroll" id="provtable">
<table><thead><tr>%%PROVHEAD%%</tr></thead>
<tbody>%%PROVROWS%%</tbody></table>
</div>
<p class="note">Showing up to 60 of <b>%%NCAND%%</b> candidates (<b>%%NMORE%%</b>
more in <code>reports/provenance/generated_candidates.csv</code>); the budget-tier
files (<code>generated_candidates_{single,multipoint,risky}.csv</code>) carry the
same columns per tier. The <b>forbidden-position check</b> is <code>ok</code> for
every candidate by construction (so it is omitted as a column) — the generators
exclude catalytic / fixed positions and an
independent post-assembly assertion re-verifies it. A blank cell = the generator
did not record that feature (e.g. a chemistry-rules candidate has no MSA
frequency), shown honestly rather than fabricated.</p>

<h2>4 · 3D structure — design region vs protected core</h2>
<div id="viewer"><div style="padding:1rem;color:var(--mut);font-size:13px">loading 3D viewer…</div></div>
<div class="vctrl" id="vctrl" style="display:none">
 <div class="cg"><span class="cgl">protein</span>
  <button class="vb on" onclick="setRep('cartoon',this)">cartoon</button>
  <button class="vb" onclick="setRep('stick',this)">sticks</button>
  <button class="vb" onclick="setRep('trace',this)">trace</button></div>
 <div class="cg"><span class="cgl">show</span>
  <button class="vb on" onclick="toggleDesign(this)">designable</button>
  <button class="vb on" onclick="toggleCore(this)">protected core</button>
  <button class="vb" onclick="toggleLabels(this)">labels</button></div>
 <div class="cg"><span class="cgl">view</span>
  <button class="vb" onclick="viewWhole()">reset</button>
  <button class="vb" onclick="viewPocket()">pocket</button>
  <button class="vb" onclick="spin(this)">spin</button>
  <button class="vb" onclick="snap()">⤓ PNG</button></div>
</div>
<div class="legend" style="margin-top:.5rem">
 <span class="lg"><span class="sw" style="background:#1D9E75"></span>designable / designed (s07)</span>
 <span class="lg"><span class="sw" style="background:#C0392B"></span>catalytic (protected)</span>
 <span class="lg"><span class="sw" style="background:#993C1D"></span>fixed (protected)</span>
 <span class="lg"><span class="sw" style="background:#67c8ff"></span>ligand (%%LIGDESCR%%)</span>
</div>
<p class="note">The WT complex with the cofactor ligand shown as cyan sticks; the
protein is a faded cartoon. The <b>designable / designed</b> positions the library
edits are green sticks, the <b>protected catalytic core</b> is red and the
<b>fixed</b> positions are dark-orange, so the design region stands out against the
protected core inside the pocket. Drag to rotate, scroll to zoom.
3Dmol.js (Rego &amp; Koes 2015).</p>

<footer>
<h2>Limitations &amp; QC</h2>
<div class="warn" style="border-left-color:#C0392B">
<p style="margin:.1rem 0 .5rem"><b>Read this as a generated design SPACE,
not a ranked list of validated mutants.</b> Specific caveats:</p>
<p style="font-size:13px;margin:.35rem 0"><b>(a) No activity / stability score yet.</b>
Candidates carry a generation <b>rationale</b> only (generator + structural /
evolutionary features). Predicted folding (s08), re-docking (s09), MD (s10) and the
final ranking (s11) are what triage them — nothing here estimates k<sub>cat</sub>/K<sub>M</sub>,
ΔΔG, or binding.</p>
<p style="font-size:13px;margin:.35rem 0"><b>(b) Buried-polar-unsatisfied is
deferred.</b> A true unsatisfied-buried-polar call needs the MUTANT side chain's
H-bond partners + burial, which are not modelled at s07 (no mutant structure exists
yet). The provenance carries <code>buried_polar_note = deferred_to_s09_s11</code>;
negative design (<code>neg_buried_core_polar</code>) is applied downstream.</p>
<p style="font-size:13px;margin:.35rem 0"><b>(c) Multipoint epistasis is
combinatorial.</b> The FuncLib-style multipoint tier combines single substitutions
that are each individually plausible, but their <b>joint</b> effect (epistasis,
cumulative destabilisation) is NOT additive and is not evaluated until s08–s10. Treat
multipoint candidates as hypotheses requiring explicit validation, not pre-validated
combinations.</p>
<p style="font-size:13px;margin:.35rem 0"><b>(d) The risk filter is a coarse,
residue-identity heuristic.</b> Risk flags (Gly/Pro/Cys introduction; a radical
physicochemical / volume change within a catalytic residue's 2-shell) use residue
identity + Cα/centroid geometry only — they are a <b>triage prior</b>, deliberately
flag-only by default (the flagged candidates are KEPT and surfaced in the
<code>risky</code> tier), not a structural energy calculation.</p>
</div>
<h2>Methods</h2>
<p><b>Designable region.</b> The mutable positions are the ligand-proximal residues
of the s04/s06 complex (within the design radius of the modelled ligand) that are
NOT catalytic, NOT user-fixed, and — when configured — not over-conserved /
fold-critical. Catalytic and fixed positions form the PROTECTED core and are never
mutated; an independent post-assembly assertion re-verifies that no assembled
candidate touches a protected position (cross-generator safety net).</p>
<p><b>Multi-strategy generation.</b> Four complementary generators populate the
library under per-generator quotas: <b>(i) contact-ensemble chemistry rules</b> —
at each designable position the FULL set of ligand contacts is aggregated into a
distance-weighted ligand-atom role vote (anion / cation / aromatic / H-bond donor /
acceptor / hydrophobic), and the chemically complementary residue pools of the top
roles are proposed; <b>(ii) gated MSA sampling</b> — family-observed substitutions
are sampled per column, dropping poorly-aligned (gappy) columns and requiring the
family actually tolerates the residue, surfacing subfamily / ESM signal at
generation; <b>(iii) ligand-aware LigandMPNN</b> (Dauparas et al. 2025), the
atomic-context-conditioned successor to ProteinMPNN (Dauparas et al. 2022), run on
the protein–ligand complex and then VALIDATED (designable-only, never catalytic /
fixed, WT-matched, bounded edits per design); and <b>(iv) a FuncLib-style multipoint
library</b> (Khersonsky et al. 2018) — 2..N-point combinations sampled from the union
of the single-substitution pools on the most ligand-central designable positions,
where active-site epistasis concentrates. Candidates are deduplicated and trimmed to
the configured budget.</p>
<p><b>Protection policy &amp; risk filters.</b> Catalytic residues (reaction
chemistry) and fixed residues (user-locked or over-conserved / fold-critical — the
locus where a Rossmann-fold cofactor-<b>switch</b> mutant would protect the
cofactor-discriminating fingerprint motif; Medvedev et al. 2022) are excluded from
every generator and re-checked post-assembly. Each candidate is additionally graded
by coarse risk heuristics — introduction of a helix-breaker (Gly / Pro) or a free
cysteine, or a radical physicochemical / side-chain-volume change within the 2-shell
(sequence ±2 or centroid ≤ 6 Å) of a catalytic residue — recorded as a flag + reason
(flag-only by default; an optional filter can drop them). Per-candidate structural
distances (to the design ligand, each co-modelled ligand, and the nearest catalytic
residue) are computed from the s04 complex for the provenance appendix.</p>
<p><b>Budget tiers.</b> The full pool is additionally tiered into single /
multipoint / risky budget views, each ranked best-first by the per-candidate score
and capped to its configured top-N, as the cost-aware shortlist for s08–s10. These
are VIEWS — the downstream pipeline still consumes the full library.
<b>Visualisation</b>: generator / tier counts via Chart.js; the per-position design
map as an inline SVG heatmap; the design region vs protected core on the WT complex
via 3Dmol.js (Rego &amp; Koes 2015). <b>All quantities are generation-time priors</b>,
to be triaged by the downstream folding / docking / MD / ranking stages — not
experimental measurements.</p>
<h2>Analysis conditions</h2>
<table><tbody>%%CONDROWS%%</tbody></table>
<h2>References</h2>
<p class="cite">
Dauparas et al. (2025) <i>Nat. Methods</i> 22:717 — LigandMPNN (atomic-context-conditioned design).
Dauparas et al. (2022) <i>Science</i> 378:49 — ProteinMPNN.
Khersonsky et al. (2018) <i>Mol. Cell</i> 72:178 — FuncLib (active-site multipoint libraries).
Goldenzweig et al. (2016) <i>Mol. Cell</i> 63:337 — PROSS (stability design).
Medvedev et al. (2022) <i>Brief. Bioinform.</i> 23:bbab371 — Rossmann-toolbox (cofactor-specificity prediction/design in Rossmann folds).
Cahn et al. (2017) <i>ACS Synth. Biol.</i> 6:326 — general cofactor-specificity switching.
Rego &amp; Koes (2015) <i>Bioinformatics</i> 31:1322 — 3Dmol.js.
</p>
</footer>
%%PDBBLOCK%%
</div>
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.0/build/3Dmol-min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
var R=%%BLOB%%;
function cssv(n){return getComputedStyle(document.body).getPropertyValue(n).trim()||'#888';}
var MUT=cssv('--mut'),GRID=cssv('--line');
function barOpts(xt,yt,horiz){return {responsive:true,maintainAspectRatio:false,
  indexAxis:horiz?'y':'x',plugins:{legend:{display:false}},
  scales:{x:{title:{display:!horiz,text:xt,color:MUT},ticks:{color:MUT,maxTicksLimit:12},grid:{display:false},beginAtZero:horiz},
   y:{title:{display:horiz,text:xt,color:MUT},ticks:{color:MUT},grid:{color:GRID},beginAtZero:!horiz}}};}
function mkGen(){
  var d=R.gen;if(!d||!d.labels.length){var c=document.getElementById('genChart');if(c)c.parentNode.style.display='none';return;}
  new Chart(document.getElementById('genChart'),{type:'bar',
    data:{labels:d.labels,datasets:[{data:d.counts,backgroundColor:d.colors}]},
    options:barOpts('candidates',null,true)});
}
function mkTier(){
  var d=R.tier;if(!d||!d.tiers.length){var c=document.getElementById('tierChart');if(c)c.parentNode.style.display='none';return;}
  new Chart(document.getElementById('tierChart'),{type:'bar',
    data:{labels:d.tiers,datasets:[{data:d.counts,backgroundColor:d.colors}]},
    options:barOpts('candidates','candidates',false)});
}

// ---- 3D viewer (design region vs protected core on the WT complex) ----
// Reuses the sibling-report 3Dmol idiom: faded cartoon + ligand sticks, with the
// s07 position sets highlighted (designable/designed green, catalytic red, fixed
// dark-orange). Generic — the residue index lists come from the run.
var V=null,MDL=null,_spin=false,_design=true,_core=true,_labels=false;
var curRep='cartoon';
var DESIGN=(R.designed_resis||[]),CAT=(R.catalytic_resis||[]),FIX=(R.fixed_resis||[]);
function applyStyle(){
  if(!V)return;
  V.setStyle({},{});
  var ps={};
  if(curRep==='cartoon')ps.cartoon={color:'#cbc9c2',opacity:0.5};
  else if(curRep==='stick')ps.stick={radius:0.1,colorscheme:'whiteCarbon'};
  else ps.cartoon={style:'trace',thickness:0.3,color:'#cbc9c2'};
  V.setStyle({hetflag:false},ps);
  if(_design&&DESIGN.length)V.addStyle({resi:DESIGN,hetflag:false},{stick:{radius:0.2,colorscheme:'greenCarbon'}});
  if(_core&&CAT.length)V.addStyle({resi:CAT,hetflag:false},{stick:{radius:0.24,colorscheme:'redCarbon'}});
  if(_core&&FIX.length)V.addStyle({resi:FIX,hetflag:false},{stick:{radius:0.2,colorscheme:'orangeCarbon'}});
  V.setStyle({hetflag:true},{stick:{radius:0.22,colorscheme:'cyanCarbon'}});
  V.render();
}
function mark(b){if(b)b.classList.toggle('on');}
function setRep(r,b){curRep=r;if(b&&b.parentNode){b.parentNode.querySelectorAll('.vb').forEach(function(x){x.classList.remove('on');});b.classList.add('on');}applyStyle();}
function toggleDesign(b){_design=!_design;mark(b);applyStyle();}
function toggleCore(b){_core=!_core;mark(b);applyStyle();}
function toggleLabels(b){if(!V)return;_labels=!_labels;mark(b);
  var all=DESIGN.concat(CAT).concat(FIX);
  if(_labels)MDL.selectedAtoms({and:[{hetflag:false},{atom:'CA'},{resi:all}]}).forEach(function(a){a._rl=V.addLabel(a.resn+a.resi,{position:a,fontSize:10,backgroundColor:'black',backgroundOpacity:0.6});});
  else MDL.selectedAtoms({}).forEach(function(a){if(a._rl){V.removeLabel(a._rl);a._rl=null;}});
  V.render();}
function selAll(){var a=DESIGN.concat(CAT).concat(FIX);return a.length?{resi:a,hetflag:false}:{hetflag:true};}
function viewWhole(){if(V){applyStyle();V.zoomTo();V.render();}}
function viewPocket(){if(V){V.zoomTo(selAll());V.zoom(1.3);V.render();}}
function spin(b){if(!V)return;_spin=!_spin;V.spin(_spin?'y':false);mark(b);}
function snap(){if(V){var a=document.createElement('a');a.href=V.pngURI();a.download='%%TARGETID%%_s07_designspace.png';a.click();}}
function initViewer(){
  var box=document.getElementById('viewer');
  if(R.has_pdb!==true){if(box)box.innerHTML='<p style="padding:1rem;color:var(--mut);font-size:13px">WT complex PDB not available — the 3D design-space view is omitted for this run. Sections 1–3 are unaffected.</p>';return;}
  if(!window.$3Dmol){box.innerHTML='<p style="padding:1rem;color:#C0392B;font-size:13px">3Dmol.js failed to load (offline?). The charts and tables above still work.</p>';return;}
  var data=document.getElementById('pdbdata');
  if(!data){box.innerHTML='<p style="padding:1rem;color:var(--mut);font-size:13px">no PDB payload.</p>';return;}
  box.innerHTML='';
  V=$3Dmol.createViewer('viewer',{backgroundColor:'white'});
  MDL=V.addModel(data.textContent,'pdb');
  document.getElementById('vctrl').style.display='flex';
  V.setHoverable({},true,
    function(a){if(!a._lab){a._lab=V.addLabel((a.resn||'')+(a.resi||'')+' · '+a.atom,{position:a,backgroundColor:'black',backgroundOpacity:0.75,fontSize:11});V.render();}},
    function(a){if(a._lab){V.removeLabel(a._lab);a._lab=null;V.render();}});
  applyStyle();
  if(DESIGN.length||CAT.length)viewPocket();else V.zoomTo();
  V.render();
}
function start(){
  if(window.Chart){mkGen();mkTier();}
  setTimeout(initViewer,150);
}
if(document.readyState!=='loading')start();else document.addEventListener('DOMContentLoaded',start);
</script></body></html>
"""
