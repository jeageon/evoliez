"""V7 mechanism-ranked portfolio report (ROADMAP_V7 Phase V7-6, Gates 4/5/6).

The final human-facing artifact of the V7 engine: it renders a *claim-safe* HTML view of
the experimental panel the ``portfolio.builder`` produced, backed by the calibrated
seven-axis ``LedgerBundle``. It is deliberately thin — the science (bands, lanes, controls,
deconvolution) is already decided upstream; this module only *presents* it, and its one hard
job is to never let the presentation over-claim.

Three gates are enforced here (ROADMAP_V7 §11):

  * Gate 4 — every page goes through ``_report_kit.page`` which runs ClaimGuard. Pre-wet-lab
    the report is bounded to hypothesis-grade / screening-level language; no activity, kcat,
    catalytic-superiority, activation-barrier or validated-lead claim can leak (a strict
    render raises, so an injected over-claim fails CI).
  * Gate 5 — full-population q-values are visually and textually separated from
    subset-level evidence (``AxisEvidenceV7.subset_level``); a subset-calibrated axis is
    never read as if it were population-calibrated.
  * Gate 6 — the panel is emitted as machine-readable JSON + CSV alongside the HTML, so the
    exact tested set is reproducible from on-disk artifacts.

The report *consumes* a ``builder.Portfolio`` and a banded ``ledger.LedgerBundle``. It does
not import the builder (that module is a sibling produced in the same phase and may not be
importable in every environment) — the portfolio is duck-typed against the shared contract
(``variant_id / mutation / lane / overall_band / significant_axes / tier_reached /
reason_to_test`` per variant; ``panel_size / lane_counts / claim_ceiling / as_rows()`` on
the panel). Mechanism-generic: no enzyme name, residue list or ligand identity is baked in.

Pure light-env module — stdlib (csv, pathlib) + the report kit + the ledger contract; no
numpy/scipy/rdkit/torch/openmm.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from evoliez.io import _report_kit as kit
from evoliez.portfolio.ledger import (
    ALL_BANDS, BAND_CONSENSUS, BAND_CONTROL, BAND_DEFERRED, BAND_EXPLORATORY,
    BAND_SIGNIFICANT, BAND_STRONG, BAND_UNRESOLVED, PANEL_LAYER_STATISTICAL, SEVEN_AXES,
    ceiling_to_claim_strength, panel_layer,
)

# band -> pill colour (strong warm-through-cool = stronger evidence; deferred = grey).
_BAND_COLOR = {
    BAND_STRONG: "#1f8a4c",
    BAND_SIGNIFICANT: "#2f9e6f",
    BAND_CONSENSUS: "#185fa5",
    BAND_EXPLORATORY: "#b8862b",
    BAND_CONTROL: "#6b5bd6",
    BAND_UNRESOLVED: "#8a8880",
    BAND_DEFERRED: "#b0aea6",
}

# stable CSV column order (shared PortfolioVariant contract). Any extra key an ``as_rows``
# implementation emits is appended (sorted) so nothing is silently dropped.
_CSV_HEADER = [
    "variant_id", "mutation", "lane", "panel_layer", "overall_band", "significant_axes",
    "tier_reached", "is_control", "control_role", "deconvolution_of", "reason_to_test",
]

# allowed, pre-wet-lab phrasing (ROADMAP_V7 §11) — used verbatim so the copy is claim-clean.
_HTML_FILE = "v7_portfolio.html"
_JSON_FILE = "v7_portfolio.json"
_CSV_FILE = "v7_portfolio.csv"


# --- small formatting helpers --------------------------------------------------------
def _band_color(band: str) -> str:
    return _BAND_COLOR.get(band, "#8a8880")


def _band_pill(band: str) -> str:
    return kit.pill(band or BAND_UNRESOLVED, _band_color(band or BAND_UNRESOLVED))


def _pretty_axis(axis: str) -> str:
    return axis.replace("_", " ")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Duck-typed attribute OR mapping access (Portfolio is pydantic; a plain dict works too)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# --- evidence aggregations (read straight off the ledger contract) -------------------
def _overall_band_distribution(bundle) -> Dict[str, int]:
    dist: Dict[str, int] = {b: 0 for b in ALL_BANDS}
    for led in _get(bundle, "ledgers", []) or []:
        b = _get(led, "overall_band", BAND_UNRESOLVED) or BAND_UNRESOLVED
        dist[b] = dist.get(b, 0) + 1
    return dist


def _per_axis_band_distribution(bundle) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {ax: {b: 0 for b in ALL_BANDS} for ax in SEVEN_AXES}
    for led in _get(bundle, "ledgers", []) or []:
        axes = _get(led, "axes", {}) or {}
        for ax in SEVEN_AXES:
            ev = axes.get(ax) if isinstance(axes, dict) else None
            band = _get(ev, "band", BAND_DEFERRED) if ev is not None else BAND_DEFERRED
            table[ax][band] = table[ax].get(band, 0) + 1
    return table


def _subset_level_counts(bundle) -> Dict[str, int]:
    """Per-axis count of ledgers whose q-value is SUBSET-level, not full-population (Gate 5)."""
    counts: Dict[str, int] = {ax: 0 for ax in SEVEN_AXES}
    for led in _get(bundle, "ledgers", []) or []:
        axes = _get(led, "axes", {}) or {}
        for ax in SEVEN_AXES:
            ev = axes.get(ax) if isinstance(axes, dict) else None
            if ev is not None and bool(_get(ev, "subset_level", False)):
                counts[ax] += 1
    return counts


def _resolve_provenance(claim_provenance, bundle):
    """Default provenance is CONSERVATIVE. When a caller supplies provenance we honour it;
    otherwise we reuse the sibling ``claims.ledger_claim_provenance`` if it is importable,
    and fall back to ``None`` (``evaluate(None)`` -> the most-conservative floor verdict).
    We never synthesise a wet-lab-unlocked provenance here."""
    if claim_provenance is not None:
        return claim_provenance
    try:  # sibling module produced in the same phase — optional at import time
        from evoliez.portfolio.claims import ledger_claim_provenance
    except Exception:
        return None
    try:
        return ledger_claim_provenance(bundle)
    except Exception:
        return None


# --- HTML sections -------------------------------------------------------------------
def _banner(claim_ceiling: str) -> str:
    strength = ceiling_to_claim_strength(claim_ceiling)
    return (
        '<div class="note"><b>Claim ceiling — hypothesis-grade / screening-level '
        f'(claim strength: {kit.esc(strength)}).</b> This panel was generated before any '
        'wet-lab measurement. It makes no claim about enzymatic activity, kinetics, or '
        'catalytic performance and asserts no validated lead. Every variant is listed as '
        'prioritized for experimental testing on the strength of a statistically supported '
        'evidence band, a reaction-geometry access signal, or its role as a control / '
        'deconvolution / uncertainty probe.</div>'
    )


def _summary_cards(portfolio, bundle) -> str:
    variants = list(_get(portfolio, "variants", []) or [])
    lane_counts = _get(portfolio, "lane_counts", {}) or {}
    n_lanes = len({ln for ln, c in lane_counts.items() if c}) if lane_counts else len(
        {_get(v, "lane", "") for v in variants})
    n_controls = sum(1 for v in variants if _get(v, "is_control", False))
    dist = _overall_band_distribution(bundle)
    strong = dist.get(BAND_STRONG, 0) + dist.get(BAND_SIGNIFICANT, 0)
    band_sub = " · ".join(f"{b.split('_')[0]} {dist[b]}" for b in ALL_BANDS if dist.get(b))
    panel_size = _get(portfolio, "panel_size", len(variants))
    return '<div class="cards">' + "".join([
        kit.card("Panel size", str(panel_size), "variants prioritized for testing"),
        kit.card("Lanes", str(n_lanes), "distinct portfolio lanes"),
        kit.card("Strong / significant", str(strong), "variants in a calibrated band"),
        kit.card("Controls", str(n_controls), "design-selected controls in panel"),
    ]) + '</div>' + (
        f'<div class="cap">Overall-band distribution: {kit.esc(band_sub)}.</div>'
        if band_sub else "")


def _band_legend() -> str:
    return kit.legend([(b, _band_color(b)) for b in ALL_BANDS])


def _variant_row(v) -> str:
    sig = _get(v, "significant_axes", []) or []
    sig_txt = ", ".join(_pretty_axis(a) for a in sig) if sig else "—"
    ctl = " <span class=\"cap\">(control)</span>" if _get(v, "is_control", False) else ""
    deconv = _get(v, "deconvolution_of", None)
    deconv_txt = (f' <span class="cap">↳ {kit.esc(deconv)}</span>' if deconv else "")
    return (
        "<tr>"
        f"<td>{kit.esc(_get(v, 'variant_id', ''))}{ctl}</td>"
        f"<td>{kit.esc(_get(v, 'mutation', '') or '—')}</td>"
        f"<td>{kit.esc(_get(v, 'lane', ''))}{deconv_txt}</td>"
        f"<td>{_band_pill(_get(v, 'overall_band', BAND_UNRESOLVED))}</td>"
        f"<td>{kit.esc(sig_txt)}</td>"
        f"<td>{kit.esc(_get(v, 'tier_reached', ''))}</td>"
        f"<td>{kit.esc(_get(v, 'reason_to_test', '') or '—')}</td>"
        "</tr>")


def _subtable(title: str, intro: str, variants) -> str:
    head = ("<tr><th>Variant</th><th>Mutation</th><th>Lane</th><th>Overall band</th>"
            "<th>Significant axes</th><th>Tier reached</th><th>Reason to test</th></tr>")
    body = "".join(_variant_row(v) for v in variants) or (
        '<tr><td colspan="7"><em>— none in this layer —</em></td></tr>')
    return (f'<h3>{title}</h3><p class="cap">{intro}</p>'
            '<div class="scroll"><table>' + head + '<tbody>' + body + '</tbody></table></div>')


def _lane_table(portfolio) -> str:
    """Two-layer view (reviewer breakthrough): Layer A = statistically supported evidence-band
    candidates; Layer B = mechanism-protected hypotheses + exploratory / control probes. On a
    hard target Layer A can be empty — that is stated explicitly rather than hidden."""
    variants = list(_get(portfolio, "variants", []) or [])
    layer_a = [v for v in variants if panel_layer(_get(v, "lane", "")) == PANEL_LAYER_STATISTICAL]
    layer_b = [v for v in variants if v not in layer_a]
    a_intro = ("Candidates that cleared a calibrated statistical evidence band (strong / "
               "significant / consensus q against the axis null). These are prioritized on the "
               "strength of computational evidence.")
    if not layer_a:
        a_intro = ("No candidate cleared a strong / significant / consensus statistical evidence "
                   "band in this run — this layer is empty. The panel below is a calibration / "
                   "mechanism-probe panel, not a set of computationally selected leads.")
    b_intro = ("Expert mechanism-protected hypotheses (forced in regardless of bands), their "
               "deconvolution probes, clean single-site probes, uncertainty probes, and "
               "controls — the calibration layer that lets the first wet-lab round estimate "
               "which evidence axes enrich hits.")
    return ('<h2>Experimental panel — two layers</h2>' + _band_legend()
            + _subtable("Layer A — statistical evidence-band candidates", a_intro, layer_a)
            + _subtable("Layer B — mechanism-protected &amp; exploratory panel", b_intro, layer_b))


def _per_axis_table(bundle) -> str:
    table = _per_axis_band_distribution(bundle)
    head = "<tr><th>Axis</th>" + "".join(f"<th>{kit.esc(b)}</th>" for b in ALL_BANDS) + "</tr>"
    rows = []
    for ax in SEVEN_AXES:
        cells = "".join(f"<td>{table[ax][b] or ''}</td>" for b in ALL_BANDS)
        rows.append(f"<tr><td>{kit.esc(_pretty_axis(ax))}</td>{cells}</tr>")
    return ('<h2>Per-axis evidence-band distribution</h2>'
            '<p class="cap">How many ledgers reach each band on each of the seven axes. A '
            'deferred axis is an honest gap (evidence not yet computed), never a zero that '
            'sinks a candidate.</p>'
            '<div class="scroll"><table>' + head + '<tbody>' + "".join(rows)
            + '</tbody></table></div>')


def _subset_note(bundle) -> str:
    counts = _subset_level_counts(bundle)
    flagged = {ax: n for ax, n in counts.items() if n}
    if not flagged:
        return ('<div class="note"><b>Full-population calibration.</b> Every axis q-value in '
                'this panel is calibrated against its full-population null model; no axis '
                'carries subset-level-only evidence.</div>')
    items = "".join(
        f"<li><b>{kit.esc(_pretty_axis(ax))}</b>: {n} variant(s) carry subset-level q-values"
        "</li>" for ax, n in flagged.items())
    return ('<div class="note" style="border-left-color:#b8862b">'
            '<b>Subset-level evidence — read separately from full-population q-values '
            '(Gate 5).</b> The axes below were calibrated against a SUBSET null, not the full '
            'population, so their q-values are subset-level and must not be interpreted as '
            f'population-calibrated evidence:<ul>{items}</ul></div>')


# --- public API ----------------------------------------------------------------------
def render_portfolio_html(portfolio, bundle, *, claim_provenance=None,
                          strict: Optional[bool] = None, generated: str = "",
                          run_id: str = "") -> str:
    """Render the claim-safe V7 portfolio HTML page.

    ``portfolio`` is a ``builder.Portfolio`` (duck-typed), ``bundle`` a banded
    ``ledger.LedgerBundle``. The body is wrapped with ``_report_kit.page`` which runs
    ClaimGuard against ``claim_provenance`` (default: conservative); under ``strict=True``
    (or env ``EVOLIEZ_STRICT_CLAIMS``) any over-claim raises."""
    run_id = run_id or _get(bundle, "run_id", "") or ""
    target = _get(portfolio, "target_id", "") or _get(bundle, "target_id", "") or "—"
    mech = _get(portfolio, "mechanism_class", "") or _get(bundle, "mechanism_class", "") or "—"
    ceiling = _get(portfolio, "claim_ceiling", "") or "L0_hypothesis"

    body_parts: List[str] = [
        '<h1>EvoLiEZ V7 — mechanism-ranked portfolio</h1>',
        f'<p class="sub">Target <code>{kit.esc(target)}</code> · mechanism '
        f'<code>{kit.esc(mech)}</code> · run <code>{kit.esc(run_id or "—")}</code></p>',
        _banner(ceiling),
        _summary_cards(portfolio, bundle),
        _lane_table(portfolio),
        _per_axis_table(bundle),
        _subset_note(bundle),
    ]
    notes = _get(portfolio, "notes", []) or []
    if notes:
        body_parts.append('<h2>Panel notes</h2><ul>'
                          + "".join(f"<li>{kit.esc(n)}</li>" for n in notes) + '</ul>')
    body_parts.append(kit.footer(generated, run_id, extra="V7 portfolio engine"))
    body = "".join(body_parts)

    provenance = _resolve_provenance(claim_provenance, bundle)
    return kit.page("EvoLiEZ V7 — portfolio report", body,
                    claim_provenance=provenance, strict=strict)


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(x) for x in value)
    return str(value)


def _write_csv(portfolio, path: Path, *, caveat: Optional[str] = None) -> None:
    rows: List[Dict[str, Any]] = []
    as_rows = _get(portfolio, "as_rows", None)
    raw = as_rows() if callable(as_rows) else []
    for r in raw:
        rows.append(dict(r) if not isinstance(r, dict) else r)
    extra: List[str] = []
    seen = set(_CSV_HEADER)
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                extra.append(k)
    header = _CSV_HEADER + sorted(extra)   # sorted -> deterministic column order
    with path.open("w", newline="", encoding="utf-8") as fh:
        # a leading #-comment carries the mandatory no-band caveat INTO the CSV itself (reviewer
        # directive), so an experimenter reading only the CSV cannot mistake it for a lead set.
        # Standard readers skip #-lines (pandas comment='#'); the per-row `panel_layer` column
        # is the parseable, per-candidate version of the same distinction.
        if caveat:
            fh.write(f"# {caveat}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        for r in rows:
            writer.writerow([_csv_cell(r.get(c)) for c in header])


def _bundle_summary(portfolio, bundle) -> Dict[str, Any]:
    return {
        "run_id": _get(bundle, "run_id", ""),
        "target_id": _get(bundle, "target_id", ""),
        "mechanism_class": _get(bundle, "mechanism_class", ""),
        "n_candidates": _get(bundle, "n_candidates", 0),
        "n_ledgers": len(_get(bundle, "ledgers", []) or []),
        "panel_size": _get(portfolio, "panel_size", None),
        "claim_ceiling": _get(portfolio, "claim_ceiling", "L0_hypothesis"),
        "axis_null_sizes": dict(_get(bundle, "axis_null_sizes", {}) or {}),
        "overall_band_distribution": _overall_band_distribution(bundle),
        "per_axis_band_distribution": _per_axis_band_distribution(bundle),
        "subset_level_counts": _subset_level_counts(bundle),
        "notes": list(_get(bundle, "notes", []) or []),
    }


def write_portfolio_report(portfolio, bundle, out_dir, *, claim_provenance=None,
                           strict: Optional[bool] = None,
                           run_id: str = "") -> Dict[str, str]:
    """Write the V7 portfolio HTML + JSON + CSV into ``out_dir`` (created if missing).

    Returns ``{'html': ..., 'json': ..., 'csv': ...}`` of the written paths (as strings).
    The JSON is ``portfolio.model_dump()`` plus a compact bundle summary (not the full
    ledger set); the CSV is ``portfolio.as_rows()`` under a stable header. Provenance
    defaults to conservative — a wet-lab-unlocked provenance is never written by default."""
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id = run_id or _get(bundle, "run_id", "") or ""

    html = render_portfolio_html(portfolio, bundle, claim_provenance=claim_provenance,
                                 strict=strict, run_id=run_id)
    html_path = out / _HTML_FILE
    html_path.write_text(html, encoding="utf-8")

    dump = _get(portfolio, "model_dump", None)
    payload = {
        "portfolio": dump() if callable(dump) else _get(portfolio, "__dict__", {}),
        "bundle_summary": _bundle_summary(portfolio, bundle),
    }
    json_path = out / _JSON_FILE
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # surface the no-band caveat (set by the builder when 0 statistical bands) into the CSV
    lane_counts = _get(portfolio, "lane_counts", {}) or {}
    n_statistical = int(lane_counts.get("strong_significant", 0)) + int(
        lane_counts.get("cross_axis_consensus", 0))
    caveat = None
    if n_statistical == 0:
        notes = _get(portfolio, "notes", []) or []
        caveat = next((n for n in notes if "lead set" in n),
                      "No statistically significant evidence band in this run; calibration / "
                      "mechanism-probe panel, not a computationally selected lead set.")
    csv_path = out / _CSV_FILE
    _write_csv(portfolio, csv_path, caveat=caveat)

    return {"html": str(html_path), "json": str(json_path), "csv": str(csv_path)}
