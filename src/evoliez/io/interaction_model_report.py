"""Self-contained HTML report for the s06b family interaction-geometry model.

Presents the self-supervised interaction-geometry model the way a structural-
bioinformatics paper would: the representative-selection funnel (homologs ->
subfamily clusters -> coverage -> N representatives), the Boltz diffusion-sample
ENSEMBLE reliability (per-sample confidence / ipTM / pLDDT / affinity over all
poses), the WT Boltz-ensemble CONTACT-FREQUENCY map (priority-#1 feature: a
per-residue contact map + per-ligand-atom summary so the residues the ligand
engages most reproducibly across the WT diffusion ensemble stand out), the
self-supervised training-data composition (consensus / alternative / outlier /
decoy from robust MAD-z pose selection), the subfamily-holdout model performance,
and an interactive 3D view of the WT complex with the high-frequency ensemble-
contact residues highlighted.

IMPORTANT (honesty / scope): the section-3 contact table + map are the
REPRODUCIBILITY of contacts across the WT's OWN Boltz diffusion samples
(``ensemble_contacts(wt.structure, wt.samples)``), NOT a per-homolog contact-
conservation analysis. The CLASSIFIER is legitimately family-based (it trains on
pose fingerprints from N homolog representatives), but the section-3 contacts are
WT-ensemble only. See the "Limitations & QC" box in the rendered report.

Everything is derived from the ACTUAL on-disk artifacts (the persisted META dict,
``interaction_graphs/s06b_artifacts.json``, ``interaction_model.json``, and the WT
complex PDB), so it works UNCHANGED for any protein / ligand — nothing here is
specific to one target. This top section is the viz-agnostic DATA layer; the render
layer (template + Chart.js + 3Dmol) lives below it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

# Fingerprint layout (features/interaction_descriptor): a fixed-length vector
#   [ distance-histogram (n_bins) | interaction-type counts (6) |
#     summary (4: mean_min, std, min, contact_fraction) | k-shell (k_nearest) ]
# The interaction-type names, in order — used to decode the consensus vector
# honestly (these are the ONLY semantic positions we surface).
_ITYPES = ["hbond", "salt_bridge", "aromatic", "hydrophobic", "vdw", "none"]
_ITYPE_COLORS = {
    "hbond": "#378ADD", "salt_bridge": "#C0392B", "aromatic": "#9b59b6",
    "hydrophobic": "#BA7517", "vdw": "#1D9E75", "none": "#888780",
}


# --------------------------------------------------------------------------- #
# DATA layer
# --------------------------------------------------------------------------- #
def _histogram(values: Sequence[float], lo: float, hi: float,
               n_bins: int = 24) -> Tuple[List[str], List[int]]:
    """Equal-width histogram of `values` over [lo, hi] -> (bin-centre labels,
    counts). Generic; used for every ensemble-quality distribution."""
    if hi <= lo:
        hi = lo + 1.0
    w = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        if v is None:
            continue
        b = int((float(v) - lo) / w)
        b = 0 if b < 0 else (n_bins - 1 if b >= n_bins else b)
        counts[b] += 1
    labels = [f"{lo + (k + 0.5) * w:.2f}" for k in range(n_bins)]
    return labels, counts


def _stats5(values: Sequence[float]) -> dict:
    """min / q1 / median / q3 / max / mean of a numeric column (robust spread
    summary for an ensemble-quality metric). Empty -> all zeros."""
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return {"min": 0.0, "q1": 0.0, "median": 0.0, "q3": 0.0, "max": 0.0,
                "mean": 0.0, "n": 0}

    def _q(p):
        if len(xs) == 1:
            return xs[0]
        i = p * (len(xs) - 1)
        lo = int(i)
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

    return {"min": round(xs[0], 4), "q1": round(_q(0.25), 4),
            "median": round(_q(0.5), 4), "q3": round(_q(0.75), 4),
            "max": round(xs[-1], 4), "mean": round(sum(xs) / len(xs), 4),
            "n": len(xs)}


def _parse_selection_note(note: str, meta: dict) -> dict:
    """Turn the human ``representative_selection`` note into an ordered funnel
    (homolog pool -> clusters covering X% -> N representatives). The note is a
    free-text string from the stage; we parse the numbers we can and ALWAYS fall
    back to the META representatives count, so a missing/odd note never breaks the
    funnel. Returns {mode, n_total, n_clusters, coverage, n_reps, steps:[...]}."""
    import re

    n_reps = int(meta.get("representatives", 0) or 0)
    n_total = n_clusters = None
    coverage = None
    mode = "fixed"
    note = note or ""
    if note.startswith("auto"):
        mode = "auto"
    # "auto: 7004/8062 largest clusters cover 90% of 10581 homologs ..."
    m = re.search(r"(\d+)\s*/\s*(\d+)\s+largest clusters", note)
    if m:
        n_clusters_used, n_clusters = int(m.group(1)), int(m.group(2))
    else:
        n_clusters_used = None
    mcov = re.search(r"cover\s+(\d+)\s*%", note)
    if mcov:
        coverage = int(mcov.group(1)) / 100.0
    mtot = re.search(r"of\s+(\d+)\s+homologs", note)
    if mtot:
        n_total = int(mtot.group(1))
    # "all N clusters" / "fixed representative_homologs=N"
    mall = re.search(r"all\s+(\d+)\s+clusters", note)
    if mall and n_clusters is None:
        n_clusters = int(mall.group(1))

    steps: List[Tuple[str, Optional[int], str]] = []
    if n_total is not None:
        steps.append(("homolog pool", n_total, "deduplicated homologs (s02)"))
    if n_clusters is not None:
        steps.append(("subfamily clusters", n_clusters,
                      "k-mer Jaccard subfamilies"))
    if n_clusters_used is not None and coverage is not None:
        steps.append((f"clusters covering {coverage:.0%}", n_clusters_used,
                      "largest subfamilies first"))
    steps.append(("representatives", n_reps,
                  "one per cluster -> Boltz ensemble"))
    return {"mode": mode, "n_total": n_total, "n_clusters": n_clusters,
            "n_clusters_used": n_clusters_used, "coverage": coverage,
            "n_reps": n_reps,
            "steps": [{"label": l, "value": v, "sub": s} for l, v, s in steps]}


def _decode_consensus(consensus: Sequence[float], k_nearest: int,
                      n_bins: int) -> dict:
    """Honestly decode the consensus fingerprint vector (interaction_model.json)
    into the few interpretable quantities its fixed layout exposes: the dominant
    interaction TYPE mix and the distance-summary block. Anything we cannot name
    is left out rather than invented. Returns {} when the vector is absent."""
    c = [float(x) for x in (consensus or [])]
    if not c:
        return {}
    off = n_bins
    types = c[off:off + len(_ITYPES)]
    summ = c[off + len(_ITYPES):off + len(_ITYPES) + 4]
    type_mix = []
    if len(types) == len(_ITYPES):
        s = sum(types) or 1.0
        type_mix = [{"type": t, "frac": round(v / s, 3),
                     "color": _ITYPE_COLORS[t]}
                    for t, v in zip(_ITYPES, types)]
    dom = (max(type_mix, key=lambda d: d["frac"])["type"]
           if type_mix else None)
    summary = {}
    if len(summ) == 4:
        summary = {"mean_min_dist": round(summ[0], 3),
                   "std_min_dist": round(summ[1], 3),
                   "min_dist": round(summ[2], 3),
                   "contact_fraction": round(summ[3], 3)}
    return {"len": len(c), "type_mix": type_mix, "dominant_type": dom,
            "summary": summary}


def _read_pdb_text(wt_pdb_path) -> Optional[str]:
    """Read the WT complex PDB for the 3D viewer (None if unavailable). Accepts a
    path or already-loaded text; never raises (the report degrades gracefully)."""
    if wt_pdb_path is None:
        return None
    try:
        s = str(wt_pdb_path)
        if "\n" in s and ("ATOM" in s or "HETATM" in s):  # already PDB text
            return s
        if os.path.exists(s):
            with open(s, "r") as fh:
                return fh.read()
    except OSError:
        return None
    return None


def find_wt_complex_pdb(complexes_dir) -> Optional[str]:
    """Locate the WT complex model_0 PDB under a run's ``complexes/`` tree:
    ``boltz/boltz_results_*/predictions/*/*_model_0.pdb`` (generic glob, picks the
    'wt' one if present). Returns the path string or None."""
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


def compute_interaction_stats(meta: dict, artifacts: dict, model: dict,
                              wt_pdb_path=None) -> dict:
    """Everything the s06b report needs, derived from the on-disk artifacts so it
    works BOTH wired into the stage and standalone.

    ``meta``      = the persisted ``interaction_model`` META dict.
    ``artifacts`` = ``s06b_artifacts.json`` (ensemble_contacts / pose_dataset /
                    edge_dataset).
    ``model``     = ``interaction_model.json`` (consensus vector, thr, layout).
    ``wt_pdb_path`` = path to (or text of) the WT complex model PDB for the 3D
                    viewer; optional — the viewer is omitted if absent.
    """
    meta = meta or {}
    artifacts = artifacts or {}
    model = model or {}
    econ = list(artifacts.get("ensemble_contacts", []) or [])
    poses = list(artifacts.get("pose_dataset", []) or [])

    k_nearest = int(model.get("k_nearest", meta.get("k_nearest", 6)) or 6)
    n_bins = int(model.get("n_bins", 8) or 8)
    cutoff = float(model.get("cutoff", meta.get("contact_cutoff", 6.0)) or 6.0)

    # ---- representative-selection funnel ---------------------------------- #
    funnel = _parse_selection_note(meta.get("representative_selection", ""), meta)

    # ---- ensemble pose reliability (the diffusion-sample ensemble quality) - #
    groups = sorted({p.get("group_id") for p in poses if p.get("group_id")})
    pose_metrics = ["confidence_score", "ligand_iptm", "complex_plddt",
                    "affinity_pred_value", "ptm", "iptm", "affinity_probability_binary"]
    # which metrics are actually populated, in priority order
    have = [m for m in pose_metrics
            if any(p.get(m) is not None for p in poses)]
    pose_hists: Dict[str, dict] = {}
    for mname in have:
        vals = [p[mname] for p in poses if p.get(mname) is not None]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        # pLDDT may be 0-1 (Boltz) or 0-100; keep it on its own scale
        labels, counts = _histogram(vals, lo, hi, n_bins=24)
        pose_hists[mname] = {"labels": labels, "counts": counts,
                             "stats": _stats5(vals)}
    # per-representative spread of the headline metric (group_id mean)
    headline = "confidence_score" if "confidence_score" in have else (
        have[0] if have else None)
    per_group: List[dict] = []
    if headline:
        for gid in groups:
            gv = [p[headline] for p in poses
                  if p.get("group_id") == gid and p.get(headline) is not None]
            if gv:
                per_group.append({"group": gid, "mean": round(sum(gv) / len(gv), 4),
                                  "min": round(min(gv), 4), "max": round(max(gv), 4),
                                  "n": len(gv)})
        per_group.sort(key=lambda d: d["mean"], reverse=True)

    # ---- WT Boltz-ensemble contact frequency (NOT homolog conservation) --- #
    # NB: these contacts are ensemble_contacts(wt.structure, wt.samples) — the
    # reproducibility of each (residue, ligand-atom) pair across the WT's OWN
    # Boltz diffusion samples, NOT a per-representative conservation matrix.
    def _g(c, k, default=0.0):
        return c.get(k, default) if isinstance(c, dict) else getattr(c, k, default)

    freqs = [float(_g(c, "contact_frequency")) for c in econ]
    f_labels, f_counts = _histogram(freqs, 0.0, 1.0, n_bins=20)
    # per-residue contact map: max + mean contact frequency over its ligand-atom
    # edges (high max frequency = residues the ligand engages most reproducibly
    # across the WT diffusion ensemble)
    by_res: Dict[int, List[float]] = {}
    res_mindist: Dict[int, float] = {}
    for c in econ:
        ri = int(_g(c, "residue_index"))
        fr = float(_g(c, "contact_frequency"))
        md = float(_g(c, "mean_distance", cutoff))
        by_res.setdefault(ri, []).append(fr)
        res_mindist[ri] = min(res_mindist.get(ri, 1e9), md)
    res_map = sorted(
        ({"residue_index": ri,
          "max_freq": round(max(fs), 4),
          "mean_freq": round(sum(fs) / len(fs), 4),
          "n_edges": len(fs),
          "min_mean_dist": round(res_mindist[ri], 3)}
         for ri, fs in by_res.items()),
        key=lambda d: d["residue_index"])
    # per-ligand-atom contact summary
    by_atom: Dict[str, List[Tuple[float, float]]] = {}
    for c in econ:
        aid = str(_g(c, "ligand_atom_id"))
        by_atom.setdefault(aid, []).append(
            (float(_g(c, "contact_frequency")), float(_g(c, "mean_distance", cutoff))))
    atom_summary = sorted(
        ({"ligand_atom_id": aid,
          "n_residues": len(v),
          "max_freq": round(max(f for f, _ in v), 4),
          "sum_freq": round(sum(f for f, _ in v), 3),
          "min_mean_dist": round(min(d for _, d in v), 3)}
         for aid, v in by_atom.items()),
        key=lambda d: d["max_freq"], reverse=True)
    # high-frequency WT-ensemble contacts: high frequency AND short mean distance
    # (the variable name `consensus_contacts` is the on-disk/JSON key; the prose
    # around it is WT-ensemble, not homolog-consensus — see the limitations box)
    consensus_contacts = sorted(
        ({"residue_index": int(_g(c, "residue_index")),
          "ligand_atom_id": str(_g(c, "ligand_atom_id")),
          "contact_frequency": round(float(_g(c, "contact_frequency")), 4),
          "mean_distance": round(float(_g(c, "mean_distance", cutoff)), 3),
          "confidence_weighted_score":
              round(float(_g(c, "confidence_weighted_score", 0.0)), 4)}
         for c in econ if float(_g(c, "contact_frequency")) >= 0.5),
        key=lambda d: (-d["contact_frequency"], d["mean_distance"]))
    n_strong = len(consensus_contacts)
    n_weak = sum(1 for f in freqs if 0.2 <= f < 0.5)

    # ---- cheap physical QC: clash-range ensemble contacts ----------------- #
    # contacts whose mean distance is below a physically plausible bond/contact
    # floor (< 1.5 Å) are flagged — they need PoseBusters/PLIP QC before they can
    # be trusted (steric clashes the diffusion ensemble did not resolve).
    _CLASH_FLOOR = 1.5
    _dists = [float(_g(c, "mean_distance", cutoff)) for c in econ]
    n_clash_contacts = sum(1 for d in _dists if d < _CLASH_FLOOR)
    min_contact_distance = round(min(_dists), 3) if _dists else None

    # residues the ligand engages most reproducibly across the WT ensemble
    # (max contact frequency >= 0.5); used for the 3D highlight in section 6
    conserved_resis = sorted({d["residue_index"] for d in res_map
                              if d["max_freq"] >= 0.5})

    # ---- training-data composition (self-supervised pose selection) ------- #
    composition = [
        ("consensus", int(meta.get("n_consensus", 0) or 0), "#1D9E75",
         "within MAD-z of the family consensus -> positive"),
        ("alternative", int(meta.get("n_alternative", 0) or 0), "#BA7517",
         "borderline band -> down-weighted, not a hard label"),
        ("outlier", int(meta.get("n_outlier", 0) or 0), "#C0392B",
         "robust MAD-z outlier pose -> negative"),
        ("decoy", int(meta.get("n_decoy", 0) or 0), "#7F77DD",
         "synthetic shuffled pose -> negative"),
        ("hard_decoy", int(meta.get("n_hard_decoy", 0) or 0), "#993C1D",
         "near-miss wrong pose -> negative"),
    ]

    # ---- model + consensus vector ---------------------------------------- #
    consensus_decoded = _decode_consensus(
        model.get("consensus", []), k_nearest, n_bins)

    auroc = meta.get("subfamily_holdout_auroc", None)
    return {
        "meta": {
            "representatives": int(meta.get("representatives", len(groups)) or 0),
            "poses_total": int(meta.get("poses_total", len(poses)) or 0),
            "train_rows": int(meta.get("train_rows", 0) or 0),
            "fp_dim": int(meta.get("fp_dim", model.get("fp_dim", 0)) or 0),
            "model_kind": str(meta.get("model_kind", model.get("kind", "—"))),
            "subfamily_holdout_auroc": (round(float(auroc), 4)
                                        if auroc is not None else None),
            "cutoff": cutoff, "k_nearest": k_nearest, "n_bins": n_bins,
        },
        "funnel": funnel,
        "n_groups": len(groups),
        "headline_metric": headline,
        "pose_hists": pose_hists, "pose_metrics_have": have,
        "per_group": per_group,
        "n_contacts": len(econ),
        "freq_hist": {"labels": f_labels, "counts": f_counts},
        "res_map": res_map, "atom_summary": atom_summary,
        "consensus_contacts": consensus_contacts,
        "n_strong": n_strong, "n_weak": n_weak,
        "n_clash_contacts": n_clash_contacts,
        "min_contact_distance": min_contact_distance,
        "clash_floor": _CLASH_FLOOR,
        "conserved_resis": conserved_resis,
        "composition": [{"label": l, "count": n, "color": c, "sub": s}
                        for l, n, c, s in composition],
        "n_train_neg": sum(n for l, n, _, _ in composition
                           if l in ("outlier", "decoy", "hard_decoy")),
        "consensus_vec": consensus_decoded,
        "model_thr": (round(float(model["thr"]), 4)
                      if model.get("thr") is not None else None),
        "model_scale": (round(float(model["scale"]), 4)
                        if model.get("scale") is not None else None),
        "pdb": _read_pdb_text(wt_pdb_path),
    }


# --------------------------------------------------------------------------- #
# Render layer
# --------------------------------------------------------------------------- #
def _band(v: float, good: float, ok: float) -> str:
    return "#1D9E75" if v >= good else ("#BA7517" if v >= ok else "#C0392B")


def build_interaction_report_html(*, target_id: str, stats: dict,
                                  conditions: List[Tuple[str, str]],
                                  generated: str) -> str:
    import html as _html
    import json as _json
    g = _html.escape
    s = stats
    m = s["meta"]
    auroc = m["subfamily_holdout_auroc"]

    def card(lab, val, sub, color="#111"):
        return (f'<div class="card"><div class="lab">{g(lab)}</div>'
                f'<div class="num" style="color:{color}">{g(val)}</div>'
                f'<div class="sub2">{g(sub)}</div></div>')

    cards = [
        card("holdout AUROC", f"{auroc:.3f}" if auroc is not None else "—",
             "subfamily-held-out", _band(auroc or 0, 0.8, 0.65)
             if auroc is not None else "#111"),
        card("representatives", f"{m['representatives']:,}",
             "subfamily reps -> ensemble"),
        card("ensemble poses", f"{m['poses_total']:,}", "Boltz diffusion samples"),
        card("training rows", f"{m['train_rows']:,}", "pose examples"),
        card("fingerprint dim", f"{m['fp_dim']}", "per-pose feature vector"),
        card("model", g(m["model_kind"]), "consensus classifier"),
    ]
    cards_html = "".join(cards)

    # ---- representative-selection funnel ---------------------------------- #
    fn = s["funnel"]
    steps = fn["steps"]
    fmax = max((st["value"] or 0) for st in steps) or 1
    funnel_rows = ""
    for st in steps:
        v = st["value"] or 0
        w = max(4.0, 100.0 * v / fmax)
        funnel_rows += (
            f'<div class="frow"><div class="flab">{g(st["label"])}</div>'
            f'<div class="ftrack"><div class="fbar" style="width:{w:.1f}%"></div>'
            f'<span class="fval">{v:,}</span></div>'
            f'<div class="fsub">{g(st["sub"])}</div></div>')
    sel_note = ("auto-selected by largest-subfamily coverage"
                if fn["mode"] == "auto" else "fixed representative count")

    # ---- per-representative spread table ---------------------------------- #
    pg = s["per_group"]
    pg_rows = "".join(
        f'<tr><td>{g(str(d["group"]))}</td><td>{d["mean"]:.3f}</td>'
        f'<td>{d["min"]:.3f}</td><td>{d["max"]:.3f}</td><td>{d["n"]}</td></tr>'
        for d in (pg[:6] + pg[-3:] if len(pg) > 9 else pg))
    if not pg_rows:
        pg_rows = "<tr><td colspan=5 class=ck>no per-representative pose metrics</td></tr>"

    # ---- WT-ensemble high-frequency contacts table ------------------------ #
    cc = s["consensus_contacts"]
    cc_rows = "".join(
        f'<tr><td>residue {d["residue_index"]}</td><td>{g(d["ligand_atom_id"])}</td>'
        f'<td><b>{d["contact_frequency"]:.2f}</b></td><td>{d["mean_distance"]:.2f} Å</td>'
        f'<td>{d["confidence_weighted_score"]:.3f}</td></tr>'
        for d in cc[:18])
    if not cc_rows:
        cc_rows = ("<tr><td colspan=5 class=ck>no contact reached frequency "
                   "≥ 0.5 in the WT ensemble</td></tr>")

    # ---- per-ligand-atom summary table ------------------------------------ #
    at = s["atom_summary"]
    at_rows = "".join(
        f'<tr><td>{g(d["ligand_atom_id"])}</td><td>{d["n_residues"]}</td>'
        f'<td>{d["max_freq"]:.2f}</td><td>{d["sum_freq"]:.2f}</td>'
        f'<td>{d["min_mean_dist"]:.2f} Å</td></tr>'
        for d in at[:16])
    if not at_rows:
        at_rows = "<tr><td colspan=5 class=ck>no ensemble contacts</td></tr>"

    # ---- consensus vector (decoded) --------------------------------------- #
    cv = s["consensus_vec"]
    cv_html = ""
    if cv:
        mix = "".join(
            f'<span class="lg"><span class="sw" style="background:{t["color"]}">'
            f'</span>{g(t["type"])} {t["frac"]:.0%}</span>'
            for t in cv.get("type_mix", []) if t["frac"] > 0.0)
        summ = cv.get("summary", {})
        cv_html = (
            f'<p class="note" style="margin:.4rem 0 .2rem"><b>Consensus fingerprint</b> '
            f'(length {cv["len"]}): dominant interaction type '
            f'<b>{g(cv.get("dominant_type") or "—")}</b>; mean nearest-residue distance '
            f'{summ.get("mean_min_dist", "—")} Å, contact fraction '
            f'{summ.get("contact_fraction", "—")}.</p>'
            f'<div class="legend">{mix}</div>')

    # ---- composition rows + headline -------------------------------------- #
    comp = s["composition"]
    comp_total = sum(c["count"] for c in comp) or 1
    comp_rows = "".join(
        f'<tr><td><span class="sw" style="background:{c["color"]}"></span>'
        f'{g(c["label"])}</td><td><b>{c["count"]:,}</b></td>'
        f'<td>{100.0 * c["count"] / comp_total:.0f}%</td>'
        f'<td class=ck>{g(c["sub"])}</td></tr>'
        for c in comp)
    pos = next((c["count"] for c in comp if c["label"] == "consensus"), 0)

    cond_rows = "".join(f"<tr><td class=ck>{g(k)}</td><td>{g(str(v))}</td></tr>"
                        for k, v in conditions)

    has_pdb = bool(s.get("pdb"))
    pdb_block = ('<script id="pdbdata" type="text/plain">' + s["pdb"] + "</script>"
                 if has_pdb else "")

    # ---- clash-range physical-QC line (only when short contacts exist) ----- #
    n_clash = int(s.get("n_clash_contacts", 0) or 0)
    floor = s.get("clash_floor", 1.5)
    mind = s.get("min_contact_distance")
    clash_qc = ""
    if n_clash > 0:
        mintxt = (f" (closest {mind:.2f} Å)"
                  if isinstance(mind, (int, float)) else "")
        clash_qc = (
            f'<div class="warn" style="border-left-color:#C0392B">'
            f'<b>Physical QC:</b> {n_clash} ensemble contact'
            f'{"s" if n_clash != 1 else ""} at clash-range &lt; {floor:g} Å'
            f'{mintxt} — physically implausible, flagged for PoseBusters / PLIP '
            f'QC before any binding-mode claim.</div>')

    # JSON blob for the client charts + 3D highlight
    blob = _json.dumps({
        "freq_hist": s["freq_hist"],
        "res_map": s["res_map"],
        "pose_hists": s["pose_hists"],
        "headline_metric": s["headline_metric"],
        "composition": [{"label": c["label"], "count": c["count"],
                         "color": c["color"]} for c in comp],
        "conserved_resis": s["conserved_resis"],
        "has_pdb": has_pdb,
        "cutoff": m["cutoff"],
    }, separators=(",", ":"))

    return (_INTERACTION_TEMPLATE
            .replace("%%TITLE%%", g(f"{target_id} — family interaction-geometry model"))
            .replace("%%TARGET%%", g(target_id))
            .replace("%%GENERATED%%", g(generated))
            .replace("%%CARDS%%", cards_html)
            .replace("%%FUNNELROWS%%", funnel_rows)
            .replace("%%SELNOTE%%", g(sel_note))
            .replace("%%NREPS%%", f"{m['representatives']:,}")
            .replace("%%NPOSES%%", f"{m['poses_total']:,}")
            .replace("%%NGROUPS%%", str(s["n_groups"]))
            .replace("%%HEADLINE%%", g(s["headline_metric"] or "confidence_score"))
            .replace("%%PGROWS%%", pg_rows)
            .replace("%%NCONTACTS%%", f"{s['n_contacts']:,}")
            .replace("%%NSTRONG%%", str(s["n_strong"]))
            .replace("%%NWEAK%%", str(s["n_weak"]))
            .replace("%%CCROWS%%", cc_rows)
            .replace("%%ATROWS%%", at_rows)
            .replace("%%CVHTML%%", cv_html)
            .replace("%%COMPROWS%%", comp_rows)
            .replace("%%NPOS%%", f"{pos:,}")
            .replace("%%NNEG%%", f"{s['n_train_neg']:,}")
            .replace("%%AUROC%%", f"{auroc:.4f}" if auroc is not None else "—")
            .replace("%%AUROCPCT%%", f"{auroc * 100:.1f}" if auroc is not None else "—")
            .replace("%%MODELKIND%%", g(m["model_kind"]))
            .replace("%%NCONSERVED%%", str(len(s["conserved_resis"])))
            .replace("%%CLASHQC%%", clash_qc)
            .replace("%%NCLASH%%", str(n_clash))
            .replace("%%TARGETID%%", g(target_id))
            .replace("%%PDBBLOCK%%", pdb_block)
            .replace("%%CONDROWS%%", cond_rows)
            .replace("%%BLOB%%", blob))


def write_interaction_report(path, **kwargs) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_interaction_report_html(**kwargs), encoding="utf-8")


_INTERACTION_TEMPLATE = r"""<!DOCTYPE html>
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
.sw{width:11px;height:11px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:middle}
.legend{display:flex;flex-wrap:wrap;gap:13px;font-size:13px;color:var(--mut);margin:.2rem 0 .6rem}
.lg{display:flex;align-items:center;gap:5px}
.chartbox{position:relative;width:100%;height:230px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;align-items:start}
@media(max-width:680px){.row2{grid-template-columns:1fr}}
.funnel{margin:.6rem 0 .2rem}
.frow{margin:.45rem 0}
.flab{font-size:13px;font-weight:500}
.ftrack{position:relative;height:26px;background:var(--surf);border-radius:5px;margin:.15rem 0;overflow:hidden}
.fbar{height:100%;background:linear-gradient(90deg,#378ADD,#1D9E75);border-radius:5px;min-width:4px}
.fval{position:absolute;right:8px;top:0;line-height:26px;font-size:12.5px;font-weight:600;color:var(--fg)}
.fsub{font-size:11.5px;color:var(--mut)}
.aurocbig{font-size:46px;font-weight:600;line-height:1;letter-spacing:-1px}
.gauge{height:14px;border-radius:7px;background:linear-gradient(90deg,#C0392B,#FFDB13,#1D9E75);position:relative;margin:.5rem 0}
.gauge .pin{position:absolute;top:-4px;width:3px;height:22px;background:var(--fg);border-radius:2px}
.gauge .mid{position:absolute;left:50%;top:0;width:1px;height:14px;background:rgba(128,128,128,.6)}
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
<p class="sub">target %%TARGET%% · self-supervised family interaction-geometry model · %%NREPS%% subfamily representatives × Boltz diffusion ensemble · generated %%GENERATED%%</p>

<div class="cards">%%CARDS%%</div>

<div class="warn" style="border-left-color:#378ADD">Everything here is derived from a
<b>Boltz-2 / AlphaFold3-class diffusion ENSEMBLE</b>, not experiment, and is a
<b>computational PRIOR for prioritisation</b> — not a measured or validated binding
mode. The <b>classifier</b> (sections 4–5) is family-based: it learns from pose
fingerprints across %%NREPS%% homolog representatives. The <b>section-3 contact
table / map</b>, however, is the <i>reproducibility of contacts across the wild-type's
own diffusion samples</i> (WT-ensemble), <b>not</b> a per-homolog conservation
analysis. Read it as a learned prior over plausible interaction geometry for
ranking, to be confirmed experimentally. See <b>Limitations &amp; QC</b> below.</div>

<h2>1 · Representative selection</h2>
<p class="note" style="margin:.1rem 0 .3rem">%%SELNOTE%% — the <b>classifier</b>
(sections 4–5) trains on a diverse panel of subfamily representatives, not the single
query, so its learned geometry is a <b>family</b> signal rather than one structure.
(The section-3 contact map is computed on the WT alone — see section 3.)</p>
<div class="funnel">%%FUNNELROWS%%</div>
<p class="note">Homologs are clustered into subfamilies (k-mer Jaccard); the
<b>largest</b> clusters are taken first (best-supported subfamilies) until they
cover the target fraction of the homolog pool, then clamped to the configured
[min, max]. One highest-identity representative per chosen cluster seeds the Boltz
family ensemble — a deeper / more diverse MSA therefore auto-scales to more
representatives (more training data).</p>

<h2>2 · Ensemble pose reliability</h2>
<p class="note" style="margin:.1rem 0 .4rem">Quality of the <b>%%NPOSES%% diffusion
samples</b> (across %%NGROUPS%% representatives). Each Boltz sample carries its own
confidence — these distributions are the reliability of the ensemble the contact
fingerprint is computed from.</p>
<div class="row2">
<div><div class="chartbox"><canvas id="h0"></canvas></div>
<p class="note" id="h0note"></p></div>
<div><div class="chartbox"><canvas id="h1"></canvas></div>
<p class="note" id="h1note"></p></div>
</div>
<div class="row2" style="margin-top:1rem">
<div><div class="chartbox"><canvas id="h2"></canvas></div>
<p class="note" id="h2note"></p></div>
<div><div class="chartbox"><canvas id="h3"></canvas></div>
<p class="note" id="h3note"></p></div>
</div>
<h2 style="font-size:15px;margin-top:1.2rem">Per-representative spread (%%HEADLINE%%)</h2>
<table><thead><tr><th>representative</th><th>mean</th><th>min</th><th>max</th><th>samples</th></tr></thead>
<tbody>%%PGROWS%%</tbody></table>
<p class="note">Top and bottom representatives by mean %%HEADLINE%% across their own
diffusion samples. A representative whose ensemble is tight and high-confidence
contributes cleaner training poses; a low/spread one is effectively
down-weighted (Boltz confidence enters pose selection as a SAMPLE WEIGHT, never as
a label).</p>

<h2>3 · WT Boltz-ensemble contact frequency <span style="font-size:13px;color:var(--mut)">(priority-#1 feature)</span></h2>
<p class="note" style="margin:.1rem 0 .4rem">The core signal: <b>WT-ensemble contact
frequency</b> per (residue, ligand-atom) pair — the fraction of the <b>wild-type's own
Boltz diffusion samples</b> in which that atom contacts that residue within %%TARGETID%%'s
interaction cutoff. This is <b>WT diffusion-sample reproducibility</b>, not a
per-homolog conservation analysis (see Limitations &amp; QC). Contacts reproduced
across the WT ensemble are the more reliable ones. %%NCONTACTS%% pairs;
<b>%%NSTRONG%%</b> reach frequency ≥ 0.5, %%NWEAK%% are weak (0.2–0.5).</p>
%%CLASHQC%%
<div class="row2">
<div><div class="chartbox"><canvas id="freqChart"></canvas></div>
<p class="note">Distribution of WT-ensemble contact frequency over all (residue,
ligand-atom) pairs. A spike near 1.0 = a contact present in nearly every WT diffusion
sample; a long low tail = transient brushes. The high-frequency pairs are the most
reproducible part of the predicted WT binding pose.</p></div>
<div><div class="chartbox"><canvas id="resmapChart"></canvas></div>
<p class="note">Per-residue contact map: each residue's <b>maximum</b> WT-ensemble
contact frequency over its ligand-atom edges. Tall bars = residues the ligand engages
most reproducibly across the <b>WT diffusion ensemble</b> — the high-frequency contacts
of the predicted WT pose, not a family-conserved binding site.</p></div>
</div>
<h2 style="font-size:15px;margin-top:1.2rem">High-frequency WT-ensemble contacts (frequency ≥ 0.5)</h2>
<table><thead><tr><th>residue</th><th>ligand atom</th><th>contact freq.</th><th>mean distance</th><th>conf.-weighted</th></tr></thead>
<tbody>%%CCROWS%%</tbody></table>
<p class="note">The most reproducible WT-ensemble contacts: high frequency AND short
mean distance, sorted by frequency. The confidence-weighted score folds in per-residue
pLDDT and ligand-ipTM so a frequent contact in a well-predicted region outranks an
equally frequent one in a low-confidence loop. <b>Ligand-atom ids are not yet
chemically annotated</b> — for runs parsed before the ligand-chain-scoping fix a
high-index atom id may belong to a co-modelled extra ligand (cofactor / substrate),
not the primary ligand.</p>
<h2 style="font-size:15px;margin-top:1.2rem">Per-ligand-atom contact summary</h2>
<table><thead><tr><th>ligand atom</th><th>residue partners</th><th>max freq.</th><th>Σ freq.</th><th>closest mean dist.</th></tr></thead>
<tbody>%%ATROWS%%</tbody></table>
<p class="note">Which ligand atoms are anchored (high max frequency, several
partners) vs solvent-exposed (low) <b>in the WT ensemble</b>. This is the
ligand's-eye view of the same WT-ensemble contact fingerprint. Atom ids are not yet
chemically annotated, so high-index ids may correspond to a co-modelled extra ligand
in pre-fix runs.</p>

<h2>4 · Training-data composition <span style="font-size:13px;color:var(--mut)">(self-supervised)</span></h2>
<p class="note" style="margin:.1rem 0 .4rem">No experimental labels are used. Poses are
pseudo-labelled from each representative's own diffusion ensemble:
<b>%%NPOS%%</b> consensus positives vs <b>%%NNEG%%</b> negatives (statistical
outliers + synthetic decoys), pooled across the %%NREPS%%-representative panel.</p>
<div class="row2">
<div><div class="chartbox" style="height:210px"><canvas id="compChart"></canvas></div></div>
<div><table><thead><tr><th>class</th><th>count</th><th>share</th><th>role</th></tr></thead>
<tbody>%%COMPROWS%%</tbody></table></div>
</div>
<p class="note">Per representative, each pose's interaction fingerprint is compared to
its group's robust centre by a <b>median / MAD z-score</b> (median-absolute-deviation,
outlier-resistant): poses within <code>select_z</code> are <b>consensus</b> positives;
beyond <code>outlier_z</code> are <b>outlier</b> negatives; the borderline band is kept
as down-weighted <b>alternative</b> (not a hard label). Synthetic <b>decoys</b> (shuffled
fingerprints) and <b>hard decoys</b> (near-miss poses) add unambiguous negatives so the
classifier learns the consensus geometry, not a class prior.</p>
<div class="warn">These are <b>pseudo-labels</b> from the model's own ensemble, not
ground truth. They encode "geometrically typical for this family" — useful as a prior,
circular if read as proof. The subfamily holdout (below) is what guards against the
classifier simply memorising that circularity.</div>

<h2>5 · Model performance</h2>
<div class="row2">
<div>
<div class="aurocbig" style="color:#1D9E75">%%AUROC%%</div>
<p class="note" style="margin:.1rem 0 .4rem">subfamily-holdout AUROC · model =
<b>%%MODELKIND%%</b></p>
<div class="gauge"><div class="mid"></div><div class="pin" id="aurocpin"></div></div>
<p class="note">0.5 = chance (mid mark), 1.0 = perfect ranking of held-out consensus
poses above decoys.</p>
%%CVHTML%%
</div>
<div>
<p style="font-size:13.5px;margin:.2rem 0"><b>Subfamily / phylogenetic holdout.</b>
One whole homolog group is removed from training; the classifier is refit on the
rest and must still rank that group's held-out consensus poses above its decoys
(AUROC = %%AUROC%%). Because the held-out subfamily is unseen, a high score means the
model learned <b>transferable interaction geometry</b>, not group-specific memorisation
— it guards against evolutionary leakage and the consensus circularity above.</p>
</div>
</div>

<h2>6 · 3D structure — high-frequency WT-ensemble contact residues</h2>
<div id="viewer"><div style="padding:1rem;color:var(--mut);font-size:13px">loading 3D viewer…</div></div>
<div class="vctrl" id="vctrl" style="display:none">
 <div class="cg"><span class="cgl">protein</span>
  <button class="vb on" onclick="setRep('cartoon',this)">cartoon</button>
  <button class="vb" onclick="setRep('stick',this)">sticks</button>
  <button class="vb" onclick="setRep('trace',this)">trace</button></div>
 <div class="cg"><span class="cgl">pocket</span>
  <button class="vb on" onclick="togglePocket(this)">consensus residues</button>
  <button class="vb" onclick="togglePsurf(this)">surface</button>
  <button class="vb" onclick="toggleLabels(this)">labels</button></div>
 <div class="cg"><span class="cgl">view</span>
  <button class="vb" onclick="viewWhole()">reset</button>
  <button class="vb" onclick="viewPocket()">pocket</button>
  <button class="vb" onclick="spin(this)">spin</button>
  <button class="vb" onclick="snap()">⤓ PNG</button></div>
</div>
<p class="note"><b>%%NCONSERVED%%</b> high-frequency contact residues (WT-ensemble
contact frequency ≥ 0.5, from section 3) are highlighted as coloured sticks on the WT
complex; the protein is a faded cartoon and the ligand is shown as sticks. This marks
where the <b>predicted WT pose</b> places its most reproducible contacts — a
computational prior for the binding mode, not a measured or family-conserved pocket.
Drag to rotate, scroll to zoom. 3Dmol.js (Rego &amp; Koes 2015).</p>

<footer>
<h2>Limitations &amp; QC</h2>
<div class="warn" style="border-left-color:#C0392B">
<p style="margin:.1rem 0 .5rem"><b>Read this report as a COMPUTATIONAL PRIOR for
prioritisation, not a measured or validated binding mode.</b> Specific caveats:</p>
%%CLASHQC%%
<p style="font-size:13px;margin:.35rem 0"><b>(a) Section-3 is WT-ensemble
reproducibility, not family conservation.</b> The contact table / map count the
fraction of the <b>wild-type's own Boltz diffusion samples</b> in which each (residue,
ligand-atom) pair is in contact. They are <b>NOT</b> a per-representative contact-
conservation matrix. A true family-conservation claim needs a residue × ligand-atom ×
representative analysis, which is <b>not yet computed</b>. (The classifier in
sections 4–5 IS family-based — it trains on fingerprints from %%NREPS%%
representatives — but the section-3 contacts are WT-only.)</p>
<p style="font-size:13px;margin:.35rem 0"><b>(b) Contact list may include co-modelled
extra ligands.</b> For runs parsed before the recent ligand-chain-scoping fix, the
ligand atom list can merge co-modelled <b>extra ligands</b> (cofactor + substrate) into
one atom set. Atoms are <b>not yet chemically annotated</b>, so a high-index ligand-atom
id may belong to an extra ligand, not the primary one.</p>
<p style="font-size:13px;margin:.35rem 0"><b>(c) Some contacts are at clash range.</b>
A subset of ensemble contacts sit at physically implausible distances (&lt; 1.5 Å);
these require <b>PoseBusters / PLIP physical QC</b> before any publication or
binding-mode claim. The count, when non-zero, is flagged above and in section 3.</p>
<p style="font-size:13px;margin:.35rem 0"><b>(d) The holdout AUROC is internal
cross-validation, not experimental validation.</b> The subfamily-holdout AUROC
(section 5) measures self-supervised consensus-vs-decoy separation on a held-out
homolog group. It is an <b>internal</b> guard against memorisation / circularity —
<b>not</b> a measured binding affinity or an experimentally confirmed pose.</p>
</div>
<h2>Methods</h2>
<p><b>Family ensemble.</b> A diverse panel of subfamily representatives (one per MSA
cluster, largest subfamilies first) was selected from the homolog pool, and for each
the protein–ligand complex was predicted with <b>Boltz-2</b> (Passaro et al. 2025),
an AlphaFold3-class (Abramson et al. 2024) diffusion co-folding model, drawing a
<b>diffusion-sample ensemble</b> per representative. Each sample carries its own
confidence (pTM / ipTM / ligand-ipTM / complex-pLDDT / PAE), used downstream only as a
per-pose <b>sample weight</b>, never as a label.</p>
<p><b>WT-ensemble contact-frequency map.</b> For every pose, a fixed-length,
ligand-size- and sequence-length-independent interaction fingerprint was extracted
(per-ligand-atom nearest-residue distance histogram + interaction-type counts +
k-nearest shell). For the <b>section-3 contact table / map</b> we computed, over the
<b>wild-type's own diffusion samples only</b>, the <b>contact frequency</b> of each
(residue, ligand-atom) pair — the fraction of WT samples in contact within the
interaction cutoff. This measures <b>WT diffusion-sample reproducibility</b>, NOT
conservation across the homolog representatives; a true family-conservation claim
would require a residue × ligand-atom × representative analysis (not computed here).</p>
<p><b>Self-supervised pose selection &amp; classifier.</b> Within each representative's
ensemble, poses were pseudo-labelled by a robust <b>median/MAD z-score</b> on the
fingerprint (Rousseeuw &amp; Croux 1993): consensus positives, outlier negatives, a
down-weighted borderline band, plus synthetic decoys and hard near-miss decoys. A
consensus/outlier classifier (gradient-boosted trees, falling back to logistic or a
distance-to-consensus heuristic) was trained on the fingerprint only — identity / MSA
membership are deliberately excluded as features so they cannot leak the
real-vs-decoy label. <b>Validation</b> is by <b>subfamily / phylogenetic holdout</b>:
one homolog group is withheld, the model refit on the rest, and AUROC measured on the
held-out group's consensus-vs-decoy poses — high transfer to an unseen subfamily
guards against evolutionary leakage and consensus circularity.</p>
<p><b>Visualisation.</b> Distributions and the contact map are rendered with Chart.js;
the high-frequency WT-ensemble contact residues are shown on the WT complex with
3Dmol.js (Rego &amp; Koes 2015). <b>All quantities are computational predictions</b>
(Boltz-ensemble-derived), intended as a holdout-validated prior over interaction
geometry for downstream ranking and mutant design — not experimental measurements.</p>
<h2>Analysis conditions</h2>
<table><tbody>%%CONDROWS%%</tbody></table>
<h2>References</h2>
<p class="cite">
Passaro et al. (2025) — Boltz-2 (diffusion co-folding + affinity).
Abramson et al. (2024) <i>Nature</i> 630:493 — AlphaFold3 (diffusion ensemble, confidence).
Jumper et al. (2021) <i>Nature</i> 596:583 — AlphaFold2 (pLDDT / PAE).
Rousseeuw &amp; Croux (1993) <i>JASA</i> 88:1273 — robust median/MAD scale.
Leys et al. (2013) <i>J. Exp. Soc. Psychol.</i> 49:764 — MAD-based outlier detection.
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
var METRIC_INFO={
 confidence_score:{t:'confidence_score',d:'Boltz 0.8·pLDDT+0.2·ipTM per sample.'},
 ligand_iptm:{t:'ligand ipTM',d:'protein–ligand interface confidence per sample (>0.8 strong).'},
 complex_plddt:{t:'complex pLDDT',d:'mean local confidence per sample.'},
 affinity_pred_value:{t:'affinity (log10 IC50/µM)',d:'Boltz-2 affinity head, ranking signal only.'},
 ptm:{t:'pTM',d:'global fold confidence per sample.'},
 iptm:{t:'ipTM',d:'overall interface confidence per sample.'},
 affinity_probability_binary:{t:'P(binder)',d:'Boltz-2 hit-vs-decoy probability per sample.'}
};
function barOpts(xt,yt){return {responsive:true,maintainAspectRatio:false,
  plugins:{legend:{display:false}},
  scales:{x:{title:{display:true,text:xt,color:MUT},ticks:{color:MUT,maxTicksLimit:10},grid:{display:false}},
   y:{title:{display:true,text:yt,color:MUT},ticks:{color:MUT},grid:{color:GRID},beginAtZero:true}}};}
function mkHist(canvasId,noteId,metric,color){
  var h=R.pose_hists[metric];if(!h){var c=document.getElementById(canvasId);if(c)c.parentNode.style.display='none';return;}
  var info=METRIC_INFO[metric]||{t:metric,d:''},st=h.stats;
  new Chart(document.getElementById(canvasId),{type:'bar',
    data:{labels:h.labels,datasets:[{data:h.counts,backgroundColor:color}]},
    options:barOpts(info.t,'poses')});
  var nn=document.getElementById(noteId);
  if(nn)nn.innerHTML='<b>'+info.t+'</b> over '+st.n+' poses — median <b>'+st.median+
    '</b> (IQR '+st.q1+'–'+st.q3+', range '+st.min+'–'+st.max+'). '+info.d;
}
function mkFreq(){
  var h=R.freq_hist;
  new Chart(document.getElementById('freqChart'),{type:'bar',
    data:{labels:h.labels,datasets:[{data:h.counts,backgroundColor:'#378ADD'}]},
    options:barOpts('ensemble contact frequency','(residue, ligand-atom) pairs')});
}
function mkResMap(){
  var rm=R.res_map;if(!rm||!rm.length){var c=document.getElementById('resmapChart');if(c)c.parentNode.style.display='none';return;}
  var labels=rm.map(function(d){return d.residue_index;});
  var vals=rm.map(function(d){return d.max_freq;});
  var cols=rm.map(function(d){return d.max_freq>=0.5?'#1D9E75':(d.max_freq>=0.2?'#BA7517':'rgba(127,119,221,.45)');});
  new Chart(document.getElementById('resmapChart'),{type:'bar',
    data:{labels:labels,datasets:[{data:vals,backgroundColor:cols}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},
      tooltip:{callbacks:{title:function(it){return 'residue '+it[0].label;}}}},
     scales:{x:{title:{display:true,text:'residue index',color:MUT},ticks:{color:MUT,maxTicksLimit:14},grid:{display:false}},
      y:{min:0,max:1,title:{display:true,text:'max contact frequency',color:MUT},ticks:{color:MUT},grid:{color:GRID}}}}});
}
function mkComp(){
  var c=R.composition;
  new Chart(document.getElementById('compChart'),{type:'bar',
    data:{labels:c.map(function(d){return d.label;}),
      datasets:[{data:c.map(function(d){return d.count;}),backgroundColor:c.map(function(d){return d.color;})}]},
    options:barOpts('training-pose class','poses')});
}
function pinAuroc(){var p=document.getElementById('aurocpin');if(!p)return;
  var v=%%AUROCPCT%%;if(v==='—'||isNaN(v)){p.style.display='none';return;}
  p.style.left=v+'%';}

// ---- 3D viewer (high-frequency WT-ensemble contact residues on the WT complex) ----
var V=null,MDL=null,_spin=false,_pocket=true,_psurf=null,_labels=false;
var curRep='cartoon';
var RESI=(R.conserved_resis||[]);
function resiSel(){return {resi:RESI,hetflag:false};}
function applyStyle(){
  if(!V)return;
  V.setStyle({},{});
  var ps={};
  if(curRep==='cartoon')ps.cartoon={color:'#cbc9c2',opacity:0.55};
  else if(curRep==='stick')ps.stick={radius:0.1,colorscheme:'whiteCarbon'};
  else ps.cartoon={style:'trace',thickness:0.3,color:'#cbc9c2'};
  V.setStyle({hetflag:false},ps);
  if(_pocket&&RESI.length){V.addStyle(resiSel(),{stick:{radius:0.2,colorscheme:'orangeCarbon'}});}
  V.setStyle({hetflag:true},{stick:{radius:0.22,colorscheme:'cyanCarbon'}});
  V.render();
}
function mark(b){if(b&&b.parentNode){b.parentNode.querySelectorAll('.vb').forEach(function(x){x.classList.remove('on');});b.classList.add('on');}}
function setRep(r,b){curRep=r;mark(b);applyStyle();}
function togglePocket(b){_pocket=!_pocket;if(b)b.classList.toggle('on');applyStyle();}
function togglePsurf(b){if(!V)return;
  if(_psurf!=null){if(_psurf!=='p')V.removeSurface(_psurf);_psurf=null;if(b)b.classList.remove('on');V.render();}
  else{if(b)b.classList.add('on');_psurf='p';
   Promise.resolve(V.addSurface($3Dmol.SurfaceType.SES,{opacity:0.6,colorscheme:'whiteCarbon'},
     {within:{distance:5,sel:{hetflag:true}},byres:true,hetflag:false},{})).then(function(r){
       var id=(r&&r.surfid!==undefined)?r.surfid:r;if(_psurf==='p')_psurf=id;else if(id!=null)V.removeSurface(id);V.render();});}}
function toggleLabels(b){if(!V)return;_labels=!_labels;if(b)b.classList.toggle('on');
  if(_labels)MDL.selectedAtoms({and:[{hetflag:false},{atom:'CA'},{resi:RESI}]}).forEach(function(a){a._rl=V.addLabel(a.resn+a.resi,{position:a,fontSize:10,backgroundColor:'black',backgroundOpacity:0.6});});
  else MDL.selectedAtoms({}).forEach(function(a){if(a._rl){V.removeLabel(a._rl);a._rl=null;}});
  V.render();}
function viewWhole(){if(V){applyStyle();V.zoomTo();V.render();}}
function viewPocket(){if(V){if(RESI.length)V.zoomTo(resiSel());else V.zoomTo({hetflag:true});V.zoom(1.4);V.render();}}
function spin(b){if(!V)return;_spin=!_spin;V.spin(_spin?'y':false);if(b)b.classList.toggle('on');}
function snap(){if(V){var a=document.createElement('a');a.href=V.pngURI();a.download='%%TARGETID%%_interaction_pocket.png';a.click();}}
function initViewer(){
  var box=document.getElementById('viewer');
  if(R.has_pdb!==true){if(box)box.innerHTML='<p style="padding:1rem;color:var(--mut);font-size:13px">WT complex PDB not available — the 3D pocket view is omitted for this run. Sections 1–5 are unaffected.</p>';return;}
  if(!window.$3Dmol){box.innerHTML='<p style="padding:1rem;color:#C0392B;font-size:13px">3Dmol.js failed to load (offline?). The charts above still work.</p>';return;}
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
  if(RESI.length)viewPocket();else V.zoomTo();
  V.render();
}
function start(){
  pinAuroc();
  if(window.Chart){
    var have=R.pose_metrics_have||Object.keys(R.pose_hists||{});
    var cols=['#378ADD','#1D9E75','#BA7517','#9b59b6'];
    var pri=['confidence_score','ligand_iptm','complex_plddt','affinity_pred_value'];
    var ordered=pri.filter(function(m){return R.pose_hists[m];});
    Object.keys(R.pose_hists||{}).forEach(function(m){if(ordered.indexOf(m)<0)ordered.push(m);});
    for(var i=0;i<4;i++){
      if(i<ordered.length)mkHist('h'+i,'h'+i+'note',ordered[i],cols[i]);
      else{var c=document.getElementById('h'+i);if(c)c.parentNode.style.display='none';
        var n=document.getElementById('h'+i+'note');if(n)n.style.display='none';}
    }
    mkFreq();mkResMap();mkComp();
  }
  setTimeout(initViewer,150);
}
if(document.readyState!=='loading')start();else document.addEventListener('DOMContentLoaded',start);
</script></body></html>
"""
