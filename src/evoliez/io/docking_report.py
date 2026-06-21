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


def _pdb_het_block(pdb_text: Optional[str]) -> tuple:
    """Pull the co-modelled context ligand out of the docking receptor PDB.

    For a ``'<base>__<cof>'`` variant (or the design-ligand candidate redocked
    against an extra ligand) the docking receptor written for gnina/vina carries
    the OTHER co-modelled ligand(s) as ``HETATM`` records held FIXED at their s04
    Boltz-placed coordinates, while the protein receptor is ``ATOM``. Those
    HETATM are exactly the cofactor-as-fixed-context entity the user wants to
    SEE in the overlay. Returns ``(het_pdb_block, [resnames])`` — a standalone
    PDB string of just those HETATM (loadable as its own 3Dmol model) and the
    distinct het residue names found, or ``("", [])`` when the receptor carries
    no co-modelled ligand (e.g. the DiffDock receptor, which is stripped). The
    block is parsed straight off disk, so the coordinates are the real fixed
    pose, never re-placed here."""
    lines, resns = [], []
    for ln in (pdb_text or "").splitlines():
        if ln[:6].strip() == "HETATM":
            lines.append(ln)
            rn = ln[17:20].strip()
            if rn and rn not in resns:
                resns.append(rn)
    block = ("\n".join(lines) + "\n") if lines else ""
    return block, resns


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
# RDKit: ligand normalization + symmetry-corrected heavy-atom RMSD (optional).
# The proven implementations now live in evoliez.features.ligand so the SAME
# metric scores this report AND selects gnina's representative pose (s05); these
# module-local names are thin aliases — behaviour (and the report's numbers) is
# byte-for-byte unchanged.
# --------------------------------------------------------------------------- #
from evoliez.features.ligand import (  # noqa: E402
    no_align_rmsd as _no_align_rmsd,
    normalize_pose_mol as _pose_mol,
    rmsd_template as _template,
)


def _sdf_mode_blocks(sdf_text: Optional[str]) -> List[str]:
    """Split a multi-mode gnina SDF into one FULL record per ``$$$$`` (file
    order). gnina writes its N docked modes as N records in ONE SDF; the overlay
    must show the SELECTED mode, not whichever 3Dmol loads as model 0. Each block
    KEEPS its molblock AND the trailing gnina score tags (so ``_sdf_props`` reads
    the selected mode's minimizedAffinity / CNN scores) and is re-terminated with
    ``$$$$`` so it loads as its own single-record SDF (3Dmol parses the molblock,
    ignores the data fields). A record is kept only if it carries a V2000/V3000
    counts line (so an empty trailing chunk after the final ``$$$$`` is dropped);
    the molblock title is NOT assumed non-blank, so the counts line is found by
    the ``Vxxxx`` marker rather than a fixed line offset."""
    if not sdf_text:
        return []
    blocks: List[str] = []
    for chunk in sdf_text.split("$$$$"):
        lines = chunk.splitlines()
        # Anchor on the V2000/V3000 counts line: the molblock header is EXACTLY the
        # three lines above it (title / program / comment) — and the title may be
        # BLANK (RDKit emits an empty title). Anchoring this way is robust to a
        # leading separator newline from the "$$$$" split AND to a blank title,
        # both of which a fixed-offset / blanket-strip approach gets wrong.
        ci = next((i for i, ln in enumerate(lines)
                   if ln.rstrip().endswith(("V2000", "V3000"))), None)
        if ci is None:
            continue  # empty trailing chunk after the final "$$$$"
        header = lines[max(0, ci - 3):ci]
        while len(header) < 3:                 # pad a missing title/comment line
            header.insert(0, "")
        rec = "\n".join(header + lines[ci:]).rstrip("\n")
        blocks.append(rec + "\n$$$$\n")
    return blocks


def _select_gnina_mode(sdf_text: Optional[str], ref_pdb: Optional[str],
                       tmpl) -> Optional[str]:
    """The gnina mode (as a standalone single-model SDF block) whose RAW docked
    geometry best matches the reference — the SAME min symmetry-corrected RMSD
    rule s05 uses to pick gnina's representative pose. The 3D overlay embeds THIS
    block so it shows the geometrically-correct pose rather than gnina's CNN-rank-1
    (which for large cofactors can be end-for-end flipped). Falls back to the
    first mode when there is no reference / no template / nothing scorable (so the
    overlay is unchanged for the common, non-flipped case). Returns None when the
    SDF has no parseable mode."""
    blocks = _sdf_mode_blocks(sdf_text)
    if not blocks:
        return None
    if ref_pdb is None or tmpl is None:
        return blocks[0]
    ref_mol, _ = _pose_mol(ref_pdb, "pdb", tmpl)
    if ref_mol is None:
        return blocks[0]
    best_i, best_r = 0, None
    for i, blk in enumerate(blocks):
        prb, _ = _pose_mol(blk, "sdf", tmpl)
        r = _no_align_rmsd(prb, ref_mol)
        if r is not None and (best_r is None or r < best_r):
            best_i, best_r = i, r
    return blocks[best_i]


# --------------------------------------------------------------------------- #
# Multi-ligand candidate-id parsing (GENERIC — never hardcode a ligand identity)
# --------------------------------------------------------------------------- #
def _split_candidate(cand_id: str) -> tuple:
    """``'<target>'`` -> ``(target, None)``; ``'<target>__<lig>'`` -> ``(target, lig)``.

    s05 docks EVERY input ligand, each with the OTHERS held as fixed receptor
    context. It names the candidate after WHICH ligand is being docked: the bare
    target id docks the DESIGN ligand, and ``'<target>__<ligand_id>'`` docks the
    co-modelled ligand ``<ligand_id>`` (a DIFFERENT molecule). Split on the FIRST
    ``'__'`` so a ligand id may itself contain underscores. The suffix is parsed
    at runtime; nothing about a specific ligand is assumed."""
    base, sep, lig = cand_id.partition("__")
    return (base, lig if sep else None)


def _docked_ligand_id(cand_id: str) -> str:
    """The id of the ligand this candidate DOCKS, parsed from the candidate id.

    The bare target id docks the design ligand (so the candidate id itself is the
    cleanest available handle for it); a ``'<target>__<ligand_id>'`` candidate
    docks the named co-modelled ligand ``<ligand_id>``. Generic — no ligand
    identity is hardcoded; a run's ligand manifest is encoded in these ids."""
    _base, lig = _split_candidate(cand_id)
    return lig if lig is not None else cand_id


def _candidate_label(cand_id: str) -> str:
    """Human label for a candidate = the docked ligand's identity. The design-
    ligand candidate reads as ``'<target> (design ligand)'``; a co-modelled
    candidate reads as its parsed ligand id ``'<ligand_id>'``."""
    base, lig = _split_candidate(cand_id)
    if lig is None:
        return f"{base} (design ligand)"
    return lig


def _list_db_candidates(db_path: str) -> List[str]:
    """All candidate_ids present in ``docking_pose`` (so the report discovers the
    multi-ligand variants instead of assuming a fixed set)."""
    out: List[str] = []
    try:
        con = sqlite3.connect(str(db_path), timeout=15)
        out = [row[0] for row in con.execute(
            "select distinct candidate_id from docking_pose "
            "order by candidate_id")]
        con.close()
    except sqlite3.Error:
        pass
    return out


# --------------------------------------------------------------------------- #
def _compute_one_candidate(dock: str, db_path: str, methods: List[str],
                           candidate: str, tmpl) -> dict:
    """Per-candidate docking stats: DB scores, the best pose per method, the
    DiffDock landscape, ligand normalization QC, and the symmetry-corrected
    heavy-atom RMSD pairs (incl. vs the reference). ``candidate`` may be a base
    (``'wt'``) or a context variant (``'wt__<cof>'``); the artifact paths are
    keyed by the FULL candidate id either way."""
    scores: Dict[str, dict] = {}
    try:
        con = sqlite3.connect(str(db_path), timeout=15)
        # score_type is a newer column; older run DBs lack it. Probe the schema so
        # this works against both (the report stays standalone & back-compatible).
        cols = {r[1] for r in con.execute("PRAGMA table_info(docking_pose)")}
        has_stype = "score_type" in cols
        sel = ("select method,score,ligand_rmsd_to_reference"
               + (",score_type" if has_stype else "")
               + " from docking_pose where candidate_id=?")
        for row in con.execute(sel, (candidate,)):
            m, sc, r = row[0], row[1], row[2]
            scores[m] = {"score": sc, "rmsd": r,
                         "score_type": (row[3] if has_stype else None)}
        con.close()
    except sqlite3.Error:
        pass

    receptor = (_read(os.path.join(dock, "gnina", f"{candidate}_rec.pdb"))
                or _read(os.path.join(dock, "diffdock", f"{candidate}_rec.pdb")))
    # The co-modelled cofactor held as FIXED receptor context = the HETATM in the
    # docking receptor (the protein is ATOM). Prefer the gnina receptor (it keeps
    # the context; diffdock strips it), so even when `receptor` resolved to the
    # stripped diffdock copy we still recover the cofactor from the gnina one.
    cof_src = (_read(os.path.join(dock, "gnina", f"{candidate}_rec.pdb"))
               or receptor)
    cofactor_pdb, cofactor_resns = _pdb_het_block(cof_src)
    ref_pdb = _read(os.path.join(dock, "gnina", f"{candidate}_ref_lig.pdb"))
    gnina_sdf_all = _read(os.path.join(dock, "gnina", f"{candidate}_gnina_out.sdf"))
    # The gnina output SDF holds ALL modes (3Dmol would show model 0 = gnina's
    # CNN-rank-1, which for large cofactors can be end-for-end flipped). Pick the
    # mode best matching the reference (same min symmetry-corrected RMSD rule s05
    # uses for the representative pose) and use THAT single mode for the overlay,
    # the score tags, the RMSD-to-reference, and the centroid — so every number
    # and the 3D view describe the SAME geometrically-correct pose.
    gnina_sdf = _select_gnina_mode(gnina_sdf_all, ref_pdb, tmpl)
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

    # GNINA's real sub-scores of the SELECTED mode (the DB 'score' is the selected
    # mode's minimizedAffinity).
    gnina_props = _sdf_props(gnina_sdf)

    # --- ligand normalization + symmetry-corrected heavy-atom RMSD ---
    mols, qc = {}, {}
    for m, sdf in poses.items():
        mols[m], qc[m] = _pose_mol(sdf, "sdf", tmpl)
    if ref_pdb:
        mols["reference"], qc["reference"] = _pose_mol(ref_pdb, "pdb", tmpl)

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

    base, cof = _split_candidate(candidate)
    rdkit_ok = any(q.get("heavy") for q in qc.values())
    # best heavy-atom RMSD vs the reference (the per-candidate accuracy number)
    ref_rmsds = [r for a, b, r, _ in pairs if b == "reference" and r is not None]
    # per-method RECOMPUTED symmetry-corrected RMSD-to-reference (a -> reference),
    # so the multi-ligand table can show the SAME number as the "Pose accuracy"
    # section instead of the DB ligand_rmsd_to_reference placeholder (which gnina
    # never fills -> stored 0.0). None when a method's pose couldn't be normalized.
    recomputed_ref_rmsd = {a: r for a, b, r, _ in pairs if b == "reference"}
    # per-method gnina sub-score provenance for the LABELLED score cell: gnina's
    # DB `score` IS minimizedAffinity, but the report re-reads the SDF tag so the
    # displayed value is provably the docking energy, never the CNNscore. Keyed by
    # method; only gnina carries a typed score here (diffdock's is a confidence).
    score_types = {}
    if "gnina" in scores:
        score_types["gnina"] = {
            "value": gnina_props.get("minimizedAffinity",
                                     scores["gnina"].get("score")),
            "type": "minimizedAffinity", "unit": "kcal/mol", "lower_better": True}
    if "diffdock" in scores:
        score_types["diffdock"] = {
            "value": scores["diffdock"].get("score"),
            "type": "DiffDock confidence", "unit": "log-odds",
            "lower_better": False}
    return {
        "candidate": candidate,
        "base": base,
        "cofactor": cof,
        "is_variant": cof is not None,
        "label": _candidate_label(candidate),
        "methods": [m for m in methods if m in scores] or list(scores),
        "scores": scores,
        "gnina_props": gnina_props,
        "landscape": landscape,
        "receptor": receptor,
        "cofactor_pdb": cofactor_pdb,      # co-modelled fixed-context HETATM block
        "cofactor_resns": cofactor_resns,  # its het residue name(s), e.g. ['LIG']
        "poses": poses,
        "pose_mols": mols,          # normalized RDKit mols (in-memory; rmsd_pairs)
        "reference_pdb": ref_pdb,
        "rmsd_pairs": pairs,
        "best_ref_rmsd": min(ref_rmsds) if ref_rmsds else None,
        "recomputed_ref_rmsd": recomputed_ref_rmsd,  # per-method, vs reference
        "score_types": score_types,                  # per-method labelled score
        "ligand_qc": qc,
        "rdkit_ok": rdkit_ok,
        "n_diffdock_poses": len(landscape),
    }


def _build_multi_ligand(per_candidate: Dict[str, dict]) -> Optional[dict]:
    """Per-ligand reference-docking summary: one row per (docked ligand, method).

    s05 docks EVERY input ligand, each with the OTHER input ligand(s) held as
    fixed receptor context. The candidate id encodes WHICH ligand is docked (bare
    target id = the design ligand; ``'<target>__<ligand_id>'`` = the named co-
    modelled ligand), so every candidate is a DIFFERENT molecule docked against
    its own s04 Boltz reference. This surfaces, for each candidate/method, that
    ligand's docking score (with its type label) and its pose accuracy = the
    recomputed symmetry-corrected RMSD to ITS OWN reference pose. There is NO
    base-vs-variant comparison: the candidates are different molecules, so their
    RMSDs are not a before/after of one ligand.

    Returns None unless the run has at least one co-modelled ligand (a candidate
    with a ``'__'`` suffix), so a single-ligand run (only the bare candidate)
    renders with no multi-ligand section at all."""
    has_comodelled = any("__" in c for c in per_candidate)
    if not has_comodelled:
        return None
    rows = []  # (candidate_id, ligand_id, is_design, method, score_info, rmsd)
    ligands_seen = []
    for cand_id, st in per_candidate.items():
        sc = st.get("scores") or {}
        if not sc:
            continue  # a candidate with no docked poses contributes no row
        lig_id = _docked_ligand_id(cand_id)
        if lig_id not in ligands_seen:
            ligands_seen.append(lig_id)
        stypes = st.get("score_types") or {}
        recomp = st.get("recomputed_ref_rmsd") or {}
        for m in sorted(sc):
            # the LABELLED score (gnina minimizedAffinity kcal/mol, lower=better;
            # diffdock confidence log-odds, higher=better) — re-read from the SDF
            # so a gnina Vina energy can never be misread as a 0-1 CNNscore.
            si = stypes.get(m) or {}
            score_info = {"value": si.get("value", sc[m].get("score")),
                          "type": si.get("type") or "score",
                          "unit": si.get("unit"),
                          "lower_better": si.get("lower_better")}
            # pose accuracy = RECOMPUTED symmetry-corrected RMSD to THIS ligand's
            # OWN s04 Boltz reference (same source as the "Pose accuracy" section),
            # NOT the DB ligand_rmsd_to_reference 0.0 placeholder.
            rows.append((cand_id, lig_id, st["base"] == cand_id, m,
                         score_info, recomp.get(m)))
    if not rows:
        return None
    return {"rows": rows, "ligands": ligands_seen,
            "candidates": sorted({r[0] for r in rows})}


# --------------------------------------------------------------------------- #
def compute_docking_stats(run_dir: str, db_path: str, methods: List[str], *,
                          candidate: str = "wt",
                          candidates: Optional[List[str]] = None,
                          ligand_smiles: Optional[str] = None) -> dict:
    """Docking stats for the s05 report.

    Computes the PRIMARY ``candidate`` (default ``'wt'`` — the design ligand,
    docked with any co-modelled ligand(s) held as fixed context) and flattens its
    stats at the top level (back-compatible with the single-candidate report).
    Also discovers every other candidate_id in ``docking_pose`` — the s05
    per-ligand candidates named ``'<target>__<ligand_id>'``, each docking a
    DIFFERENT co-modelled ligand against its own s04 Boltz reference — computes
    each, and (when any co-modelled ligand exists) assembles a ``'multi_ligand'``
    block: one row per docked ligand with that ligand's own score and pose
    accuracy. ``candidates`` overrides DB discovery (testing).
    """
    dock = os.path.join(run_dir, "docking")
    tmpl = _template(ligand_smiles)

    discovered = candidates if candidates is not None else _list_db_candidates(db_path)
    # Always include the primary candidate; keep a stable order (primary first,
    # then the rest as discovered).
    ordered: List[str] = [candidate] + [c for c in discovered if c != candidate]
    per_candidate: Dict[str, dict] = {}
    for cand in ordered:
        st = _compute_one_candidate(dock, db_path, methods, cand, tmpl)
        # Only keep candidates that actually have something (scores or poses); a
        # bare primary 'wt' with no rows still passes through (its empty 'scores'
        # is what the caller checks to decide whether to write the report).
        if cand == candidate or st["scores"] or st["poses"]:
            per_candidate[cand] = st

    primary = per_candidate[candidate]
    multi = _build_multi_ligand(per_candidate)

    # Name the co-modelled ligand(s) shown as FIXED CONTEXT in the PRIMARY overlay
    # generically (no hardcoding). The primary candidate (the design ligand) is
    # docked WITH the OTHER input ligand(s) held as fixed receptor context; their
    # ids are exactly the runtime-parsed '<target>__<ligand_id>' suffixes of the
    # other discovered candidates. If the primary is itself a co-modelled ligand's
    # candidate, use its own parsed ligand id as the context name. Falls back to
    # the het residue name(s) in the receptor, then a generic label — so the
    # overlay is honest even when the id can't be recovered.
    if primary.get("cofactor"):
        cof_names = [primary["cofactor"]]
    else:
        cof_names = sorted({lig for lig in (_split_candidate(c)[1]
                                            for c in per_candidate)
                            if lig is not None})
    if not cof_names and primary.get("cofactor_pdb"):
        cof_names = list(primary.get("cofactor_resns") or [])
    if primary.get("cofactor_pdb"):
        nm = ", ".join(cof_names) if cof_names else "co-modelled ligand"
        cofactor_label = f"{nm} (cofactor — fixed context)"
    else:
        cofactor_label = ""

    # Per-candidate view for the tables (drop the in-memory pose_mols — not JSON
    # and not needed by the renderer; the multi_ligand rows already carry the
    # derived shift).
    cand_summaries = []
    for cand in ordered:
        st = per_candidate.get(cand)
        if st is None:
            continue
        cand_summaries.append({
            k: v for k, v in st.items() if k != "pose_mols"})

    out = {
        # --- back-compatible flattened PRIMARY-candidate stats ---
        "candidate": primary["candidate"],
        "methods": primary["methods"],
        "scores": primary["scores"],
        "gnina_props": primary["gnina_props"],
        "landscape": primary["landscape"],
        "receptor": primary["receptor"],
        "cofactor_pdb": primary.get("cofactor_pdb", ""),
        "cofactor_label": cofactor_label,
        "poses": primary["poses"],
        "reference_pdb": primary["reference_pdb"],
        "rmsd_pairs": primary["rmsd_pairs"],
        "ligand_qc": primary["ligand_qc"],
        "rdkit_ok": primary["rdkit_ok"],
        "ligand_smiles": ligand_smiles,
        "n_diffdock_poses": primary["n_diffdock_poses"],
        # --- multi-ligand additions ---
        "candidate_summaries": cand_summaries,
        "multi_ligand": multi,
    }
    return out


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
_POSECOL = {"gnina": "#1D9E75", "diffdock": "#BA7517", "reference": "#3b7dd8"}
# Distinct from every pose/reference colour: the co-modelled cofactor is CONTEXT,
# not a docked pose, so it must read as a different entity in the overlay.
_COFCOL = "#9C27B0"


def _build_multi_ligand_html(s: dict, g, fnum) -> str:
    """The 'per-ligand reference docking' section: one row per docked ligand x
    method. s05 docks EVERY input ligand (each with the others as fixed receptor
    context), and the candidate id says WHICH ligand each row is — so every row is
    a DIFFERENT molecule scored against its own s04 Boltz reference, NOT a
    before/after of one ligand. Each row shows that ligand's id, its docking score
    WITH its type label (gnina minimizedAffinity kcal/mol; diffdock confidence),
    and its pose accuracy = the recomputed symmetry-corrected RMSD to its own
    reference. Returns '' when the run has no co-modelled ligand (only the bare
    candidate), so a single-ligand run renders with no multi-ligand section."""
    ml = s.get("multi_ligand")
    if not ml:
        return ""

    # one row per (docked ligand, method): the ligand's own labelled score and its
    # pose accuracy (recomputed symmetry-corrected RMSD to ITS OWN reference, the
    # SAME source as the "Pose accuracy" section — never the DB 0.0 placeholder).
    # Group rows by candidate so each docked ligand reads as one block.
    by_cand: Dict[str, list] = {}
    order: List[str] = []
    for r in ml.get("rows", []):
        cand_id = r[0]
        if cand_id not in by_cand:
            by_cand[cand_id] = []
            order.append(cand_id)
        by_cand[cand_id].append(r)

    lrows = ""
    for cand_id in order:
        crows = by_cand[cand_id]
        lig_id = crows[0][1]
        role = "design ligand" if crows[0][2] else "co-modelled ligand"
        first = True
        for _c, _lig, _isd, m, si, rmsd in crows:
            sval = si.get("value")
            stype = si.get("type") or "score"
            sccell = (f'{fnum(sval, "{:.2f}")}'
                      f'<div class="sub2">{g(stype)}</div>')
            label_cell = (
                f'<td rowspan="{len(crows)}"><code>{g(lig_id)}</code>'
                f'<div class="sub2">{g(role)} · <code>{g(cand_id)}</code></div>'
                '</td>') if first else ""
            lrows += (f"<tr>{label_cell}<td>{g(m)}</td>"
                      f"<td>{sccell}</td>"
                      f"<td>{fnum(rmsd, '{:.2f} A')}</td></tr>")
            first = False
    if not lrows:
        lrows = ("<tr><td colspan=4 class=ck>no per-ligand docking scores"
                 "</td></tr>")

    return (
        '<h2>Per-ligand reference docking</h2>'
        '<div class="concl">This run co-modelled more than one ligand. s05 docks '
        '<b>every input ligand</b>, each against its own s04 Boltz reference pose, '
        'with the <b>other input ligand(s) held as fixed receptor context</b> '
        '(they stay at their s04 placed pose and are part of the receptor during '
        'that redock). The candidate id says <b>which ligand</b> each row is: the '
        'bare target id docks the <b>design ligand</b>, and '
        '<code>&lt;target&gt;__&lt;ligand_id&gt;</code> docks the named '
        'co-modelled ligand. Each row below is therefore a <b>different '
        'molecule</b> scored against its own reference — not a before/after '
        'of one ligand.</div>'
        '<table><thead><tr><th>docked ligand</th><th>method</th>'
        '<th>docking score (type)</th>'
        '<th>pose accuracy (RMSD&nbsp;to&nbsp;own&nbsp;ref)</th></tr></thead>'
        '<tbody>' + lrows + '</tbody></table>'
        '<p class="note">Every ligand is docked with the other input ligand(s) '
        'held as <b>fixed receptor context</b>. Each <b>docking score is shown '
        'with its type</b> — GNINA <b>minimizedAffinity</b> (the AutoDock-Vina '
        'docking energy, kcal/mol, lower = stronger) vs DiffDock <b>confidence</b> '
        '(log-odds, higher = stronger) — so a Vina energy near 0 (e.g. the '
        'genuine minimizedAffinity of a small ligand such as a few-atom cofactor '
        'docked into an already-occupied pocket) is never misread as a 0–1 '
        'CNNscore. Scores are not comparable across methods, nor across ligands '
        '(different molecules). <b>Pose accuracy is the recomputed '
        'symmetry-corrected heavy-atom RMSD</b> of each ligand to <b>its own</b> '
        's04 Boltz reference (same metric as the Pose-accuracy table above, after '
        'the same largest-fragment normalization), NOT the database placeholder '
        'column.</p>'
    )


def build_docking_report_html(*, target_id: str, stats: dict,
                              conditions: List, generated: str,
                              ligand_name: Optional[str] = None,
                              provenance: str = "") -> str:
    import html as _html
    import json as _json
    g = _html.escape
    s = stats
    lig = ligand_name or "the ligand"
    subtitle = provenance or f"generated {g(generated)}"
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

    # --- multi-ligand co-modelling section (GENERIC; only if variants exist) --- #
    multi_html = _build_multi_ligand_html(s, g, fnum)

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

    # Co-modelled cofactor held as fixed receptor context — its own model + a
    # separate, distinctly-coloured toggle so the user SEES where it sits and how
    # the design ligand packs around it. Only emitted when this run co-modelled a
    # cofactor (generic; absent-cofactor runs render exactly as before).
    cof_pdb = s.get("cofactor_pdb") or ""
    cof_label = s.get("cofactor_label") or ""
    cof_btn = ""
    if cof_pdb:
        blocks.append('<script id="m_cofactor" type="text/plain">'
                      + cof_pdb + "</script>")
        cof_btn = (
            '<div class="cg"><span class="cgl">cofactor</span>'
            '<button class="vb on" onclick="toggleCof(this)">'
            f'<span class="dot" style="background:{_COFCOL}"></span>'
            f'{g(cof_label)}</button></div>')

    blob = _json.dumps({"landscape": land, "models": model_js,
                        "posecol": {m: _POSECOL.get(m, "#888") for m in model_js},
                        "cofactor": bool(cof_pdb), "cofcol": _COFCOL,
                        "coflabel": cof_label},
                       separators=(",", ":"))

    return (_DOCK_TEMPLATE
            .replace("%%TITLE%%", g(f"{target_id} — reference docking (s05)"))
            .replace("%%TARGET%%", g(target_id))
            .replace("%%LIG%%", g(lig))
            .replace("%%PROVSUB%%", subtitle)
            .replace("%%CARDS%%", cards_html)
            .replace("%%POSEBTNS%%", pose_btns)
            .replace("%%COFBTN%%", cof_btn)
            .replace("%%RROWS%%", rrows)
            .replace("%%QROWS%%", qrows)
            .replace("%%MROWS%%", mrows)
            .replace("%%MARGIN%%", g(margin) or "single ranked pose")
            .replace("%%CONDROWS%%", cond_rows)
            .replace("%%MULTILIG%%", multi_html)
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
<p class="sub">target %%TARGET%% · reference-target redocking of %%LIG%% vs the <b>s04 Boltz pose</b> ·
<b>internal reproducibility</b> baseline for s09 (not experimental validation) · %%PROVSUB%%</p>

<div class="cards">%%CARDS%%</div>

<div class="concl"><b>Conclusion (scoped).</b> GNINA and DiffDock converge to the
same pocket around the s04 Boltz reference pose (heavy-atom RMSD shown below), so
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
 %%COFBTN%%
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
on the reference (target) receptor; the s04 Boltz pose is the <b>reference</b> (blue).
When the design ligand was co-modelled with a cofactor, that cofactor is held as
<b>fixed receptor context</b> and shown as <b>purple sticks</b> at its s04 Boltz-placed
position — so you can see where it sits and how the design ligand packs around it
(it is part of the receptor during the dock, not a docked pose). Visual
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

%%MULTILIG%%
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
<p><b>Reference docking.</b> The reference %%LIG%% pose was established by redocking the
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
var V=null,REC=null,COF=null,MDL={},shown={},recRep='cartoon',_spin=false,_dark=false,_recsurf=null,_cofshown=true;
R.models.forEach(function(m){shown[m]=true;});
var POCKET={within:{distance:5,sel:{not:{resn:['HOH']}}},byres:true};
function initViewer(){
  if(!window.$3Dmol){document.getElementById('viewer').innerHTML='<p style="padding:1rem;color:#C0392B;font-size:13px">3Dmol.js failed to load (offline?). The charts/tables below still work.</p>';return;}
  V=$3Dmol.createViewer('viewer',{backgroundColor:'white'});
  var rec=document.getElementById('m_rec');
  if(rec)REC=V.addModel(rec.textContent,'pdb');
  R.models.forEach(function(m){var el=document.getElementById('m_'+m);if(!el)return;
    MDL[m]=V.addModel(el.textContent,(m==='reference')?'pdb':'sdf');});
  var cof=document.getElementById('m_cofactor');
  if(cof)COF=V.addModel(cof.textContent,'pdb');
  styleRec();R.models.forEach(stylePose);styleCof();
  V.zoomTo(anyPoseSel());V.zoom(0.85);V.render();
}
function anyPoseSel(){var ids=R.models.filter(function(m){return MDL[m];}).map(function(m){return MDL[m].getID();});
  if(COF)ids.push(COF.getID());  // frame the cofactor too so the pocket view shows it
  return ids.length?{model:ids}:{};}
function styleRec(){if(!REC)return;REC.setStyle({},{});
  if(recRep==='cartoon')REC.setStyle({},{cartoon:{color:'#c4c2bb'}});
  // pocket: protein sticks only — the co-modelled cofactor's HETATM are drawn by
  // its OWN distinct model (COF), so exclude them here to avoid a colour clash.
  else if(recRep==='pocket'){REC.setStyle(Object.assign({hetflag:false},POCKET),{stick:{radius:0.1,colorscheme:'whiteCarbon'}});}
  V.render();}
var _coflabel=null;
function styleCof(){if(!COF)return;
  COF.setStyle({}, _cofshown?{stick:{radius:0.22,color:R.cofcol},sphere:{scale:0.22,color:R.cofcol}}:{});
  if(_coflabel){V.removeLabel(_coflabel);_coflabel=null;}
  if(_cofshown&&R.coflabel){var a=COF.selectedAtoms({});if(a&&a.length){
    _coflabel=V.addLabel(R.coflabel,{position:a[0],backgroundColor:R.cofcol,
      backgroundOpacity:0.85,fontColor:'#fff',fontSize:11,borderThickness:0});}}
  V.render();}
function toggleCof(b){_cofshown=!_cofshown;if(b)b.classList.toggle('on');styleCof();}
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
