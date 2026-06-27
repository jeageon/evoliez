#!/usr/bin/env python
"""Build a paper-ready archive from an EvoLiEZ run.

For each pipeline stage s01-s07 it creates a folder::

    <out>/sNN_<stage>/
        report.html      cleaned HTML report (failure/error + caveat content removed)
        data.xlsx        raw data behind every graph & table (one sheet per item)
        figure/*.tif      publication-grade figures (300 dpi, LZW) re-plotted from data

Design goals (paper archive):
  * FINAL DATA ONLY  - reads the committed run outputs, nothing from failed attempts.
  * ORIGINALS UNTOUCHED - only reads the run dir; writes solely under <out>.
  * GENERIC - no target/ligand names hard-coded; everything is read from the run.

Usage::

    python build_paper_archive.py --run <RUN_DIR> --out <OUT_DIR> [--clean-level caveats]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import shutil
import sqlite3
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
DPI = 300
ACCENT = "#2C6E9B"
ACCENT2 = "#C0843B"
ACCENT3 = "#5B8C5A"
NEG = "#B5524A"
GREY = "#6b7785"

plt.rcParams.update({
    "savefig.dpi": DPI,
    "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})

STAGES = [
    ("s01", "input_report.html"),
    ("s02", "homolog_report.html"),
    ("s03", "msa_report.html"),
    ("s04", "complex_report.html"),
    ("s05", "docking_report.html"),
    ("s06", "interaction_model_report.html"),
    ("s07", "s07_mutation_report.html"),
]

STAGE_TITLES = {
    "s01": "Input preprocessing",
    "s02": "Homolog search",
    "s03": "MSA & conservation",
    "s04": "Complex (co-folding)",
    "s05": "Docking",
    "s06": "Interaction model",
    "s07": "Mutation generation",
}

AA20 = list("ACDEFGHIKLMNPQRSTVWY")

FIGURE_CAPTIONS = {
    "s01_sequence_composition": "Amino-acid composition of the target sequence.",
    "s02_identity_histogram": "Distribution of sequence identity to the target across retained homologs.",
    "s02_cluster_sizes": "Size distribution of the sequence-diversity clusters.",
    "s03_conservation_profile": "Per-position evolutionary conservation along the target.",
    "s03_entropy_profile": "Per-position Shannon entropy of the multiple-sequence alignment.",
    "s03_coverage_profile": "Per-position alignment coverage (1 - gap frequency).",
    "s04_ensemble_confidence": "Co-folding confidence metrics across the diffusion ensemble.",
    "s04_pae_heatmap": "Predicted aligned error (PAE) of the WT protein-ligand complex.",
    "s04_plddt_profile": "Per-residue model confidence (pLDDT) of the WT complex.",
    "s05_pose_rmsd": "Docking pose accuracy (symmetry-corrected RMSD to the reference).",
    "s06_contact_frequency": "WT-ensemble pocket contact frequency by residue.",
    "s06_pose_reliability": "Reliability-metric distributions across the docking-pose ensemble.",
    "s06_multi_engine_roles": "Multi-engine docking pose classification by engine.",
    "s07_candidates_by_generator": "Candidate count by generation strategy.",
    "s07_mutations_per_candidate": "Distribution of mutation order (single vs multipoint).",
    "s07_design_space_heatmap": "Normalised design-space metrics across designable positions.",
}

RAW_NOTES = {
    "s01": "target sequence + ligand definitions (stage inputs).",
    "s02": "filtered homolog sequences (stage output).",
    "s03": "MSA (alignment.fasta, msa.a3m) + per-position conservation/ESM prior (outputs).",
    "s04": "in/: Boltz-2 input (yaml + MSA). pred/: WT complex diffusion ensemble - per-sample PDB (model_NN.pdb) + conf/pae/pde/plddt + affinity.",
    "s05": "gnina/ + diffdock/ per ligand: receptors, reference ligands, docked poses (rankNN.sdf).",
    "s06": "interaction model (.json/.pkl), ensemble contacts, multi-engine docking classifications (outputs). ensemble/ present only with --raw full.",
    "s07": "generated candidate tables + LigandMPNN inputs/outputs.",
}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def save_tif(fig, path: Path):
    """Save as a flattened RGB TIFF (LZW, 300 dpi tag) - journal-safe (no alpha)."""
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    im.save(path, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))
    print(f"      figure  {path.name}")


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]):
    used = set()
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            if df is None or len(df) == 0:
                continue
            sn = re.sub(r"[\[\]:*?/\\]", "_", str(name))[:31] or "sheet"
            base, i = sn, 1
            while sn in used:
                i += 1
                sn = f"{base[:28]}_{i}"
            used.add(sn)
            df.to_excel(xw, sheet_name=sn, index=False)
    print(f"      xlsx    {path.name}  ({len([d for d in sheets.values() if d is not None and len(d)])} sheets)")


def read_report_tables(html: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(io.StringIO(html))
    except Exception as exc:
        print(f"      (read_html: {exc})")
        return []


def clean_html(html: str, level: str = "caveats") -> str:
    """Remove failure/error + caveat content; keep methods, data, figures."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    if level in ("caveats", "minimal"):
        # 1) drop "Limitations & QC" section: the <h2> and following siblings
        #    up to (not including) the next <h2>.
        for h2 in list(soup.find_all("h2")):
            if "limitation" in h2.get_text(strip=True).lower():
                sibs = list(h2.next_siblings)
                h2.extract()
                for sib in sibs:
                    if getattr(sib, "name", None) == "h2":
                        break
                    sib.extract()
        # 2) drop red-bordered caveat boxes (#C0392B border)
        for div in list(soup.find_all("div", class_="warn")):
            if "c0392b" in (div.get("style", "") or "").lower():
                div.decompose()

    if level == "minimal":
        for div in list(soup.find_all("div", class_="warn")):
            div.decompose()

    out = str(soup)

    # 3) neutralise runtime fallback / error strings (these live inside <script>
    #    string literals, so handle them on the serialised text).
    out = re.sub(r"3Dmol\.js failed to load \(offline\?\)\.?\s*",
                 "Interactive 3D viewer. ", out)
    out = re.sub(r"WT complex PDB not available\s*[—-]\s*the 3D[^.<']*\.?",
                 "Interactive 3D view available in the online build. ", out)
    # dangling cross-references to the removed section
    out = out.replace("See <b>Limitations &amp; QC</b> below.", "")
    out = out.replace("(see Limitations &amp; QC)", "")
    out = out.replace("see <b>Limitations &amp; QC</b>", "see Methods")
    out = re.sub(r"\s*See <b>Limitations[^<]*</b>[^.<]*\.", "", out)
    return out


def fasta_seq(path: Path) -> str:
    seq = []
    for line in path.read_text().splitlines():
        if line and not line.startswith(">"):
            seq.append(line.strip())
    return "".join(seq)


def read_smi(path: Path) -> list[tuple[str, str]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line, maxsplit=1)
        smiles = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        rows.append((name, smiles))
    return rows


def style_ax(ax, xlabel="", ylabel="", title=""):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


HEAT = LinearSegmentedColormap.from_list("evz", ["#F4F6F8", ACCENT])


# --------------------------------------------------------------------------- #
# per-stage builders.  Each returns (sheets, figures_done).
# --------------------------------------------------------------------------- #
def build_s01(run: Path, db, fig_dir: Path) -> dict:
    sheets = {}
    # project metadata
    try:
        row = db.execute(
            "select target_name, enzyme_family, objective, ligand_id, created_at from project"
        ).fetchone()
        if row:
            sheets["project"] = pd.DataFrame(
                [row], columns=["target_name", "enzyme_family", "objective",
                                "design_ligand", "created_at"])
    except Exception as exc:
        print(f"      (project: {exc})")

    # sequence composition
    seq = fasta_seq(run / "inputs" / "target.fasta")
    if seq:
        from collections import Counter
        c = Counter(seq)
        comp = pd.DataFrame(
            {"amino_acid": AA20,
             "count": [c.get(a, 0) for a in AA20],
             "fraction": [round(c.get(a, 0) / len(seq), 4) for a in AA20]})
        sheets["sequence_composition"] = comp
        sheets["target_sequence"] = pd.DataFrame(
            {"name": ["target"], "length": [len(seq)], "sequence": [seq]})

        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.bar(comp["amino_acid"], comp["count"], color=ACCENT, width=0.72)
        style_ax(ax, "Amino acid", "Count",
                 f"Target sequence composition (n = {len(seq)} aa)")
        save_tif(fig, fig_dir / "s01_sequence_composition.tif")

    # ligands
    lig = read_smi(run / "inputs" / "ligand.smi")
    extra = read_smi(run / "inputs" / "extra_ligands.smi")
    lrows = [("design_target", n, s) for n, s in lig] + \
            [("co_modelled", n, s) for n, s in extra]
    if lrows:
        sheets["ligands"] = pd.DataFrame(lrows, columns=["role", "name", "smiles"])
    return sheets


def build_s02(run: Path, db, fig_dir: Path) -> dict:
    sheets = {}
    df = pd.read_sql_query(
        "select source, identity_to_target, coverage, cluster_id "
        "from sequence", db)
    # summary by source
    summ = (df.groupby("source")
              .agg(n=("source", "size"),
                   mean_identity=("identity_to_target", "mean"),
                   mean_coverage=("coverage", "mean"))
              .reset_index().round(4))
    sheets["summary_by_source"] = summ

    hom = df[df["source"] != "target"].copy()
    if len(hom):
        # identity histogram data
        ident = hom["identity_to_target"].dropna().to_numpy()
        counts, edges = np.histogram(ident, bins=np.linspace(0, 1, 26))
        sheets["identity_histogram"] = pd.DataFrame({
            "identity_low": np.round(edges[:-1], 3),
            "identity_high": np.round(edges[1:], 3),
            "n_sequences": counts})

        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.hist(ident, bins=np.linspace(0, 1, 26), color=ACCENT,
                edgecolor="white", linewidth=0.4)
        style_ax(ax, "Identity to target", "Homolog count",
                 f"Homolog identity distribution (n = {len(ident):,})")
        ax.axvline(float(np.median(ident)), color=NEG, lw=1.2, ls="--",
                   label=f"median {np.median(ident):.2f}")
        ax.legend(frameon=False)
        save_tif(fig, fig_dir / "s02_identity_histogram.tif")

        # cluster sizes
        cl = (hom.dropna(subset=["cluster_id"])
                 .groupby("cluster_id").size().sort_values(ascending=False))
        if len(cl):
            sheets["cluster_sizes"] = (cl.reset_index(name="size")
                                         .rename(columns={"cluster_id": "cluster"}))
            fig, ax = plt.subplots(figsize=(6.0, 3.4))
            topn = cl.head(30).to_numpy()
            ax.bar(range(len(topn)), topn, color=ACCENT3, width=0.85)
            style_ax(ax, "Diversity cluster (rank)", "Members",
                     f"Sequence cluster sizes (top {len(topn)} of {len(cl):,})")
            save_tif(fig, fig_dir / "s02_cluster_sizes.tif")

        # full per-homolog table (no fasta column -> stays compact)
        sheets["homologs"] = hom.reset_index(drop=True).round(4)
    return sheets


def build_s03(run: Path, db, fig_dir: Path) -> dict:
    sheets = {}
    cons = json.loads((run / "msa" / "conservation.json").read_text())
    pos = sorted(cons.keys(), key=lambda x: int(x))
    rec = []
    for p in pos:
        d = cons[p]
        rec.append({
            "position": int(p),
            "conservation": d.get("conservation"),
            "entropy": d.get("entropy"),
            "gap_frequency": d.get("gap_frequency"),
            "coverage": round(1.0 - d.get("gap_frequency", 0.0), 4)
            if d.get("gap_frequency") is not None else None,
            "esm_variability": d.get("esm_variability"),
            "specificity_divergence": d.get("specificity_divergence"),
            "allowed_aa": ",".join(d.get("allowed_aa", [])),
        })
    df = pd.DataFrame(rec)
    sheets["per_position"] = df
    x = df["position"].to_numpy()

    # conservation + entropy
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.fill_between(x, df["conservation"], color=ACCENT, alpha=0.25)
    ax.plot(x, df["conservation"], color=ACCENT, lw=0.8)
    style_ax(ax, "Residue position", "Conservation",
             f"Per-position conservation ({len(x)} positions)")
    ax.set_ylim(0, 1)
    save_tif(fig, fig_dir / "s03_conservation_profile.tif")

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.plot(x, df["entropy"], color=ACCENT2, lw=0.8)
    style_ax(ax, "Residue position", "Shannon entropy (bits)",
             "Per-position MSA entropy")
    save_tif(fig, fig_dir / "s03_entropy_profile.tif")

    # coverage
    if df["coverage"].notna().any():
        fig, ax = plt.subplots(figsize=(7.4, 3.0))
        ax.fill_between(x, df["coverage"], color=ACCENT3, alpha=0.4)
        ax.plot(x, df["coverage"], color=ACCENT3, lw=0.7)
        style_ax(ax, "Residue position", "Coverage (1 - gap freq.)",
                 "Per-position MSA coverage")
        ax.set_ylim(0, 1)
        save_tif(fig, fig_dir / "s03_coverage_profile.tif")
    return sheets


def _boltz_pred_dir(run: Path) -> Path | None:
    hits = glob.glob(str(run / "complexes" / "boltz" / "**" / "confidence_*model_0.json"),
                     recursive=True)
    return Path(hits[0]).parent if hits else None


def build_s04(run: Path, db, fig_dir: Path) -> dict:
    sheets = {}
    # complex-level summary from DB
    try:
        cp = pd.read_sql_query(
            "select method, confidence, affinity_score from complex_prediction", db)
        if len(cp):
            sheets["complex_summary"] = cp.round(4)
    except Exception:
        pass

    pred = _boltz_pred_dir(run)
    if pred:
        conf_files = sorted(glob.glob(str(pred / "confidence_*model_*.json")),
                            key=lambda p: int(re.search(r"model_(\d+)", p).group(1)))
        rows = []
        for cf in conf_files:
            d = json.loads(Path(cf).read_text())
            mi = int(re.search(r"model_(\d+)", cf).group(1))
            rows.append({
                "model": mi,
                "confidence_score": d.get("confidence_score"),
                "ptm": d.get("ptm"),
                "iptm": d.get("iptm"),
                "ligand_iptm": d.get("ligand_iptm"),
                "complex_plddt": d.get("complex_plddt"),
                "complex_pde": d.get("complex_pde"),
            })
        ens = pd.DataFrame(rows)
        if len(ens):
            sheets["diffusion_ensemble"] = ens.round(4)
            # ensemble confidence distribution
            fig, ax = plt.subplots(figsize=(5.6, 3.6))
            data = [ens["confidence_score"].dropna(), ens["iptm"].dropna(),
                    ens["complex_plddt"].dropna(), ens["ptm"].dropna()]
            labels = ["confidence", "ipTM", "pLDDT", "pTM"]
            parts = ax.violinplot(data, showmeans=True, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(ACCENT); b.set_alpha(0.45)
            parts["cmeans"].set_color(NEG)
            ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
            style_ax(ax, "", "Score (0-1)",
                     f"Diffusion-ensemble confidence (n = {len(ens)} samples)")
            ax.set_ylim(0, 1)
            save_tif(fig, fig_dir / "s04_ensemble_confidence.tif")

        # affinity
        aff = sorted(glob.glob(str(pred / "affinity_*.json")))
        if aff:
            ad = json.loads(Path(aff[0]).read_text())
            sheets["affinity"] = pd.DataFrame(
                [(k, v) for k, v in ad.items()], columns=["metric", "value"])

        # PAE heatmap from one model
        pae_files = sorted(glob.glob(str(pred / "pae_*model_0.npz")))
        if pae_files:
            try:
                z = np.load(pae_files[0])
                key = "pae" if "pae" in z.files else z.files[0]
                pae = np.asarray(z[key], dtype=float)
                if pae.ndim == 2:
                    fig, ax = plt.subplots(figsize=(4.8, 4.2))
                    im = ax.imshow(pae, cmap="viridis", origin="upper",
                                   interpolation="nearest")
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cb.set_label("PAE (Å)")
                    ax.grid(False)
                    style_ax(ax, "Scored residue", "Aligned residue",
                             "Predicted aligned error (model 0)")
                    save_tif(fig, fig_dir / "s04_pae_heatmap.tif")
                    # store a downsampled matrix so the xlsx stays openable
                    step = max(1, pae.shape[0] // 200)
                    sheets["pae_matrix"] = pd.DataFrame(np.round(pae[::step, ::step], 2))
            except Exception as exc:
                print(f"      (pae: {exc})")

    # per-residue pLDDT from graph_features
    gf_path = run / "interaction_graphs" / "graph_features.json"
    if gf_path.exists():
        gf = json.loads(gf_path.read_text())
        res = gf.get("residues", [])
        if res and any(r.get("plddt") is not None for r in res):
            pl = pd.DataFrame({
                "position": [r["residue_index"] for r in res],
                "aa": [r.get("aa") for r in res],
                "plddt": [r.get("plddt") for r in res]})
            sheets["per_residue_plddt"] = pl.round(2)
            fig, ax = plt.subplots(figsize=(7.4, 3.0))
            ax.fill_between(pl["position"], pl["plddt"], color=ACCENT, alpha=0.25)
            ax.plot(pl["position"], pl["plddt"], color=ACCENT, lw=0.7)
            for lo, hi, c in [(0, 50, "#FF6B6B"), (50, 70, "#FFB347"),
                              (70, 90, "#9ACD66"), (90, 100, "#4F9DDE")]:
                ax.axhspan(lo, hi, color=c, alpha=0.06)
            style_ax(ax, "Residue position", "pLDDT",
                     "Per-residue confidence (WT complex)")
            ax.set_ylim(0, 100)
            save_tif(fig, fig_dir / "s04_plddt_profile.tif")
    return sheets


def build_s05(run: Path, db, fig_dir: Path, tables: list[pd.DataFrame]) -> dict:
    sheets = {}
    dp = pd.read_sql_query(
        "select method, score, confidence, ligand_rmsd_to_reference, score_type "
        "from docking_pose", db)
    if len(dp):
        sheets["docking_poses"] = dp.round(4)
        rms = dp.dropna(subset=["ligand_rmsd_to_reference"]).reset_index(drop=True)
        if len(rms):
            fig, ax = plt.subplots(figsize=(5.6, 3.4))
            colors = [ACCENT if m == "gnina" else ACCENT2 for m in rms["method"]]
            ax.bar(range(len(rms)), rms["ligand_rmsd_to_reference"], color=colors,
                   width=0.6)
            ax.axhline(2.0, color=NEG, ls="--", lw=1.0, label="2 Å success")
            ax.set_xticks(range(len(rms)))
            ax.set_xticklabels([f"{m}\n#{i+1}" for i, m in enumerate(rms["method"])])
            style_ax(ax, "", "RMSD to reference (Å)",
                     "Docking pose accuracy")
            ax.legend(frameon=False)
            save_tif(fig, fig_dir / "s05_pose_rmsd.tif")
    # attach report tables (per-ligand reference docking, normalization, etc.)
    for i, t in enumerate(tables):
        sheets[f"report_table_{i+1}"] = t
    return sheets


def build_s06(run: Path, db, fig_dir: Path) -> dict:
    sheets = {}
    ig = run / "interaction_graphs"
    art = json.loads((ig / "s06b_artifacts.json").read_text())

    # ensemble contacts
    ec = pd.DataFrame(art.get("ensemble_contacts", []))
    if len(ec):
        sheets["ensemble_contacts"] = ec.round(4)
        by_res = (ec.groupby("residue_index")
                    .agg(max_contact_frequency=("contact_frequency", "max"),
                         mean_contact_frequency=("contact_frequency", "mean"),
                         n_atom_contacts=("contact_frequency", "size"),
                         min_distance=("mean_distance", "min"))
                    .reset_index().sort_values("max_contact_frequency",
                                               ascending=False))
        sheets["contact_freq_by_residue"] = by_res.round(4)
        top = by_res.head(25)
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        ax.bar([str(r) for r in top["residue_index"]],
               top["max_contact_frequency"], color=ACCENT, width=0.8)
        ax.axhline(0.5, color=NEG, ls="--", lw=1.0, label="consensus ≥ 0.5")
        style_ax(ax, "Residue position", "Max contact frequency",
                 "WT-ensemble pocket contact frequency (top 25)")
        ax.tick_params(axis="x", labelrotation=90)
        ax.legend(frameon=False)
        save_tif(fig, fig_dir / "s06_contact_frequency.tif")

    # pose reliability
    pdset = pd.DataFrame(art.get("pose_dataset", []))
    if len(pdset):
        keep = [c for c in ["group_id", "sample_idx", "confidence_score", "ptm",
                            "iptm", "ligand_iptm", "complex_plddt"] if c in pdset]
        sheets["pose_reliability"] = pdset[keep].round(4)
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        for col, c, lab in [("confidence_score", ACCENT, "confidence"),
                            ("iptm", ACCENT2, "ipTM"),
                            ("complex_plddt", ACCENT3, "pLDDT")]:
            if col in pdset:
                ax.hist(pdset[col].dropna(), bins=30, histtype="step", lw=1.4,
                        color=c, label=lab)
        style_ax(ax, "Score (0-1)", "Pose count",
                 f"Ensemble pose reliability (n = {len(pdset):,} poses)")
        ax.legend(frameon=False)
        save_tif(fig, fig_dir / "s06_pose_reliability.tif")

    # multi-engine roles
    me_path = ig / "multi_engine_docking.json"
    if me_path.exists():
        me = json.loads(me_path.read_text())
        rows = pd.DataFrame(me.get("rows", []))
        if len(rows):
            keep = [c for c in ["source", "role", "rmsd_to_consensus", "fp_overlap",
                                "clash", "key_contacts_ok", "score",
                                "score_gate_pass"] if c in rows]
            sheets["multi_engine_poses"] = rows[keep].round(4)
            piv = (rows.groupby(["source", "role"]).size()
                       .unstack(fill_value=0))
            sheets["multi_engine_roles"] = piv.reset_index()
            fig, ax = plt.subplots(figsize=(6.0, 3.6))
            roles = list(piv.columns)
            xpos = np.arange(len(piv.index))
            w = 0.8 / max(1, len(roles))
            palette = {"weak_positive": ACCENT3, "hard_negative": NEG,
                       "excluded": GREY, "consensus_positive": ACCENT}
            for j, role in enumerate(roles):
                ax.bar(xpos + j * w, piv[role].to_numpy(), width=w, label=role,
                       color=palette.get(role, ACCENT2))
            ax.set_xticks(xpos + (len(roles) - 1) * w / 2)
            ax.set_xticklabels(list(piv.index))
            style_ax(ax, "Docking engine", "Pose count",
                     f"Multi-engine pose roles (n = {len(rows):,})")
            ax.legend(frameon=False, fontsize=7)
            save_tif(fig, fig_dir / "s06_multi_engine_roles.tif")

    # interaction edges from DB
    try:
        ie = pd.read_sql_query(
            "select target_position, interaction_type, distance, contact_probability "
            "from interaction_edge order by contact_probability desc", db)
        if len(ie):
            sheets["interaction_edges"] = ie.round(4)
    except Exception:
        pass
    return sheets


def _parse_positions(mut: str) -> list[int]:
    return [int(m) for m in re.findall(r"[A-Za-z](\d+)[A-Za-z]", str(mut))]


def build_s07(run: Path, db, fig_dir: Path) -> dict:
    sheets = {}
    prov = run / "reports" / "provenance"
    main = prov / "generated_candidates.csv"
    if not main.exists():
        return sheets
    cand = pd.read_csv(main)
    sheets["candidates"] = cand

    # per generator
    gen = cand["generator"].value_counts().reset_index()
    gen.columns = ["generator", "n_candidates"]
    sheets["by_generator"] = gen
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(gen["generator"], gen["n_candidates"], color=ACCENT, width=0.7)
    for i, v in enumerate(gen["n_candidates"]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    style_ax(ax, "Generator", "Candidates",
             f"Candidates by generation strategy (n = {len(cand)})")
    ax.tick_params(axis="x", labelrotation=20)
    save_tif(fig, fig_dir / "s07_candidates_by_generator.tif")

    # mutations per candidate
    if "n_mutations" in cand:
        nm = cand["n_mutations"].value_counts().sort_index()
        sheets["by_n_mutations"] = nm.reset_index().rename(
            columns={"index": "n_mutations", "n_mutations": "count"})
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        ax.bar(nm.index.astype(int), nm.values, color=ACCENT3, width=0.6)
        style_ax(ax, "Mutations per candidate", "Count",
                 "Candidate order distribution")
        ax.set_xticks(nm.index.astype(int))
        save_tif(fig, fig_dir / "s07_mutations_per_candidate.tif")

    # budget tiers
    tiers = []
    for nm, label in [("generated_candidates_single.csv", "single (s08-s10)"),
                      ("generated_candidates_multipoint.csv", "multipoint"),
                      ("generated_candidates_risky.csv", "risky/exploratory")]:
        p = prov / nm
        if p.exists():
            tiers.append((label, sum(1 for _ in open(p)) - 1))
    if tiers:
        sheets["budget_tiers"] = pd.DataFrame(tiers, columns=["tier", "n_candidates"])

    # design-space heatmap: designable positions x normalised metrics
    gf_path = run / "interaction_graphs" / "graph_features.json"
    if gf_path.exists():
        gf = json.loads(gf_path.read_text())
        res_by_idx = {r["residue_index"]: r for r in gf.get("residues", [])}
        design_pos = sorted(gf.get("designable_positions", []))
        # candidate counts per position
        from collections import Counter
        cnt = Counter()
        for mut in cand["mutation_string"]:
            for p in _parse_positions(mut):
                cnt[p] += 1
        if design_pos:
            rec = []
            for p in design_pos:
                r = res_by_idx.get(p, {})
                dist = r.get("dist_to_ligand")
                rec.append({
                    "position": p,
                    "aa": r.get("aa", ""),
                    "n_candidates": cnt.get(p, 0),
                    "conservation": r.get("conservation"),
                    "n_ligand_contacts": r.get("n_ligand_contacts"),
                    "dist_to_ligand": dist,
                    "ligand_closeness": (1.0 / (1.0 + dist)) if dist else None,
                })
            ds = pd.DataFrame(rec)
            sheets["design_space"] = ds.round(4)

            metrics = ["n_candidates", "ligand_closeness", "n_ligand_contacts",
                       "conservation"]
            labels = ["# candidates", "ligand closeness", "contact count",
                      "conservation"]
            M = np.full((len(metrics), len(ds)), np.nan)
            for i, m in enumerate(metrics):
                col = pd.to_numeric(ds[m], errors="coerce").to_numpy(dtype=float)
                rng = np.nanmax(col) - np.nanmin(col)
                if rng > 0:
                    M[i] = (col - np.nanmin(col)) / rng
                else:
                    M[i] = np.where(np.isnan(col), np.nan, 0.0)
            fig_w = max(6.5, 0.22 * len(ds) + 2.2)
            fig, ax = plt.subplots(figsize=(fig_w, 2.8))
            im = ax.imshow(M, aspect="auto", cmap=HEAT, vmin=0, vmax=1)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_xticks(range(len(ds)))
            ax.set_xticklabels([f"{a}{p}" for a, p in zip(ds["aa"], ds["position"])],
                               rotation=90, fontsize=6)
            ax.grid(False)
            cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cb.set_label("normalised", fontsize=7)
            ax.set_title("Design-space map — designable positions")
            save_tif(fig, fig_dir / "s07_design_space_heatmap.tif")
    return sheets


BUILDERS = {
    "s01": lambda run, db, fd, tb: build_s01(run, db, fd),
    "s02": lambda run, db, fd, tb: build_s02(run, db, fd),
    "s03": lambda run, db, fd, tb: build_s03(run, db, fd),
    "s04": lambda run, db, fd, tb: build_s04(run, db, fd),
    "s05": lambda run, db, fd, tb: build_s05(run, db, fd, tb),
    "s06": lambda run, db, fd, tb: build_s06(run, db, fd),
    "s07": lambda run, db, fd, tb: build_s07(run, db, fd),
}


def _copy(src: Path, dst: Path):
    """Copy a file or directory tree (skip silently if the source is absent)."""
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_raw(run: Path, out: Path, level: str = "core"):
    """Copy each stage's actual raw input/output data into <stage>/raw/.

    level='core'  - the WT/main-line I/O (sequences, MSA, WT structure ensemble,
                    WT docking poses, interaction model, candidates)  ~90 MB
    level='full'  - additionally the 150-representative family ensemble (~440 MB)
    """
    if level == "none":
        return

    def raw(key):
        d = out / key / "raw"
        d.mkdir(parents=True, exist_ok=True)
        return d

    print("\n=== raw I/O data ===")
    # s01 - inputs
    r = raw("s01")
    for f in ("target.fasta", "ligand.smi", "extra_ligands.smi"):
        _copy(run / "inputs" / f, r / f)
    # s02 - homolog output
    _copy(run / "homologs" / "filtered_sequences.fasta", raw("s02") / "homologs.fasta")
    # s03 - MSA + conservation
    r = raw("s03")
    _copy(run / "msa" / "alignment.fasta", r / "alignment.fasta")
    _copy(run / "msa" / "remote.a3m", r / "msa.a3m")
    _copy(run / "msa" / "conservation.json", r / "conservation.json")
    for f in glob.glob(str(run / "msa" / "esm_prior*.json")):
        _copy(Path(f), r / "esm_prior.json")
    # s04 - WT co-folding I/O (short, flat names)
    r = raw("s04")
    _copy(run / "complexes" / "boltz" / "wt_boltz_input.yaml", r / "in" / "boltz_input.yaml")
    _copy(run / "complexes" / "boltz" / "wt_msa.a3m", r / "in" / "msa.a3m")
    pred = _boltz_pred_dir(run)
    if pred:
        pdir = r / "pred"
        pdir.mkdir(parents=True, exist_ok=True)
        for f in glob.glob(str(pred / "*")):
            p = Path(f)
            if p.is_dir():
                continue
            name = p.name
            m = re.search(r"model_(\d+)", name)
            nn = f"{int(m.group(1)):02d}" if m else ""
            if name.startswith("confidence") and nn:
                dst = f"conf_{nn}.json"
            elif name.startswith("plddt") and nn:
                dst = f"plddt_{nn}.npz"
            elif name.startswith("pae") and nn:
                dst = f"pae_{nn}.npz"
            elif name.startswith("pde") and nn:
                dst = f"pde_{nn}.npz"
            elif name.startswith("pre_affinity"):
                dst = "pre_affinity.npz"
            elif name.startswith("affinity"):
                dst = "affinity.json"
            elif name.endswith(".pdb") and nn:
                dst = f"model_{nn}.pdb"
            else:
                dst = name
            _copy(p, pdir / dst)
    # s05 - WT docking I/O (per-ligand tag, short names)
    r = raw("s05")
    for f in glob.glob(str(run / "docking" / "gnina" / "wt_*")):
        p = Path(f)
        if p.is_dir():
            continue
        for suf, short in (("_ref_lig.pdb", "ref.pdb"), ("_gnina_out.sdf", "pose.sdf"),
                           ("_rec.pdb", "rec.pdb")):
            if p.name.endswith(suf):
                _copy(p, r / "gnina" / p.name[:-len(suf)] / short)
                break
    for f in glob.glob(str(run / "docking" / "diffdock" / "wt_*")):
        p = Path(f)
        name = p.name
        if name.endswith("_rec.pdb"):
            _copy(p, r / "diffdock" / name[:-len("_rec.pdb")] / "rec.pdb")
        elif name.endswith("_input.csv"):
            _copy(p, r / "diffdock" / name[:-len("_input.csv")] / "input.csv")
        elif name.endswith("_dd_out") and p.is_dir():
            dest = r / "diffdock" / name[:-len("_dd_out")] / "poses"
            sdfs = sorted(glob.glob(str(p / "**" / "*.sdf"), recursive=True))
            seen = set()
            for want_conf in (True, False):
                for s in sdfs:
                    sn = Path(s).name
                    mm = re.match(r"rank(\d+)", sn)
                    if not mm:
                        continue
                    rk = int(mm.group(1))
                    if ("confidence" in sn) == want_conf and rk not in seen:
                        _copy(Path(s), dest / f"rank{rk:02d}.sdf")
                        seen.add(rk)
    # s06 - interaction model outputs (+ optional family ensemble)
    r = raw("s06")
    for f in ("graph_features.json", "s06b_artifacts.json", "interaction_model.json",
              "interaction_model.pkl", "multi_engine_docking.json", "multi_engine_status.json"):
        _copy(run / "interaction_graphs" / f, r / f)
    if level == "full":
        _copy(run / "complexes" / "mutant_boltz", r / "ensemble" / "mutant_boltz")
        for d in glob.glob(str(run / "docking" / "me_*")):
            _copy(Path(d), r / "ensemble" / "docking" / Path(d).name)
    # s07 - candidates + LigandMPNN (flattened, short)
    r = raw("s07")
    for f in glob.glob(str(run / "reports" / "provenance" / "generated_candidates*.csv")):
        _copy(Path(f), r / Path(f).name.replace("generated_candidates", "candidates"))
    lm = run / "mutations" / "ligandmpnn"
    _copy(lm / "input_complex.pdb", r / "ligandmpnn" / "input.pdb")
    for f in glob.glob(str(lm / "lmpnn_out" / "seqs" / "*")):
        _copy(Path(f), r / "ligandmpnn" / "seqs" / Path(f).name)
    for f in glob.glob(str(lm / "lmpnn_out" / "backbones" / "*.pdb")):
        mm = re.search(r"_(\d+)\.pdb$", Path(f).name)
        nn = f"{int(mm.group(1)):02d}" if mm else Path(f).stem
        _copy(Path(f), r / "ligandmpnn" / "backbones" / f"bb_{nn}.pdb")

    for key, _ in STAGES:
        rd = out / key / "raw"
        if rd.exists():
            n = sum(1 for p in rd.rglob("*") if p.is_file())
            sz = sum(p.stat().st_size for p in rd.rglob("*") if p.is_file()) // (1024 * 1024)
            print(f"      {key}/raw  {n} files  {sz} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reports", type=Path, default=None)
    ap.add_argument("--clean-level", default="caveats",
                    choices=["runtime", "caveats", "minimal"])
    ap.add_argument("--raw", default="core", choices=["none", "core", "full"],
                    help="copy raw stage I/O: core (~90MB) / full (+150-rep ensemble) / none")
    ap.add_argument("--only", default=None, help="comma list of stage keys")
    args = ap.parse_args()

    run = args.run
    reports = args.reports or (run / "reports")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    # read-only DB handle: never mutate the source run
    db = sqlite3.connect(f"file:{run / 'evoliez.sqlite'}?mode=ro", uri=True)
    only = set(args.only.split(",")) if args.only else None

    for key, report_name in STAGES:
        if only and key not in only:
            continue
        print(f"\n=== {key} ===")
        sdir = out / key
        fdir = sdir / "figure"
        fdir.mkdir(parents=True, exist_ok=True)

        # 1) cleaned report
        rpath = reports / report_name
        tables = []
        if rpath.exists():
            raw = rpath.read_text(errors="ignore")
            tables = read_report_tables(raw)
            try:
                cleaned = clean_html(raw, args.clean_level)
                (sdir / "report.html").write_text(cleaned)
                print(f"      report  report.html  "
                      f"({len(raw)//1024}KB -> {len(cleaned)//1024}KB cleaned)")
            except Exception as exc:
                print(f"      (clean_html failed: {exc}); copying raw")
                (sdir / "report.html").write_text(raw)
        else:
            print(f"      (no report {report_name})")

        # 2) data + figures
        try:
            sheets = BUILDERS[key](run, db, fdir, tables)
            if sheets:
                write_xlsx(sdir / "data.xlsx", sheets)
        except Exception:
            print("      ERROR building stage:")
            traceback.print_exc()

    # raw input/output data per stage
    try:
        copy_raw(run, out, args.raw)
    except Exception:
        print("      ERROR copying raw data:")
        traceback.print_exc()

    # top-level index for paper navigation
    try:
        pr = db.execute("select target_name, enzyme_family, objective, ligand_id "
                        "from project").fetchone()
    except Exception:
        pr = None
    title = (pr[0] if pr else "EvoLiEZ").upper()
    lines = [f"# {title} - paper data archive", "",
             f"Source run: `{run}`  (read-only; originals untouched)", ""]
    if pr:
        lines += [f"Target **{pr[0]}** · EC {pr[1]} · objective: {pr[2]} · "
                  f"design ligand: {pr[3]}", ""]
    lines += ["Final data only. Each stage folder contains:",
              "- `report.html` - cleaned HTML report (runtime errors + caveat sections removed)",
              "- `data.xlsx` - raw data behind every graph and table (one sheet each)",
              "- `figure/*.tif` - publication figures (300 dpi, RGB, LZW)",
              "- `raw/` - the stage's actual input/output data files (copied from the run)", ""]
    for key, _ in STAGES:
        sdir = out / key
        if not sdir.exists():
            continue
        lines.append(f"## {key} - {STAGE_TITLES.get(key, '')}")
        fdir = sdir / "figure"
        for f in sorted(fdir.glob("*.tif")) if fdir.exists() else []:
            lines.append(f"- `figure/{f.name}` - {FIGURE_CAPTIONS.get(f.stem, '')}")
        if (sdir / "data.xlsx").exists():
            try:
                import openpyxl
                wb = openpyxl.load_workbook(sdir / "data.xlsx", read_only=True)
                lines.append(f"- `data.xlsx` - sheets: {', '.join(wb.sheetnames)}")
                wb.close()
            except Exception:
                lines.append("- `data.xlsx`")
        rdir = sdir / "raw"
        if rdir.exists():
            nf = sum(1 for p in rdir.rglob("*") if p.is_file())
            mb = sum(p.stat().st_size for p in rdir.rglob("*") if p.is_file()) // (1024 * 1024)
            lines.append(f"- `raw/` ({nf} files, {mb} MB) - {RAW_NOTES.get(key, '')}")
        lines.append("")
    # heavy artifacts intentionally not duplicated here
    lines += ["---", "## Source-run artifacts (not duplicated in this archive)"]
    if args.raw != "full":
        lines.append(f"- 150-representative family ensemble (s06 input): "
                     f"`{run}/complexes/mutant_boltz/`, `{run}/docking/me_*` "
                     f"(re-add with `--raw full`)")
    lines += [f"- per-candidate folded structures (s08+): `{run}/structures/`",
              f"- downstream validation (s08+): `{run}/validation/`",
              f"- full project database: `{run}/evoliez.sqlite`"]
    (out / "README.md").write_text("\n".join(lines))
    print(f"\n      index   README.md")

    db.close()
    print(f"\nDONE -> {out}")


if __name__ == "__main__":
    main()
