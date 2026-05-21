"""Scan a run directory and catalog the artifacts the figures module consumes.

The pipeline emits a deep tree under ``run_dir``; this module locates each
file the report needs and returns a :class:`ReportArtifacts` record.  Every
field is optional - if a stage hasn't run yet (or its outputs were
cleaned), the corresponding entry is ``None``/empty and downstream
renderers skip that figure rather than crash.

Path conventions match :class:`evoliez.io.paths.ProjectPaths` and the
Boltz / MD stage writers; see the spec at the top of each entry below.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from evoliez.figures.types import ReportArtifacts


def discover(run_dir: Path) -> ReportArtifacts:
    """Catalog every artifact under ``run_dir`` the report can use.

    The function is intentionally tolerant: missing directories and missing
    files are reported as ``None`` instead of raising, so a partial run
    (e.g., the user ran ``--to s05_mutations``) still yields a usable
    manifest.
    """
    run_dir = Path(run_dir)

    arts = ReportArtifacts(run_dir=run_dir)
    if not run_dir.exists():
        return arts

    # ---- top-level state / reports --------------------------------------
    arts.state_json = _first_existing(run_dir / "_state.json")
    state_meta = _read_state_meta(arts.state_json)

    reports = run_dir / "reports"
    arts.final_candidates_csv = _first_existing(reports / "final_candidates.csv")
    arts.focused_library_csv = _first_existing(reports / "focused_library.csv")
    arts.benchmark_json = _first_existing(reports / "benchmark.json")
    arts.benchmark_csv = _first_existing(reports / "benchmark.csv")
    arts.provenance_json = _first_existing(reports / "provenance.json")

    # ---- inputs (target FASTA, ligand SMILES) ---------------------------
    # Several discovery sources, ordered by reliability:
    #   1. inputs/target.fasta or .fa (canonical drop)
    #   2. inputs/*.fasta glob (looser but explicit)
    #   3. run_dir/target.fasta (some setups skip the inputs/ subdir)
    #   4. examples/<project_dir>/target.fasta referenced from provenance
    # Any one match is enough; missing target_fasta just gates the
    # mutation_map and sequence-track figures, the rest still render.
    inputs = run_dir / "inputs"
    fasta_candidates: List[Path] = [
        inputs / "target.fasta",
        inputs / "target.fa",
    ]
    if inputs.exists():
        fasta_candidates.extend(sorted(inputs.glob("*.fasta")))
    fasta_candidates.append(run_dir / "target.fasta")
    fasta_candidates.append(run_dir / "target.fa")
    arts.target_fasta = _first_existing(*fasta_candidates)

    # Ligand SMILES: file in inputs/ OR _state.json meta. The pipeline
    # writes meta.ligand_smiles in s01_input, so this is a reliable
    # late-stage fallback even when no inputs/ligand.smi was provided.
    arts.ligand_smiles = (
        _read_ligand_smiles(inputs)
        or _smiles_from_meta(state_meta)
        or _smiles_from_provenance(arts.provenance_json)
    )

    # ---- MSA / conservation ---------------------------------------------
    msa = run_dir / "msa"
    arts.alignment_fasta = _first_existing(
        msa / "alignment.fasta",
        msa / "alignment.a3m",
    )
    arts.conservation_json = _first_existing(msa / "conservation.json")

    # ---- complexes: WT + per-mutant Boltz outputs -----------------------
    complexes = run_dir / "complexes"
    arts.wt_complex_pdb = _find_wt_complex(complexes)
    arts.mutant_complex_pdbs = _find_mutant_complexes(complexes)
    arts.homolog_complex_dirs = _find_homolog_complex_dirs(run_dir)

    # ---- MD per-candidate dirs ------------------------------------------
    arts.md_dirs = _find_md_dirs(run_dir / "md")

    # ---- interaction graphs ---------------------------------------------
    igraphs = run_dir / "interaction_graphs"
    arts.interaction_model_json = _first_existing(igraphs / "interaction_model.json")

    # ---- ML datasets ----------------------------------------------------
    arts.ml_datasets = _find_ml_datasets(run_dir / "ml_datasets")

    return arts


# ---------------------------------------------------------------------------
# helpers - kept private and small so the public API stays a single function
# ---------------------------------------------------------------------------


def _first_existing(*candidates: Path) -> Optional[Path]:
    """Return the first existing path, else ``None``."""
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def _read_ligand_smiles(inputs_dir: Path) -> Optional[str]:
    """Look for an inputs/ligand.smi / ligand.txt with a single SMILES line."""
    if not inputs_dir.exists():
        return None
    for name in ("ligand.smi", "ligand.smiles", "ligand.txt"):
        p = inputs_dir / name
        if p.exists():
            try:
                first = p.read_text().splitlines()
            except OSError:
                continue
            for line in first:
                line = line.strip()
                if line and not line.startswith("#"):
                    # SMILES files sometimes have "SMILES NAME" on a line.
                    return line.split()[0]
    return None


def _read_state_meta(state_path: Optional[Path]) -> Dict[str, Any]:
    """Return ``_state.json`` ``meta`` dict, or ``{}`` if missing/unparseable.

    The pipeline records small JSON-safe facts there (sequence length,
    ligand SMILES, n_homologs, etc.) - they're cheap to read once at
    report-build time and give us a reliable fallback when canonical
    file artifacts (e.g., inputs/ligand.smi) were never written.
    """
    if state_path is None or not state_path.exists():
        return {}
    try:
        doc = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    meta = doc.get("meta")
    return meta if isinstance(meta, dict) else {}


def _smiles_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    """Pull ``meta.ligand_smiles`` (string) if present."""
    s = meta.get("ligand_smiles")
    return s if isinstance(s, str) and s.strip() else None


def _smiles_from_provenance(provenance_path: Optional[Path]) -> Optional[str]:
    """Last-resort fallback: provenance.json may carry the SMILES too."""
    if provenance_path is None or not provenance_path.exists():
        return None
    try:
        doc = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("ligand_smiles", "smiles"):
        s = doc.get(key)
        if isinstance(s, str) and s.strip():
            return s
    return None


def _find_wt_complex(complexes: Path) -> Optional[Path]:
    """Best Boltz model for the WT complex.

    Boltz emits ``complexes/boltz_results_wt_boltz_input/predictions/wt_boltz_input/<...>_model_0.pdb``;
    we pick the lexicographically-first ``*model_0.pdb`` so the same input
    deterministically picks the same file (no mtime fragility).
    """
    if not complexes.exists():
        return None
    pred_root = complexes / "boltz_results_wt_boltz_input" / "predictions"
    if pred_root.exists():
        hits = sorted(pred_root.rglob("*model_0.pdb"))
        if hits:
            return hits[0]
    # Fallback: any top-level WT pdb the pipeline may have promoted.
    for name in ("wt_complex.pdb", "wt.pdb"):
        p = complexes / name
        if p.exists():
            return p
    return None


_MUT_DIR_RE = re.compile(r"boltz_results_mut_(?P<cand>.+?)_boltz_input$")


def _find_mutant_complexes(complexes: Path) -> Dict[str, Path]:
    """Map cand_id -> best Boltz PDB for each mutant.

    Layout: ``complexes/mutant_boltz/boltz_results_mut_<cand>_boltz_input/predictions/.../*model_0.pdb``.
    """
    out: Dict[str, Path] = {}
    if not complexes.exists():
        return out
    mutant_root = complexes / "mutant_boltz"
    if not mutant_root.exists():
        return out
    for sub in sorted(mutant_root.iterdir()):
        if not sub.is_dir():
            continue
        m = _MUT_DIR_RE.match(sub.name)
        if not m:
            continue
        cand = m.group("cand")
        preds = sub / "predictions"
        if not preds.exists():
            continue
        hits = sorted(preds.rglob("*model_0.pdb"))
        if hits:
            out[cand] = hits[0]
    return out


def _find_homolog_complex_dirs(run_dir: Path) -> List[Path]:
    """Per-homolog complex output dirs (stage s06b representatives)."""
    reps = run_dir / "complexes" / "representatives"
    if not reps.exists():
        return []
    return sorted(p for p in reps.iterdir() if p.is_dir())


def _find_md_dirs(md_root: Path) -> Dict[str, Path]:
    """Each child directory of ``md/`` is a candidate id."""
    if not md_root.exists():
        return {}
    out: Dict[str, Path] = {}
    for sub in sorted(md_root.iterdir()):
        if sub.is_dir():
            out[sub.name] = sub
    return out


def _find_ml_datasets(ml_root: Path) -> Dict[str, Path]:
    """Map dataset stem -> csv path under ``ml_datasets/`` (e.g. ``pose_level``)."""
    if not ml_root.exists():
        return {}
    out: Dict[str, Path] = {}
    for csv in sorted(ml_root.glob("*.csv")):
        out[csv.stem] = csv
    return out
