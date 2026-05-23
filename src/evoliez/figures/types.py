"""Dataclasses for the figures/HTML-report subsystem.

Two records:

* :class:`FigureSpec` - one entry that lands in ``visual_manifest.json`` and
  pairs a rendered file (PNG / SVG / HTML fragment) with the data files it
  came from, the renderer used, and any runtime parameters.
* :class:`ReportArtifacts` - the catalog of input files
  :func:`evoliez.figures.discovery.discover` finds in a run directory; every
  optional field is ``None`` (or empty) when the corresponding pipeline
  stage hasn't produced output yet.

Python 3.9-compatible typing (``Optional``/``List``/``Dict``) - the wider
project still supports 3.9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FigureSpec:
    """One figure in the report."""

    figure_id: str            # e.g., "02_conservation_heatmap"
    section: str              # e.g., "msa"
    title: str                # human title
    description: str          # one-line description
    path: Path                # PNG / SVG / HTML fragment path (relative to report dir)
    source_files: List[Path]  # data files this was built from
    renderer: str             # "matplotlib", "pymol", "3dmol", "rdkit", "ffmpeg"
    params: Dict[str, Any] = field(default_factory=dict)
    generated_at: Optional[str] = None  # ISO 8601 UTC


@dataclass
class ReportArtifacts:
    """Catalog of available run artifacts, returned by :func:`discover`."""

    run_dir: Path
    target_fasta: Optional[Path] = None
    ligand_smiles: Optional[str] = None
    alignment_fasta: Optional[Path] = None
    conservation_json: Optional[Path] = None
    wt_complex_pdb: Optional[Path] = None
    mutant_complex_pdbs: Dict[str, Path] = field(default_factory=dict)
    homolog_complex_dirs: List[Path] = field(default_factory=list)
    final_candidates_csv: Optional[Path] = None
    focused_library_csv: Optional[Path] = None
    benchmark_json: Optional[Path] = None
    benchmark_csv: Optional[Path] = None
    provenance_json: Optional[Path] = None
    state_json: Optional[Path] = None
    md_dirs: Dict[str, Path] = field(default_factory=dict)
    interaction_model_json: Optional[Path] = None
    graph_features_json: Optional[Path] = None
    homolog_identities: List[float] = field(default_factory=list)
    ml_datasets: Dict[str, Path] = field(default_factory=dict)
    fingerprint_matrix_csv: Optional[Path] = None
