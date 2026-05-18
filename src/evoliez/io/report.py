"""Final report generation (spec sections 16 & 4.3)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence

from evoliez.config import Config
from evoliez.io.paths import ProjectPaths
from evoliez.ranking.score import ScoreBreakdown, rationale
from evoliez.types import Candidate


def write_reports(
    cfg: Config,
    paths: ProjectPaths,
    ranked: Sequence[Candidate],
    breakdowns: dict[str, ScoreBreakdown],
) -> List[Path]:
    written: List[Path] = []
    fmts = set(cfg.output.report_format)

    if "csv" in fmts or True:  # CSV is always emitted (machine-readable record)
        written.append(_write_candidates_csv(paths, ranked))
        written.append(_write_library_csv(cfg, paths, ranked))
    if "markdown" in fmts:
        written.append(_write_markdown(cfg, paths, ranked, breakdowns))
    if "pymol_session" in fmts:
        written.append(_write_pymol(paths, ranked))
    return written


def _write_candidates_csv(paths: ProjectPaths, ranked: Sequence[Candidate]) -> Path:
    p = paths.reports / "final_candidates.csv"
    with p.open("w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(
            ["rank", "candidate_id", "mutations", "generator", "final_score",
             "ml_score", "stability_ddg", "docking_score", "md_lite_score"]
        )
        for i, c in enumerate(ranked, 1):
            wri.writerow(
                [i, c.candidate_id, c.mutation_str, c.generator,
                 c.scores.get("final_score", 0.0), c.scores.get("ml_score", 0.0),
                 c.scores.get("ddg_fold", 0.0), c.scores.get("docking_score", 0.0),
                 c.scores.get("md_lite_score", 0.0)]
            )
    return p


def _write_library_csv(
    cfg: Config, paths: ProjectPaths, ranked: Sequence[Candidate]
) -> Path:
    p = paths.reports / "focused_library.csv"
    n = cfg.output.final_library_size
    with p.open("w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(["well", "candidate_id", "mutations", "final_score"])
        for i, c in enumerate(ranked[:n]):
            well = f"{chr(65 + i // 12)}{i % 12 + 1}"
            wri.writerow([well, c.candidate_id, c.mutation_str,
                          c.scores.get("final_score", 0.0)])
    return p


def _write_markdown(
    cfg: Config,
    paths: ProjectPaths,
    ranked: Sequence[Candidate],
    breakdowns: dict[str, ScoreBreakdown],
) -> Path:
    p = paths.reports / "final_report.md"
    L: List[str] = []
    L.append(f"# EvoLiEZ report — {cfg.project.name}\n")
    L.append(f"- Objective: **{cfg.project.objective}**")
    L.append(f"- Target: `{cfg.input.target_id}`  Ligand: `{cfg.input.ligand.id}`")
    L.append(f"- Backend: `{cfg.backend.value}`  Candidates ranked: {len(ranked)}")
    L.append(f"- Focused library size: {min(cfg.output.final_library_size, len(ranked))}\n")
    L.append("## Top candidates\n")
    for i, c in enumerate(ranked[: cfg.output.top_single_mutants], 1):
        bd = breakdowns.get(c.candidate_id)
        L.append(f"### {i}. {c.mutation_str}  (score {c.scores.get('final_score', 0.0):.3f})")
        L.append(f"- candidate_id: `{c.candidate_id}`  generator: {c.generator}")
        if bd:
            for line in rationale(c, bd):
                L.append(f"  - {line}")
        md_fail = c.details.get("md_failure_reasons")
        if md_fail:
            L.append(f"  - MD note: {md_fail}")
        L.append("")
    L.append("## Method summary\n")
    L.append(
        "Scores combine MSA permissiveness, ligand atom-residue interaction "
        "gain, complex confidence, redocking consistency, contact "
        "preservation, ΔΔG stability and short-MD behaviour (spec §16). "
        "Computational predictions are testable hypotheses, not guarantees of "
        "activity."
    )
    p.write_text("\n".join(L) + "\n")
    return p


def _write_pymol(paths: ProjectPaths, ranked: Sequence[Candidate]) -> Path:
    p = paths.reports / "session.pml"
    lines = ["# PyMOL helper - load top complexes", "bg_color white"]
    for c in ranked[:10]:
        path = c.details.get("complex_path")
        if path:
            lines.append(f"load {path}, {c.candidate_id}")
    p.write_text("\n".join(lines) + "\n")
    return p
