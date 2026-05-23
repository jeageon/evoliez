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
    # P0.5 / P0.6: the report carries the new evidence/provenance columns
    # so a server check (`grep has-col ...`) can confirm they reached
    # disk; without this the data lived only on cand.scores in memory
    # and never appeared in the user-facing CSV.
    cols = [
        "rank", "candidate_id", "mutations", "generator",
        # P0a: gate columns FIRST so the eye lands on them - the user can
        # immediately filter for `is_blocked=0` to get trustworthy top-N.
        "is_blocked", "block_reason",
        "final_score", "ml_score", "stability_ddg", "docking_score",
        "md_lite_score",
        # P0.5 — evidence class + Boltz delta source + pool tag
        "evidence_class", "pool", "boltz_delta_source",
        "uncertainty",
        # P0.1 / P0.3 — pose validity + PLIF recovery + method disagreement
        "pose_validity_status", "plif_recovery", "plif_recovery_min",
        "docking_method_disagreement",
        # P0.6 — MD provenance (FF / HMR / replicas / did-run)
        "md_did_run", "md_passed", "md_status",
        "md_ligand_forcefield", "md_hmr_enabled",
        "md_timestep_fs", "md_replicas_run",
    ]
    with p.open("w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(cols)
        for i, c in enumerate(ranked, 1):
            wri.writerow([
                i, c.candidate_id, c.mutation_str, c.generator,
                # P0a
                int(bool(c.details.get("is_blocked")
                         or c.scores.get("is_blocked"))),
                c.details.get("block_reason")
                  or c.scores.get("block_reason", ""),
                c.scores.get("final_score", 0.0),
                c.scores.get("ml_score", 0.0),
                c.scores.get("ddg_fold", 0.0),
                c.scores.get("docking_score", 0.0),
                c.scores.get("md_lite_score", 0.0),
                # P0.5
                c.scores.get("evidence_class", c.details.get("evidence_class", "")),
                c.scores.get("pool", c.details.get("pool", "")),
                c.scores.get("boltz_delta_source",
                             c.details.get("boltz_delta_source", "")),
                c.scores.get("uncertainty", ""),
                # P0.1 / P0.3
                c.scores.get("pose_validity_status", ""),
                c.scores.get("plif_recovery", ""),
                c.scores.get("plif_recovery_min", ""),
                c.scores.get("docking_method_disagreement", ""),
                # P0.6
                c.scores.get("md_did_run", ""),
                int(bool(c.details.get("md_passed"))) if "md_passed" in c.details else "",
                c.scores.get("md_status", ""),
                c.scores.get("md_ligand_forcefield", ""),
                c.scores.get("md_hmr_enabled", ""),
                c.scores.get("md_timestep_fs", ""),
                c.scores.get("md_replicas_run", ""),
            ])
    return p


def _write_library_csv(
    cfg: Config, paths: ProjectPaths, ranked: Sequence[Candidate]
) -> Path:
    """Top-N focused library CSV.

    P0a: hard-filters out blocked candidates (evidence=Reject, invalid
    pose, failed MD). The wet-lab library should NEVER contain
    candidates we've already flagged as unreliable - even if their
    final_score happens to be high.
    """
    p = paths.reports / "focused_library.csv"
    n = cfg.output.final_library_size
    # Hard-filter blocked candidates so the library is trustworthy.
    accepted = [
        c for c in ranked
        if not (c.details.get("is_blocked")
                or c.scores.get("is_blocked"))
    ]
    with p.open("w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(["well", "candidate_id", "mutations", "final_score",
                      "evidence_class"])
        for i, c in enumerate(accepted[:n]):
            well = f"{chr(65 + i // 12)}{i % 12 + 1}"
            wri.writerow([
                well, c.candidate_id, c.mutation_str,
                c.scores.get("final_score", 0.0),
                c.scores.get("evidence_class",
                             c.details.get("evidence_class", "")),
            ])
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
        rec = c.details.get("recommendation")
        if rec:
            L.append(
                f"  - **{rec}** (uncertainty "
                f"{c.scores.get('uncertainty', 0.0):.2f})"
            )
        if c.scores.get("ts_geometry_score") is not None:
            L.append(
                f"  - catalytic geometry score "
                f"{c.scores.get('ts_geometry_score', 0.0):.2f}"
            )
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
