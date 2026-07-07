"""Wet-lab calibration loop for v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from evoliez.mechanism import vocab
from evoliez.ranking.claim_guard import ClaimProvenance

from .assay_schema import AssayRecord, load_assay_csv
from .metrics import hit_rate_by_lane, low_ml_false_negative_rate, replicate_variance


class CalibrationResult(BaseModel):
    claim_level: str = "L0_uncalibrated"
    records: List[AssayRecord] = Field(default_factory=list)
    metrics: Dict[str, object] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


def calibrate_assay_results(path: Optional[Path]) -> CalibrationResult:
    if path is None or not Path(path).exists():
        return CalibrationResult(
            claim_level="L0_uncalibrated",
            metrics={"status": "no_wetlab_file"},
            warnings=["no wet-lab file supplied; claims remain L0"],
        )
    records = load_assay_csv(Path(path))
    replicate_counts: Dict[str, int] = {}
    lanes = {r.lane for r in records}
    for r in records:
        replicate_counts[r.mutation] = replicate_counts.get(r.mutation, 0) + 1
    has_controls = "baseline" in lanes and any("control" in lane for lane in lanes)
    has_replicates = any(n >= 2 for n in replicate_counts.values())
    warnings: List[str] = []
    if not has_controls:
        warnings.append("baseline/control lanes missing; do not upgrade beyond L1")
    if not has_replicates:
        warnings.append("replicates missing; do not treat single endpoint as calibrated truth")
    claim_level = "L2_calibrated" if has_controls and has_replicates else "L1_screened"
    metrics = {
        "hit_rate_by_lane": hit_rate_by_lane(records),
        "low_ml_false_negative_rate": low_ml_false_negative_rate(records),
        "replicate_variance": replicate_variance(records),
        "n_records": len(records),
    }
    return CalibrationResult(
        claim_level=claim_level,
        records=records,
        metrics=metrics,
        warnings=warnings,
    )


def claim_provenance_for_calibration(result: CalibrationResult) -> ClaimProvenance:
    wetlab = result.claim_level in {"L1_screened", "L2_calibrated"}
    controls = result.claim_level == "L2_calibrated"
    return ClaimProvenance(
        reference_claim_strength=vocab.UNCALIBRATED,
        geometry_claim_ceiling=vocab.UNCALIBRATED,
        ensemble_claim_ceiling=vocab.UNCALIBRATED,
        wetlab_replicated=wetlab and controls,
        only_short_md=True,
        known_active_controls=controls,
        known_inactive_controls=controls,
        de_novo_pose_disagreement_high=True,
        wt_reaction_geometry_sparse=True,
        ligand_parameterization_uncertain=True,
    )


def render_calibration_report(result: CalibrationResult) -> str:
    lines = [
        "# V4 wet-lab calibration report",
        "",
        f"Claim level: {result.claim_level}",
        "",
    ]
    if result.claim_level == "L0_uncalibrated":
        lines.append("No assay file was supplied. The workflow remains hypothesis-grade triage.")
    else:
        lines.append("Assay data were ingested for this enzyme-family calibration scope.")
        lines.append("")
        lines.append("Metrics:")
        lines.append("```json")
        lines.append(json.dumps(result.metrics, indent=2, sort_keys=True))
        lines.append("```")
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def write_calibration_report(result: CalibrationResult, markdown_path: Path, json_path: Path) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_calibration_report(result))
    json_path.write_text(result.model_dump_json(indent=2))
