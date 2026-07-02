"""Wet-lab assay table schema for v4 calibration."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


REQUIRED_ASSAY_COLUMNS = (
    "variant_id",
    "mutation",
    "lane",
    "expression",
    "soluble_fraction",
    "NADP_activity",
    "NAD_activity",
    "assay_conditions",
    "replicate_id",
)


class AssayRecord(BaseModel):
    variant_id: str
    mutation: str
    lane: str
    expression: float
    soluble_fraction: float
    NADP_activity: float
    NAD_activity: float
    assay_conditions: str
    replicate_id: str
    kcat: Optional[float] = None
    KM_NADP: Optional[float] = None
    KM_formate: Optional[float] = None
    kcat_KM_NADP: Optional[float] = None
    kcat_KM_NAD: Optional[float] = None
    Tm: Optional[float] = None
    residual_activity: Optional[float] = None
    notes: str = ""

    @model_validator(mode="after")
    def _check_range(self) -> "AssayRecord":
        if self.expression < 0 or self.soluble_fraction < 0:
            raise ValueError("expression and soluble_fraction must be non-negative")
        return self


def validate_assay_columns(columns: List[str]) -> None:
    missing = [c for c in REQUIRED_ASSAY_COLUMNS if c not in columns]
    if missing:
        raise ValueError("assay table missing required columns: " + ", ".join(missing))


def load_assay_csv(path: Path) -> List[AssayRecord]:
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        validate_assay_columns(reader.fieldnames or [])
        return [AssayRecord(**_coerce_row(row)) for row in reader]


def _coerce_row(row: Dict[str, str]) -> Dict[str, object]:
    numeric = {
        "expression",
        "soluble_fraction",
        "NADP_activity",
        "NAD_activity",
        "kcat",
        "KM_NADP",
        "KM_formate",
        "kcat_KM_NADP",
        "kcat_KM_NAD",
        "Tm",
        "residual_activity",
    }
    out: Dict[str, object] = dict(row)
    for key in numeric:
        val = out.get(key)
        if val in (None, ""):
            out[key] = None if key not in REQUIRED_ASSAY_COLUMNS else 0.0
        else:
            out[key] = float(val)  # type: ignore[arg-type]
    return out
