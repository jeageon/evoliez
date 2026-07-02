"""Enzyme-agnostic assay label schema with provenance (ML strategy §C / §4).

The v4 ``experimental/assay_schema.py`` is a target-specific instance (its columns are one
enzyme's readouts). This module is the mechanism-agnostic generalization the strategy asks
for: a label carries WHAT was measured (``label_type`` / ``readout_name``), on WHICH
target/mechanism, in WHICH direction it is good, and WHERE it came from (``source`` +
``confidence``). The central rule enforced in code: a **computed weak label never becomes a
supervised training target** — only ``wetlab`` / ``literature`` sources may. This keeps a
predictor/simulation-derived number from silently training an "activity predictor" (the
ClaimGuard principle).

No enzyme-specific readout names appear here; a mechanism template supplies those.
Pure-stdlib; runs in the light env.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# --- label provenance sources --------------------------------------------------------
WETLAB = "wetlab"
LITERATURE = "literature"
COMPUTED_WEAK = "computed_weak"
#: sources that MAY be used as a supervised training target
SUPERVISED_SOURCES = {WETLAB, LITERATURE}

# --- optimization direction ----------------------------------------------------------
MAXIMIZE = "maximize"
MINIMIZE = "minimize"
TARGET_RANGE = "target_range"
_DIRECTIONS = {MAXIMIZE, MINIMIZE, TARGET_RANGE}

# --- label kinds (enzyme-agnostic; the strategy's assay_type set) ---------------------
LABEL_TYPES = {
    "expression", "solubility", "stability", "substrate_conversion",
    "product_formation", "specificity", "kinetics", "inhibition", "growth_fitness",
    # generic activity bucket (a mechanism decides what "activity" means via readout_name)
    "activity",
}

# label_type values that are ACTIVITY/KINETIC objectives (an activity-predictor head over
# these needs wet-lab data; computed weak labels of these types can never be supervised).
ACTIVITY_LABEL_TYPES = {"activity", "kinetics", "substrate_conversion",
                        "product_formation", "growth_fitness"}


class LabelProvenanceError(RuntimeError):
    pass


@dataclass
class AssayLabel:
    """One measured (or computed-weak) value for one variant, with full provenance."""

    mutation: str
    label_type: str
    value: float
    direction: str = MAXIMIZE
    readout_name: str = ""
    unit: Optional[str] = None
    target_id: str = ""
    enzyme_family: str = ""
    mechanism_class: str = ""
    substrate: Optional[str] = None
    cofactor: Optional[str] = None
    product: Optional[str] = None
    replicate_n: int = 1
    batch_id: Optional[str] = None
    source: str = WETLAB
    confidence: str = "medium"

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {self.direction!r}")
        if self.source not in (SUPERVISED_SOURCES | {COMPUTED_WEAK}):
            raise ValueError(
                f"source must be one of {sorted(SUPERVISED_SOURCES | {COMPUTED_WEAK})}, "
                f"got {self.source!r}")
        if self.label_type not in LABEL_TYPES:
            raise ValueError(f"label_type must be one of {sorted(LABEL_TYPES)}, got {self.label_type!r}")

    @property
    def is_supervised_source(self) -> bool:
        return self.source in SUPERVISED_SOURCES

    @property
    def is_activity(self) -> bool:
        return self.label_type in ACTIVITY_LABEL_TYPES


def assert_supervised_allowed(labels: Iterable[AssayLabel]) -> None:
    """Raise if any label offered as a supervised target is a computed weak label.

    An activity/kinetics supervised head requires an experimental source — a computed
    weak label of an activity type can never train it (the ClaimGuard rule)."""
    for lab in labels:
        if not lab.is_supervised_source:
            raise LabelProvenanceError(
                f"{lab.mutation}:{lab.label_type} has source={lab.source!r}; a "
                f"computed_weak label must not be a supervised target. Use it as a "
                f"feature / weak label / prior instead.")


def supervised_targets(
    labels: Iterable[AssayLabel],
    *,
    label_type: Optional[str] = None,
    readout_name: Optional[str] = None,
) -> Dict[str, float]:
    """Extract ``{mutation: value}`` for supervised training, filtered by kind and source.

    Only ``wetlab``/``literature`` sources are kept. Optionally restrict to one
    ``label_type`` and/or ``readout_name`` so different assays are never mixed into one
    target (the strategy's "do not mix different assays as one activity label")."""
    out: Dict[str, float] = {}
    for lab in labels:
        if not lab.is_supervised_source:
            continue
        if label_type is not None and lab.label_type != label_type:
            continue
        if readout_name is not None and lab.readout_name != readout_name:
            continue
        out[lab.mutation] = lab.value
    return out


_NUMERIC = {"value", "replicate_n"}


def load_assay_labels(path: Path) -> List[AssayLabel]:
    """Load a long-format assay CSV (one row per measurement). Required columns:
    ``mutation, label_type, value``. All other AssayLabel fields are optional columns."""
    rows: List[AssayLabel] = []
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        missing = {"mutation", "label_type", "value"} - cols
        if missing:
            raise ValueError(f"assay CSV missing required columns: {sorted(missing)}")
        for row in reader:
            kw: Dict[str, object] = {}
            for k, v in row.items():
                if k is None or v in (None, ""):
                    continue
                if k == "value":
                    kw[k] = float(v)
                elif k == "replicate_n":
                    kw[k] = int(float(v))
                elif k in AssayLabel.__dataclass_fields__:
                    kw[k] = v
            rows.append(AssayLabel(**kw))  # type: ignore[arg-type]
    return rows
