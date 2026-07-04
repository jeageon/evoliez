"""ROADMAP_V3 ML2 regression: the s08 supervised-label guard now checks label
PROVENANCE, not just the column name. A computed/predicted value in an `activity`
column must never become a supervised training target — only wetlab/literature
sources may. Legacy name-only datasets still load (back-compat).
"""
from __future__ import annotations

import csv
import types

from evoliez.stages.s08_reranker import RerankerStage


def _ctx(path):
    return types.SimpleNamespace(config=types.SimpleNamespace(
        input=types.SimpleNamespace(experimental_dataset=str(path))))


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


def test_long_format_keeps_only_wetlab_source(tmp_path):
    p = _write(tmp_path / "assay.csv",
               ["mutation", "label_type", "value", "source"],
               [["A1G", "activity", "1.5", "wetlab"],
                ["B2C", "activity", "0.3", "computed_weak"],   # must be excluded
                ["C3D", "activity", "2.1", "literature"]])
    labels = RerankerStage()._load_labels(_ctx(p))
    assert labels == {"A1G": 1.5, "C3D": 2.1}
    assert "B2C" not in labels  # computed_weak never a supervised target


def test_long_format_all_computed_yields_no_targets(tmp_path):
    p = _write(tmp_path / "assay.csv",
               ["mutation", "label_type", "value", "source"],
               [["A1G", "activity", "1.5", "computed_weak"],
                ["B2C", "activity", "0.3", "computed_weak"]])
    labels = RerankerStage()._load_labels(_ctx(p))
    assert labels == {}  # nothing supervised -> s08 falls back to heuristic


def test_legacy_wide_with_source_column_honours_provenance(tmp_path):
    p = _write(tmp_path / "legacy.csv",
               ["mutation", "activity", "source"],
               [["A1G", "2.0", "wetlab"],
                ["B2C", "9.9", "computed_weak"]])   # skipped
    labels = RerankerStage()._load_labels(_ctx(p))
    assert labels == {"A1G": 2.0}


def test_legacy_wide_without_source_still_loads(tmp_path):
    # back-compat: a legacy dataset with no provenance column still trains
    p = _write(tmp_path / "legacy2.csv",
               ["mutation", "activity"],
               [["A1G", "2.0"], ["B2C", "1.0"]])
    labels = RerankerStage()._load_labels(_ctx(p))
    assert labels == {"A1G": 2.0, "B2C": 1.0}
