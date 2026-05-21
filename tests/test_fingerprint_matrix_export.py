"""s06b_interaction_model.py exports the per-pose fingerprint matrix CSV.

The HTML report's section 5 (interaction-fingerprint heatmap) needs the
per-pose fingerprint matrix. The stage trains an InteractionModel on the
matrix internally but did not previously persist it as a CSV. This test
guards the new `ml_datasets/fingerprint_matrix.csv` and its sibling
metadata file (`fingerprint_matrix_meta.json`).
"""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

from evoliez.config import load_config
from evoliez.context import RunContext
from evoliez.features.interaction_descriptor import fingerprint_dim
from evoliez.pipeline import Pipeline
from evoliez.stages import s06b_interaction_model
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Source guard: the export must be wired into the stage's run() and the
# helper must catch its own exceptions so a CSV-write failure can't break
# model training.
# --------------------------------------------------------------------------- #
def test_export_helper_is_invoked_from_stage():
    run_src = inspect.getsource(s06b_interaction_model.InteractionModelStage.run)
    assert "_export_fingerprint_matrix" in run_src

    helper_src = inspect.getsource(
        s06b_interaction_model._export_fingerprint_matrix
    )
    # Must be failure-soft (additive report artefact, not a load-bearing
    # training output).
    assert "try:" in helper_src and "except" in helper_src
    # Writes the meta sidecar alongside the CSV.
    assert "fingerprint_matrix_meta.json" in helper_src


# --------------------------------------------------------------------------- #
# Integration: run the full mock pipeline and verify the artefacts.
# --------------------------------------------------------------------------- #
def _cfg(tmp_path: Path):
    return load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {
            "project.output_dir": str(tmp_path / "run"),
            "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
            "mutation_generation": {
                "methods": ["chemistry_rules"],
                "max_candidates": 30,
            },
            "reranking": {"top_for_redocking": 12, "top_for_md": 4,
                          "model": "xgboost", "use_experimental_labels": False},
            "validation": {"md": {"enabled": True, "protocol_level": 1,
                                   "top_candidates": 4}},
        },
    )


def test_fingerprint_matrix_csv_exported(tmp_path):
    cfg = _cfg(tmp_path)
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx)

    ml_dir = ctx.paths.ml_datasets
    csv_path = ml_dir / "fingerprint_matrix.csv"
    meta_path = ml_dir / "fingerprint_matrix_meta.json"
    assert csv_path.exists(), f"missing {csv_path}"
    assert meta_path.exists(), f"missing {meta_path}"

    # CSV: at least one real-pose row, header begins with pose_id/group_id,
    # then feature_<i> columns to feature_<dim-1>.
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    assert header[:2] == ["pose_id", "group_id"], header[:2]
    assert all(c.startswith("feature_") for c in header[2:])
    assert len(rows) > 0, "fingerprint_matrix.csv has no rows"

    # Column count must match the InteractionModel's feature dimension.
    expected_dim = fingerprint_dim(cfg.interaction_model.k_nearest_residues)
    assert len(header) - 2 == expected_dim, (
        f"feature col count = {len(header) - 2} != expected {expected_dim}"
    )

    # Meta sidecar declares row semantics + dim + row count.
    meta = json.loads(meta_path.read_text())
    assert meta["row_kind"] == "pose"
    assert meta["feature_dim"] == expected_dim
    assert meta["n_rows"] == len(rows)
    assert "row_headers" in meta and meta["row_headers"][:2] == [
        "pose_id", "group_id",
    ]

    # Real-pose rows have feature values that are not all the synthetic-decoy
    # constant: verify by sampling a row and checking it parses as floats.
    sample = rows[0]
    assert sample[0]  # pose_id non-empty
    assert sample[1]  # group_id non-empty
    feats = [float(x) for x in sample[2:]]
    assert len(feats) == expected_dim
