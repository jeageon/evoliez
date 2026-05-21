"""Wave 4-B regression tests: discovery + plot polish.

Pin the new data sources discovery learned about in Wave 4-B:

* ``interaction_graphs/graph_features.json`` is now an artifact field.
* ``homolog_identities`` is populated from
  ``interaction_model.json`` -> ``_state.json`` meta -> ``alignment.fasta``,
  in that priority order.
* ``mutation_map`` renders from ``_state.json`` meta when the graph
  features file is absent (previously it honest-skipped).
* ``identity_distribution`` renders when ``artifacts.homolog_identities``
  is pre-populated, even if no FASTA / interaction model is on disk.

The companion tests in ``test_figure_plots.py`` cover the truly-empty
artifacts skip paths - those are intentionally left alone here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoliez.figures.discovery import discover
from evoliez.figures.types import ReportArtifacts


# ---------------------------------------------------------------------- #
# discovery: graph_features_json + homolog_identities
# ---------------------------------------------------------------------- #


def test_discovery_finds_graph_features_json(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "interaction_graphs").mkdir(parents=True)
    (run / "interaction_graphs" / "graph_features.json").write_text(
        json.dumps(
            {
                "designable_positions": [10, 20, 30],
                "catalytic_positions": [50],
                "binding_site_positions": [40, 60],
            }
        )
    )

    arts = discover(run)
    assert arts.graph_features_json is not None
    assert arts.graph_features_json.name == "graph_features.json"


def test_discovery_homolog_identities_from_interaction_model(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "interaction_graphs").mkdir(parents=True)
    (run / "interaction_graphs" / "interaction_model.json").write_text(
        json.dumps(
            {"homologs": [{"identity": 0.8}, {"identity": 0.7}, {"identity": 0.6}]}
        )
    )

    arts = discover(run)
    assert arts.homolog_identities == [0.8, 0.7, 0.6]


def test_discovery_homolog_identities_from_alignment_fasta(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "msa").mkdir(parents=True)
    (run / "msa" / "alignment.fasta").write_text(
        ">wt\nMAKVLCVL\n"
        ">homolog_1\nMAKVLCVA\n"  # 7/8 = 87.5%
        ">homolog_2\nMAKVLCAA\n"  # 6/8 = 75%
    )

    arts = discover(run)
    assert len(arts.homolog_identities) == 2
    assert abs(arts.homolog_identities[0] - 0.875) < 0.001
    assert abs(arts.homolog_identities[1] - 0.75) < 0.001


def test_discovery_homolog_identities_prefers_interaction_model_over_fasta(
    tmp_path: Path,
) -> None:
    """interaction_model.json wins when both sources exist."""
    run = tmp_path / "run"
    (run / "interaction_graphs").mkdir(parents=True)
    (run / "interaction_graphs" / "interaction_model.json").write_text(
        json.dumps({"homologs": [{"identity": 0.9}]})
    )
    (run / "msa").mkdir(parents=True)
    (run / "msa" / "alignment.fasta").write_text(">wt\nMAKVL\n>h1\nAAAAA\n")

    arts = discover(run)
    assert arts.homolog_identities == [0.9]


def test_discovery_homolog_identities_uses_state_meta_forward_compat(
    tmp_path: Path,
) -> None:
    """If a future pipeline writes meta.homolog_identities it should be honored."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "_state.json").write_text(
        json.dumps({"meta": {"homolog_identities": [0.85, 0.65, 0.45]}})
    )

    arts = discover(run)
    assert arts.homolog_identities == [0.85, 0.65, 0.45]


def test_discovery_no_homologs_anywhere(tmp_path: Path) -> None:
    """All three sources missing -> empty list, no crash."""
    run = tmp_path / "run"
    run.mkdir()
    arts = discover(run)
    assert arts.homolog_identities == []
    assert arts.graph_features_json is None


# ---------------------------------------------------------------------- #
# mutation_map: render from _state.json meta when graph_features absent
# ---------------------------------------------------------------------- #


def test_mutation_map_renders_from_state_meta_only(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import mutation_map

    run = tmp_path / "run"
    run.mkdir()
    fasta = run / "target.fasta"
    fasta.write_text(">wt\n" + ("A" * 50) + "\n")
    # No graph_features.json - only state meta carries the positions.
    state = run / "_state.json"
    state.write_text(
        json.dumps(
            {
                "meta": {
                    "designable_positions": [5, 12, 22],
                    "catalytic_positions": [30],
                }
            }
        )
    )

    arts = ReportArtifacts(
        run_dir=run,
        target_fasta=fasta,
        state_json=state,
    )
    out = tmp_path / "mut.png"
    spec = mutation_map.render(arts, out)

    assert spec is not None
    assert spec.figure_id == "06_mutation_design_space"
    assert out.exists() and out.stat().st_size > 0
    # Both categories from meta should be reflected in the counts.
    counts = spec.params["counts"]
    assert counts.get("designable") == 3
    assert counts.get("catalytic") == 1


def test_mutation_map_prefers_graph_features_over_meta(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import mutation_map

    run = tmp_path / "run"
    (run / "interaction_graphs").mkdir(parents=True)
    fasta = run / "target.fasta"
    fasta.write_text(">wt\n" + ("A" * 80) + "\n")
    gf = run / "interaction_graphs" / "graph_features.json"
    gf.write_text(
        json.dumps(
            {
                "designable_positions": [1, 2, 3, 4, 5],
                "catalytic_positions": [70],
                "binding_site_positions": [60, 65],
            }
        )
    )
    # State meta says something different - graph_features should win.
    state = run / "_state.json"
    state.write_text(
        json.dumps({"meta": {"designable_positions": [99]}})
    )

    arts = ReportArtifacts(
        run_dir=run,
        target_fasta=fasta,
        state_json=state,
        graph_features_json=gf,
    )
    out = tmp_path / "mut.png"
    spec = mutation_map.render(arts, out)

    assert spec is not None
    assert spec.params["counts"]["designable"] == 5  # not 1
    assert spec.params["counts"]["catalytic"] == 1
    assert spec.params["counts"]["binding_site"] == 2


def test_mutation_map_still_skips_without_fasta(tmp_path: Path) -> None:
    """No sequence length means we can't draw anything - existing skip path."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import mutation_map

    arts = ReportArtifacts(run_dir=tmp_path)
    out = tmp_path / "mut.png"
    assert mutation_map.render(arts, out) is None
    assert not out.exists()


# ---------------------------------------------------------------------- #
# identity_distribution: render from artifacts.homolog_identities
# ---------------------------------------------------------------------- #


def test_identity_distribution_renders_from_artifact_list(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import identity_distribution

    arts = ReportArtifacts(
        run_dir=tmp_path,
        homolog_identities=[0.8, 0.75, 0.7, 0.6, 0.5],
    )
    out = tmp_path / "id.png"
    spec = identity_distribution.render(arts, out)

    assert spec is not None
    assert spec.figure_id == "02_identity_distribution"
    assert out.exists() and out.stat().st_size > 0
    assert spec.params["n_homologs"] == 5
    # Fractions got scaled to percentages.
    assert spec.params["max_identity"] <= 100.0
    assert spec.params["min_identity"] >= 0.0


def test_identity_distribution_artifact_list_normalizes_mixed_scales(
    tmp_path: Path,
) -> None:
    """Some sources emit fractions, some percentages - we normalize."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import identity_distribution

    arts = ReportArtifacts(
        run_dir=tmp_path,
        homolog_identities=[0.8, 75.0, 0.5, 60.0],
    )
    out = tmp_path / "id.png"
    spec = identity_distribution.render(arts, out)

    assert spec is not None
    # max should be 80% (0.8 -> 80) or 75% (already %), so <= 100.
    assert spec.params["max_identity"] <= 100.0
    # min should be 50% (0.5 -> 50), not 0.5.
    assert spec.params["min_identity"] >= 1.0


def test_identity_distribution_still_skips_when_empty(tmp_path: Path) -> None:
    """Empty artifacts on every source - the existing skip path must still fire."""
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import identity_distribution

    arts = ReportArtifacts(run_dir=tmp_path)
    out = tmp_path / "id.png"
    assert identity_distribution.render(arts, out) is None
    assert not out.exists()
