"""Tests for the section-5 (interaction fingerprint) wave.

Covers:

* :mod:`evoliez.figures.plots.fingerprint_heatmap` - renders the
  per-pose fingerprint CSV produced by stage s06b into a PNG, returns
  ``None`` gracefully when the CSV is missing.
* :mod:`evoliez.figures.plots.top_contacts_table` - renders a top-K
  contact summary table (matplotlib ``Table`` -> PNG) from
  ``edge_level.csv``.
* :mod:`evoliez.figures.three_d.contact_lines` - builds a 3Dmol
  viewer context whose ``custom_script`` field draws dashed
  cylinders from each ligand atom to its top contacting residue.
* :func:`evoliez.figures.discovery.discover` - exposes the new
  ``fingerprint_matrix_csv`` field.
* :func:`evoliez.figures.html.builder._plot_dispatch` - dispatches
  both new 2D figures under the ``05_fingerprint`` section.

The matplotlib-dependent tests use ``pytest.importorskip("matplotlib")``
so the suite passes on the barebones laptop env (``.venv-light``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("evoliez.figures.style")

from evoliez.figures.types import FigureSpec, ReportArtifacts  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _empty_artifacts(run_dir: Path) -> ReportArtifacts:
    run_dir.mkdir(parents=True, exist_ok=True)
    return ReportArtifacts(run_dir=run_dir)


def _write_fingerprint_matrix(
    path: Path, n_rows: int = 6, n_features: int = 12
) -> Path:
    """Write a synthetic ``fingerprint_matrix.csv`` matching stage s06b."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["pose_id", "group_id"] + [f"feature_{i}" for i in range(n_features)]
    lines = [",".join(header)]
    for i in range(n_rows):
        feats = [f"{((i * 7 + j) % 11) / 10.0:.4f}" for j in range(n_features)]
        lines.append(",".join([f"grpA__p{i:04d}", "grpA", *feats]))
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_edge_level(path: Path, n_rows: int = 20) -> Path:
    """Write a synthetic ``edge_level.csv`` matching stage s11 schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "residue_index",
        "ligand_atom_id",
        "mean_distance",
        "contact_frequency",
        "confidence_weighted_score",
        "weak_contact",
    ]
    lines = [",".join(header)]
    # Carefully choose frequencies so top-K ordering is deterministic.
    for i in range(n_rows):
        residue = 10 + i
        atom = f"C{i % 5}"
        freq = round(0.95 - 0.03 * i, 4)
        dist = round(3.0 + 0.1 * i, 3)
        lines.append(
            ",".join(
                [str(residue), atom, f"{dist}", f"{freq}", f"{freq * 0.9:.4f}", "1"]
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_wt_pdb(path: Path) -> Path:
    """Minimal PDB: a handful of CAs + one ligand atom matching edge CSV.

    Coordinates are arbitrary; the ``contact_lines`` renderer only
    needs to be able to look up CA positions and ligand atom positions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # PDB columns are fixed-width; format strings keep them spec-compliant.
    atom_lines = []
    # ATOM CAs for residues 10..29 (matches edge_level fixture above).
    serial = 1
    for resnum in range(10, 30):
        atom_lines.append(
            "ATOM  {:>5d}  CA  ALA A{:>4d}    "
            "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C".format(
                serial, resnum, float(resnum), 0.0, 0.0
            )
        )
        serial += 1
    # HETATM ligand atoms C0..C4 (matches edge_level fixture above).
    for i in range(5):
        atom_lines.append(
            "HETATM{:>5d}  C{}  LIG L   1    "
            "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C".format(
                serial, i, float(i), 5.0, 5.0
            )
        )
        serial += 1
    atom_lines.append("END")
    path.write_text("\n".join(atom_lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# fingerprint_heatmap
# ---------------------------------------------------------------------------


def test_fingerprint_heatmap_renders(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    from evoliez.figures.plots import fingerprint_heatmap

    csv_path = _write_fingerprint_matrix(
        tmp_path / "run" / "ml_datasets" / "fingerprint_matrix.csv"
    )
    art = _empty_artifacts(tmp_path / "run")
    art.fingerprint_matrix_csv = csv_path

    out = tmp_path / "heatmap.png"
    spec = fingerprint_heatmap.render(art, out)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "05_fingerprint_heatmap"
    assert spec.section == "fingerprint"
    assert out.exists() and out.stat().st_size > 0
    assert spec.params["n_features"] == 12
    assert spec.params["n_rows"] == 6


def test_fingerprint_heatmap_returns_none_without_csv(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import fingerprint_heatmap

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "heatmap.png"
    assert fingerprint_heatmap.render(art, out) is None
    assert not out.exists()


def test_fingerprint_heatmap_samples_when_huge(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    from evoliez.figures.plots import fingerprint_heatmap

    csv_path = _write_fingerprint_matrix(
        tmp_path / "fp.csv", n_rows=120, n_features=8
    )
    art = _empty_artifacts(tmp_path / "run")
    art.fingerprint_matrix_csv = csv_path
    out = tmp_path / "heatmap.png"
    spec = fingerprint_heatmap.render(art, out)

    assert spec is not None
    assert spec.params["sampled"] is True
    assert spec.params["n_rows"] == 50
    assert spec.params["n_rows_total"] == 120


# ---------------------------------------------------------------------------
# top_contacts_table
# ---------------------------------------------------------------------------


def test_top_contacts_table_renders(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import top_contacts_table

    csv_path = _write_edge_level(
        tmp_path / "run" / "ml_datasets" / "edge_level.csv", n_rows=20
    )
    art = _empty_artifacts(tmp_path / "run")
    art.ml_datasets = {"edge_level": csv_path}

    out = tmp_path / "top.png"
    spec = top_contacts_table.render(art, out, top_k=10)

    assert isinstance(spec, FigureSpec)
    assert spec.figure_id == "05_top_contacts"
    assert spec.section == "fingerprint"
    assert out.exists() and out.stat().st_size > 0
    assert spec.params["n_rows"] == 10
    assert spec.params["top_k"] == 10
    assert spec.params["n_rows_total"] == 20


def test_top_contacts_table_returns_none_when_missing(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from evoliez.figures.plots import top_contacts_table

    art = _empty_artifacts(tmp_path / "run")
    out = tmp_path / "top.png"
    assert top_contacts_table.render(art, out) is None
    assert not out.exists()


# ---------------------------------------------------------------------------
# contact_lines (3Dmol viewer context)
# ---------------------------------------------------------------------------


def test_contact_lines_builds_custom_script(tmp_path: Path) -> None:
    from evoliez.figures.three_d import contact_lines

    pdb_path = _write_wt_pdb(tmp_path / "run" / "complexes" / "wt.pdb")
    csv_path = _write_edge_level(
        tmp_path / "run" / "ml_datasets" / "edge_level.csv", n_rows=10
    )
    art = _empty_artifacts(tmp_path / "run")
    art.wt_complex_pdb = pdb_path
    art.ml_datasets = {"edge_level": csv_path}

    ctx = contact_lines.render(art)

    assert ctx is not None
    assert ctx.get("missing") is False
    assert "custom_script" in ctx
    assert ctx["custom_script"].count("addCylinder") > 0
    assert ctx["n_contacts"] > 0


def test_contact_lines_returns_none_without_pdb(tmp_path: Path) -> None:
    from evoliez.figures.three_d import contact_lines

    csv_path = _write_edge_level(
        tmp_path / "run" / "ml_datasets" / "edge_level.csv", n_rows=10
    )
    art = _empty_artifacts(tmp_path / "run")
    art.ml_datasets = {"edge_level": csv_path}
    assert contact_lines.render(art) is None


def test_contact_lines_returns_none_without_edges(tmp_path: Path) -> None:
    from evoliez.figures.three_d import contact_lines

    pdb_path = _write_wt_pdb(tmp_path / "run" / "complexes" / "wt.pdb")
    art = _empty_artifacts(tmp_path / "run")
    art.wt_complex_pdb = pdb_path
    assert contact_lines.render(art) is None


# ---------------------------------------------------------------------------
# discovery.fingerprint_matrix_csv
# ---------------------------------------------------------------------------


def test_discovery_finds_fingerprint_matrix(tmp_path: Path) -> None:
    from evoliez.figures.discovery import discover

    csv_path = _write_fingerprint_matrix(
        tmp_path / "ml_datasets" / "fingerprint_matrix.csv"
    )
    arts = discover(tmp_path)
    assert arts.fingerprint_matrix_csv == csv_path


def test_discovery_missing_fingerprint_matrix(tmp_path: Path) -> None:
    from evoliez.figures.discovery import discover

    tmp_path.mkdir(exist_ok=True)
    arts = discover(tmp_path)
    assert arts.fingerprint_matrix_csv is None


# ---------------------------------------------------------------------------
# builder dispatch wiring (source-level guard)
# ---------------------------------------------------------------------------


def test_builder_dispatch_includes_section_5_figures() -> None:
    """Both new figure IDs are registered under the 05_fingerprint section."""
    from evoliez.figures.html import builder

    dispatch = builder._plot_dispatch()
    if not dispatch:
        pytest.skip("matplotlib plots unavailable; dispatch empty")
    assert "05_fingerprint_heatmap" in dispatch
    assert "05_top_contacts" in dispatch
    assert dispatch["05_fingerprint_heatmap"][0] == "05_fingerprint"
    assert dispatch["05_top_contacts"][0] == "05_fingerprint"


def test_builder_source_wires_contact_lines() -> None:
    """The builder body references the contact_lines viewer renderer."""
    import inspect

    from evoliez.figures.html import builder

    src = inspect.getsource(builder)
    assert "contact_lines" in src
    assert "render_contact_lines" in src
