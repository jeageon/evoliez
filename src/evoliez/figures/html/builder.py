"""Central orchestrator for the EvoLiEZ HTML report package.

:func:`build_report` ties together every layer of the figures subsystem:

* Foundation (``discover``, :class:`~evoliez.figures.manifest.ManifestBuilder`,
  matplotlib style).
* 2D plot renderers (:mod:`evoliez.figures.plots`).
* 3D viewer renderers (:mod:`evoliez.figures.three_d`, optional - imported
  defensively so the builder still works if that sibling package isn't
  finished or PyMOL is unavailable).
* Packaging (:mod:`evoliez.figures.packaging`, optional - same defensive
  treatment).

It produces a portable ``<output_dir>/`` containing ``visual_report.html``,
``visual_manifest.json``, and the ``figures/`` ``static/`` ``pdb/`` (and
optionally ``movies/``) subdirectories, then zips it to ``output_zip``.
The function is deliberately tolerant: a missing optional dependency or a
stage that hasn't run yet downgrades to a warning + skipped figure rather
than an exception.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evoliez.figures.discovery import discover
from evoliez.figures.html import STATIC_DIR, TEMPLATES_DIR, VENDOR_DIR
from evoliez.figures.manifest import ManifestBuilder
from evoliez.figures.style import apply_mpl_style
from evoliez.figures.types import FigureSpec, ReportArtifacts
from evoliez.logging_utils import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Defensive imports - sibling agents may not have shipped yet.
# ---------------------------------------------------------------------------

try:
    from evoliez.figures.three_d import (  # type: ignore[attr-defined]
        make_viewer_context,
        pymol_available,
        render_mutation_overlay,
        render_pocket_view,
        render_pose_ensemble,
    )

    _THREED_OK = True
except Exception:  # noqa: BLE001 - any failure in the sibling package
    _THREED_OK = False

    def pymol_available() -> bool:  # type: ignore[no-redef]
        return False

    def make_viewer_context(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[no-redef]
        return {"missing": True, "pdb_text": "", "id": kwargs.get("viewer_id", "viewer")}

    def render_pocket_view(*args: Any, **kwargs: Any) -> Optional[FigureSpec]:  # type: ignore[no-redef]
        return None

    def render_pose_ensemble(*args: Any, **kwargs: Any) -> Optional[FigureSpec]:  # type: ignore[no-redef]
        return None

    def render_mutation_overlay(*args: Any, **kwargs: Any) -> Optional[FigureSpec]:  # type: ignore[no-redef]
        return None


try:
    from evoliez.figures.packaging import (  # type: ignore[attr-defined]
        apply_mock_watermark,
        build_self_contained,
        build_zip,
        stamp_figures_with_mock,
    )

    _PACKAGING_OK = True
except Exception:  # noqa: BLE001
    _PACKAGING_OK = False

    # Try a partial import (bundler is the most-likely-present submodule).
    try:
        from evoliez.figures.packaging.bundler import build_zip  # type: ignore[no-redef]

        _PACKAGING_OK = True
    except Exception:  # noqa: BLE001
        def build_zip(report_dir: Path, output_zip: Path, **kwargs: Any) -> Path:  # type: ignore[no-redef]
            """Minimal fallback ZIP builder when the sibling agent hasn't shipped."""
            import zipfile

            output_zip = Path(output_zip)
            output_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(Path(report_dir).rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(report_dir).as_posix())
            return output_zip

    try:
        from evoliez.figures.packaging.selfcontained import build_self_contained  # type: ignore[no-redef]
    except Exception:  # noqa: BLE001
        def build_self_contained(report_dir: Path, output_html: Path, **kwargs: Any) -> Path:  # type: ignore[no-redef]
            output_html = Path(output_html)
            output_html.parent.mkdir(parents=True, exist_ok=True)
            src = Path(report_dir) / "visual_report.html"
            if src.exists():
                output_html.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            return output_html

    def apply_mock_watermark(html_path: Path, *args: Any, **kwargs: Any) -> Path:  # type: ignore[no-redef]
        """Fallback mock-watermark injector when the sibling agent hasn't shipped."""
        p = Path(html_path)
        if not p.exists():
            return p
        text = p.read_text(encoding="utf-8")
        if "mock-backend-banner" in text or "MOCK DATA" in text.upper():
            return p
        banner = (
            '<div class="mock-backend-banner" role="alert">'
            "&#9888; MOCK DATA - illustrative only, not for publication."
            "</div>\n"
        )
        if "<body>" in text:
            text = text.replace("<body>", "<body>\n" + banner, 1)
        else:
            text = banner + text
        p.write_text(text, encoding="utf-8")
        return p

    def stamp_figures_with_mock(figures_dir: Path, **kwargs: Any) -> List[Path]:  # type: ignore[no-redef]
        return []


# ---------------------------------------------------------------------------
# 2D plot dispatch table.
# ---------------------------------------------------------------------------


def _plot_dispatch() -> Dict[str, Tuple[str, Callable[..., Optional[FigureSpec]]]]:
    """Map figure_id -> (section_key, render_callable).

    Plot modules are imported lazily so a bare-bones environment without
    matplotlib still loads the builder for unit tests.  Each entry's
    callable matches the renderer contract:
    ``render(artifacts, out_path, *, style, **kwargs) -> Optional[FigureSpec]``.
    """
    try:
        from evoliez.figures.plots import (
            benchmark_recovery,
            conservation,
            evidence_distribution,
            fingerprint_heatmap,
            identity_distribution,
            library_diversity,
            ligand_2d,
            md_key_distances,
            md_rmsd,
            mutation_map,
            score_waterfall,
            top_contacts_table,
        )
    except Exception:  # noqa: BLE001
        return {}

    return {
        "00_evidence_distribution": ("00_overview", evidence_distribution.render),
        "00_benchmark_recovery": ("00_overview", benchmark_recovery.render),
        "01_ligand_2d": ("01_input", ligand_2d.render),
        "02_conservation_heatmap": ("02_msa", conservation.render),
        "02_identity_distribution": ("02_msa", identity_distribution.render),
        "05_fingerprint_heatmap": ("05_fingerprint", fingerprint_heatmap.render),
        "05_top_contacts": ("05_fingerprint", top_contacts_table.render),
        "06_mutation_design_space": ("06_mutation", mutation_map.render),
        "07_score_waterfall": ("07_reranking", score_waterfall.render),
        "08_md_rmsd_timeseries": ("08_md", md_rmsd.render),
        "08_md_key_distances": ("08_md", md_key_distances.render),
        "09_library_diversity": ("09_final_library", library_diversity.render),
    }


# ---------------------------------------------------------------------------
# Section skeleton.
# ---------------------------------------------------------------------------

_SECTION_TITLES: Dict[str, str] = {
    "00_overview": "Run Overview",
    "01_input": "Input / Target",
    "02_msa": "MSA / Homolog",
    "03_boltz_complex": "Boltz Complex",
    "04_pose_ensemble": "Pose Ensemble",
    "05_fingerprint": "Interaction Fingerprint",
    "06_mutation": "Mutation Generation",
    "07_reranking": "Reranking / Validation",
    "08_md": "MD Validation",
    "09_final_library": "Final Library",
}


def _empty_sections() -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "title": title,
            "figures": [],
            "tables": [],
            "viewers": [],
            "notes": [],
        }
        for key, title in _SECTION_TITLES.items()
    }


# ---------------------------------------------------------------------------
# Run metadata.
# ---------------------------------------------------------------------------


def _read_state(artifacts: ReportArtifacts) -> Dict[str, Any]:
    if artifacts.state_json is None or not Path(artifacts.state_json).exists():
        return {}
    try:
        return json.loads(Path(artifacts.state_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_provenance(artifacts: ReportArtifacts) -> Dict[str, Any]:
    if artifacts.provenance_json is None or not Path(artifacts.provenance_json).exists():
        return {}
    try:
        return json.loads(Path(artifacts.provenance_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _evidence_summary(artifacts: ReportArtifacts) -> Dict[str, int]:
    """Counter-style breakdown of evidence_class in final_candidates.csv."""
    counts: Dict[str, int] = {}
    csv_path = artifacts.final_candidates_csv
    if csv_path is None or not Path(csv_path).exists():
        return counts
    try:
        with Path(csv_path).open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                ec = (row.get("evidence_class") or "Uncertain").strip() or "Uncertain"
                counts[ec] = counts.get(ec, 0) + 1
    except OSError:
        return counts
    return counts


def _final_candidate_rows(artifacts: ReportArtifacts) -> List[Dict[str, Any]]:
    csv_path = artifacts.final_candidates_csv
    if csv_path is None or not Path(csv_path).exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with Path(csv_path).open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                rows.append({k: (v or "") for k, v in row.items()})
    except OSError:
        return []
    return rows


def _catalytic_targets(state: Dict[str, Any], provenance: Dict[str, Any]) -> List[str]:
    """Best-effort recovery of catalytic residue labels (e.g. ``D222``)."""
    candidates = (
        state.get("catalytic_targets"),
        state.get("catalytic_residues"),
        provenance.get("catalytic_targets"),
        provenance.get("catalytic_residues"),
    )
    for blob in candidates:
        if isinstance(blob, list) and blob:
            return [str(x) for x in blob]
    return []


def _run_title(artifacts: ReportArtifacts, state: Dict[str, Any]) -> str:
    tgt = state.get("target") or state.get("target_id")
    if tgt:
        return f"EvoLiEZ Report - {tgt}"
    if artifacts.run_dir is not None:
        return f"EvoLiEZ Report - {Path(artifacts.run_dir).name}"
    return "EvoLiEZ Report"


# ---------------------------------------------------------------------------
# Static asset copying.
# ---------------------------------------------------------------------------


def _copy_static_assets(output_dir: Path) -> None:
    """Copy the package's static/ + vendor/ into ``output_dir/static/``.

    Missing assets log + skip rather than raise; ``visual_report.html``
    references them with relative URLs but the rest of the report stays
    functional even if (say) the vendor JS hasn't been vendored yet.
    """
    out_static = output_dir / "static"
    out_static.mkdir(parents=True, exist_ok=True)
    (out_static / "vendor").mkdir(parents=True, exist_ok=True)

    pairs = [
        (STATIC_DIR / "report.css", out_static / "report.css"),
        (STATIC_DIR / "report.js", out_static / "report.js"),
        (VENDOR_DIR / "3Dmol-min.js", out_static / "vendor" / "3Dmol-min.js"),
    ]
    for src, dst in pairs:
        if not src.exists():
            _LOG.warning("static asset missing: %s", src)
            continue
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            _LOG.warning("could not copy %s -> %s: %s", src, dst, exc)


def _copy_pdb_assets(
    artifacts: ReportArtifacts, output_dir: Path
) -> Dict[str, Path]:
    """Copy WT + top-candidate PDBs into ``output_dir/pdb/``.

    Returns ``{label: dest_path}`` for any successful copy.  Best-effort -
    missing source files simply skip.
    """
    pdb_dir = output_dir / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    wt = artifacts.wt_complex_pdb
    if wt is not None and Path(wt).exists():
        dst = pdb_dir / "wt.pdb"
        try:
            shutil.copy2(wt, dst)
            out["wt"] = dst
        except OSError as exc:
            _LOG.warning("could not copy WT PDB: %s", exc)

    # Top 3 mutant PDBs (alphabetical for determinism).
    for label, src in list(artifacts.mutant_complex_pdbs.items())[:3]:
        if not Path(src).exists():
            continue
        dst = pdb_dir / f"mut_{label}.pdb"
        try:
            shutil.copy2(src, dst)
            out[label] = dst
        except OSError as exc:
            _LOG.warning("could not copy mutant PDB %s: %s", label, exc)

    return out


# ---------------------------------------------------------------------------
# FigureSpec helpers - rewrite paths to be relative to output_dir.
# ---------------------------------------------------------------------------


def _rebase_figure_path(spec: FigureSpec, output_dir: Path) -> FigureSpec:
    """Rewrite ``spec.path`` to be relative to ``output_dir`` for the HTML.

    Renderers write absolute paths; the HTML templates emit them verbatim
    into ``<img src="...">`` so we want them relative for portability.
    Idempotent for already-relative paths.
    """
    p = Path(spec.path)
    if p.is_absolute():
        try:
            spec.path = Path(p.relative_to(output_dir))
        except ValueError:
            # Path is absolute but not under output_dir - leave as is.
            spec.path = p
    return spec


def _spec_to_template_dict(spec: FigureSpec) -> Dict[str, Any]:
    """``FigureSpec`` -> dict the figure_card template iterates over."""
    raw = asdict(spec)
    raw["path"] = Path(raw["path"]).as_posix()
    raw["source_files"] = [Path(p).as_posix() for p in raw.get("source_files", [])]
    return raw


# ---------------------------------------------------------------------------
# 3D viewers + PyMOL renderers.
# ---------------------------------------------------------------------------


def _build_3d_section_assets(
    artifacts: ReportArtifacts,
    figures_dir: Path,
    *,
    style: str,
    skip_3d: bool,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[FigureSpec], List[FigureSpec]]:
    """Build {section_key: [viewer dicts]} + PyMOL specs + inline-fallback specs.

    The 3Dmol inline viewers always work (no native dep); PyMOL PNG
    renders are only attempted when the binary is on PATH and
    ``skip_3d=False``.

    For each of the three section-specific PyMOL renderers (pocket,
    pose-ensemble, mutation-overlay), we *always* attempt the call.  If
    the renderer returns ``None`` (PyMOL absent, render failed, etc.) and
    a WT complex PDB is available, we substitute a 3Dmol.js inline
    viewer for that section and record a placeholder ``FigureSpec`` with
    ``renderer="3dmol-inline"`` so the manifest tracks the fallback.

    Returns a 3-tuple: ``(viewers_per_section, pymol_specs,
    inline_fallback_specs)``.  ``pymol_specs`` lists successful PyMOL PNG
    renders; ``inline_fallback_specs`` lists the placeholder specs for
    inline viewer substitutions.
    """
    viewers_per_section: Dict[str, List[Dict[str, Any]]] = {
        "03_boltz_complex": [],
        "04_pose_ensemble": [],
        "08_md": [],
        "09_final_library": [],
    }
    pymol_specs: List[FigureSpec] = []
    inline_fallback_specs: List[FigureSpec] = []

    if skip_3d:
        return viewers_per_section, pymol_specs, inline_fallback_specs

    wt_pdb = (
        Path(artifacts.wt_complex_pdb)
        if artifacts.wt_complex_pdb is not None
        and Path(artifacts.wt_complex_pdb).exists()
        else None
    )

    # ---- 3Dmol inline viewers (always safe, independent of PyMOL) ------
    # These are the legacy unconditional viewers that pre-date the
    # per-renderer fallback below; they still ship by default so users
    # always get an interactive WT view in section 03 and per-mutant
    # views in section 09.
    if wt_pdb is not None:
        try:
            ctx = make_viewer_context(
                wt_pdb,
                viewer_id="wt-complex",
                height=480,
                title="WT-ligand complex",
            )
            if ctx and not ctx.get("missing"):
                viewers_per_section["03_boltz_complex"].append(ctx)
        except Exception as exc:  # noqa: BLE001 - sibling agent may be flaky
            _LOG.warning("3Dmol viewer (WT) failed: %s", exc)

    # Top 3 mutant viewers for section 09 (final library).
    for label, pdb in list(artifacts.mutant_complex_pdbs.items())[:3]:
        try:
            ctx = make_viewer_context(
                Path(pdb),
                viewer_id=f"mut-{label}",
                height=360,
                title=f"Mutant: {label}",
            )
            if ctx and not ctx.get("missing"):
                viewers_per_section["09_final_library"].append(ctx)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("3Dmol viewer (mut %s) failed: %s", label, exc)

    # ---- Per-renderer PyMOL attempts + 3Dmol fallback -------------------
    # Each renderer is invoked unconditionally; the renderers themselves
    # short-circuit (returning None) when PyMOL is unavailable or inputs
    # are missing.  When None comes back AND we have a WT PDB to fall
    # back on, we build a 3Dmol inline viewer for the section so the
    # report never shows an empty 3D pane.
    pymol_section_map = {
        "03_boltz_binding_pocket": "03_boltz_complex",
        "04_boltz_pose_ensemble": "04_pose_ensemble",
        "09_final_library_structure": "09_final_library",
    }
    for fid, render_fn in (
        ("03_boltz_binding_pocket", render_pocket_view),
        ("04_boltz_pose_ensemble", render_pose_ensemble),
        ("09_final_library_structure", render_mutation_overlay),
    ):
        out_path = figures_dir / f"{fid}.png"
        spec: Optional[FigureSpec] = None
        try:
            spec = render_fn(artifacts, out_path, style=style)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("PyMOL %s failed: %s", fid, exc)
            spec = None

        if spec is not None:
            # Force the figure_id to match our manifest layout.
            spec.figure_id = fid
            pymol_specs.append(spec)
            continue

        # PyMOL render unavailable - try 3Dmol fallback if WT exists.
        section_key = pymol_section_map[fid]
        fallback_pairs = _build_inline_fallback(
            fid, artifacts, wt_pdb=wt_pdb
        )
        if not fallback_pairs:
            continue
        for viewer_ctx, fallback_spec in fallback_pairs:
            viewers_per_section[section_key].append(viewer_ctx)
            inline_fallback_specs.append(fallback_spec)

    return viewers_per_section, pymol_specs, inline_fallback_specs


def _build_inline_fallback(
    figure_id: str,
    artifacts: ReportArtifacts,
    *,
    wt_pdb: Optional[Path],
) -> List[Tuple[Dict[str, Any], FigureSpec]]:
    """Build (viewer_context, FigureSpec) pairs for a PyMOL-renderer fallback.

    Returns an empty list when no fallback can be built (typically
    because no WT PDB is available).  Returns 1 pair for sections 4 and
    9; section 3 returns 2 pairs (full pLDDT-colored view + pocket
    close-up) so the Boltz complex section satisfies the expert plan's
    "predicted 3D structure + binding pocket close-up + Boltz
    confidence coloring" trio.
    """
    if wt_pdb is None:
        return []

    if figure_id == "03_boltz_binding_pocket":
        pairs: List[Tuple[Dict[str, Any], FigureSpec]] = []
        # ---- Viewer 1: full structure, pLDDT (B-factor) colored. -----
        v1_id = "wt_pocket_plddt"
        v1_title = "WT complex (pLDDT confidence-colored)"
        v1_desc = (
            "3Dmol.js inline viewer (file://-safe). Cartoon colored by "
            "Boltz per-residue pLDDT confidence (red < 50, orange < 70, "
            "yellow < 90, blue >= 90)."
        )
        v1_ctx = make_viewer_context(
            wt_pdb,
            viewer_id=v1_id,
            height=480,
            title=v1_title,
            color_by="plddt",
            pocket_zoom=None,
        )
        if v1_ctx and not v1_ctx.get("missing"):
            v1_spec = FigureSpec(
                figure_id="03_boltz_complex_3dmol",
                section="boltz_complex",
                title="WT complex (pLDDT-colored, interactive)",
                description=v1_desc,
                # No on-disk file -- the viewer lives in <script
                # type="text/plain"> inside the rendered HTML.  Use
                # ``Path(".")`` so the bundler's "figure file exists"
                # check resolves to the report root and never errors;
                # ``params["inline"] = True`` is the real marker.
                path=Path("."),
                source_files=[Path(wt_pdb)],
                renderer="3dmol-inline",
                params={
                    "inline": True,
                    "viewer_id": v1_id,
                    "inline_uri": f"inline://{v1_id}",
                    "color_by": "plddt",
                },
            )
            pairs.append((v1_ctx, v1_spec))

        # ---- Viewer 2: pocket close-up, spectrum (default) colored. --
        v2_id = "wt_pocket_closeup"
        v2_title = "Binding pocket close-up"
        v2_desc = (
            "3Dmol.js inline viewer (file://-safe). Zoomed to ~8 A "
            "around the ligand so reviewers can see the pocket "
            "interactions Boltz predicted."
        )
        v2_ctx = make_viewer_context(
            wt_pdb,
            viewer_id=v2_id,
            height=480,
            title=v2_title,
            color_by="spectrum",
            pocket_zoom=8.0,
        )
        if v2_ctx and not v2_ctx.get("missing"):
            v2_spec = FigureSpec(
                figure_id="03_boltz_complex_3dmol_closeup",
                section="boltz_complex",
                title="Binding pocket close-up (interactive)",
                description=v2_desc,
                path=Path("."),
                source_files=[Path(wt_pdb)],
                renderer="3dmol-inline",
                params={
                    "inline": True,
                    "viewer_id": v2_id,
                    "inline_uri": f"inline://{v2_id}",
                    "color_by": "spectrum",
                    "pocket_zoom": 8.0,
                },
            )
            pairs.append((v2_ctx, v2_spec))

        return pairs

    if figure_id == "04_boltz_pose_ensemble":
        # Prefer the best Boltz model_0.pdb when we can find it; fall
        # back to the WT complex itself otherwise.  The PNG ensemble
        # overlay was the value-add for paper figures; the inline
        # viewer's job is interactive review.
        pose_pdb = _best_pose_pdb(artifacts) or wt_pdb
        viewer_id = "pose_ensemble_inline"
        title = "WT pose (3Dmol interactive viewer)"
        description = (
            "3Dmol.js inline viewer (file://-safe). PyMOL pose-ensemble "
            "overlay unavailable - rotate/zoom in browser."
        )
        viewer_ctx = make_viewer_context(
            pose_pdb,
            viewer_id=viewer_id,
            height=480,
            title=title,
        )
        if not viewer_ctx or viewer_ctx.get("missing"):
            return []
        spec = FigureSpec(
            figure_id="04_pose_ensemble_3dmol",
            section="pose_ensemble",
            title="WT pose (interactive)",
            description=description,
            path=Path("."),
            source_files=[Path(pose_pdb)],
            renderer="3dmol-inline",
            params={
                "inline": True,
                "viewer_id": viewer_id,
                "inline_uri": f"inline://{viewer_id}",
            },
        )
        return [(viewer_ctx, spec)]

    if figure_id == "09_final_library_structure":
        # Highlight residues come from the top candidates' mutation
        # tokens; the PyMOL renderer ignores the candidate list when
        # absent, so we mirror that and skip the fallback if nothing
        # is available.
        highlight_positions = _top_mutation_positions(artifacts, top_k=10)
        if not highlight_positions:
            return []
        viewer_id = "final_library_structure_inline"
        title = "Top-K mutation positions on WT (interactive)"
        description = (
            "3Dmol.js inline viewer (file://-safe). PyMOL mutation "
            "overlay unavailable - rotate/zoom in browser."
        )
        viewer_ctx = make_viewer_context(
            wt_pdb,
            viewer_id=viewer_id,
            height=480,
            highlight_residues=highlight_positions,
            title=title,
        )
        if not viewer_ctx or viewer_ctx.get("missing"):
            return []
        source_files: List[Path] = [Path(wt_pdb)]
        csv_path = artifacts.final_candidates_csv
        if csv_path is not None and Path(csv_path).exists():
            source_files.append(Path(csv_path))
        spec = FigureSpec(
            figure_id="09_final_library_3dmol",
            section="final_library",
            title="Top-K mutation positions (interactive)",
            description=description,
            path=Path("."),
            source_files=source_files,
            renderer="3dmol-inline",
            params={
                "inline": True,
                "viewer_id": viewer_id,
                "inline_uri": f"inline://{viewer_id}",
                "highlight_residues": list(highlight_positions),
            },
        )
        return [(viewer_ctx, spec)]

    return []


def _best_pose_pdb(artifacts: ReportArtifacts) -> Optional[Path]:
    """Return the best WT Boltz pose (``*_model_0.pdb``) if discoverable.

    Mirrors :mod:`evoliez.figures.three_d.pose_ensemble` lookup so the
    inline fallback shows the same primary pose the PyMOL overlay would
    have used as its reference.  Returns ``None`` when nothing is found.
    """
    run_dir = getattr(artifacts, "run_dir", None)
    if run_dir is None:
        return None
    pred_root = (
        Path(run_dir)
        / "complexes"
        / "boltz_results_wt_boltz_input"
        / "predictions"
        / "wt_boltz_input"
    )
    if pred_root.exists():
        hits = sorted(pred_root.glob("*_model_0.pdb"))
        if hits:
            return hits[0]
    alt_root = (
        Path(run_dir)
        / "complexes"
        / "boltz_results_wt_boltz_input"
        / "predictions"
    )
    if alt_root.exists():
        hits = sorted(alt_root.rglob("*_model_0.pdb"))
        if hits:
            return hits[0]
    return None


def _top_mutation_positions(
    artifacts: ReportArtifacts, *, top_k: int = 10
) -> List[int]:
    """Parse ``final_candidates.csv`` for the top-K rows' mutation positions.

    Mirrors :mod:`evoliez.figures.three_d.mutation_overlay` parsing so the
    inline fallback highlights the same residues PyMOL would have. Returns
    an ordered, de-duplicated list (best-first).
    """
    csv_path = getattr(artifacts, "final_candidates_csv", None)
    if csv_path is None or not Path(csv_path).exists():
        return []
    import re

    token_re = re.compile(r"^[A-Z](?P<pos>\d+)[A-Z*]$")
    out: List[int] = []
    seen: set = set()
    try:
        with Path(csv_path).open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i >= max(1, int(top_k)):
                    break
                raw = row.get("mutations") or row.get("mutation") or ""
                for tok in re.split(r"[,;+\s]+", str(raw).strip()):
                    if not tok:
                        continue
                    m = token_re.match(tok.strip())
                    if not m:
                        continue
                    try:
                        pos = int(m.group("pos"))
                    except ValueError:
                        continue
                    if pos in seen:
                        continue
                    seen.add(pos)
                    out.append(pos)
    except OSError:
        return []
    return out


# ---------------------------------------------------------------------------
# Tables for downstream sections.
# ---------------------------------------------------------------------------


def _per_sample_table(artifacts: ReportArtifacts) -> Optional[Dict[str, Any]]:
    """Pose-ensemble summary table from final_candidates.csv.

    Falls back to a minimal table from the mutant PDB list when the CSV
    isn't present.  Returns ``None`` if neither source has anything.
    """
    rows = _final_candidate_rows(artifacts)
    if rows:
        keep = [
            c for c in (
                "candidate_id",
                "mutation",
                "evidence_class",
                "final_score",
                "docking_score",
                "md_lite_score",
            )
            if any(c in r for r in rows)
        ]
        if not keep:
            keep = list(rows[0].keys())[:6]
        return {
            "id": "pose_ensemble_summary",
            "columns": keep,
            "rows": [{c: r.get(c, "") for c in keep} for r in rows[:20]],
        }
    if artifacts.mutant_complex_pdbs:
        return {
            "id": "pose_ensemble_summary",
            "columns": ["candidate", "pdb"],
            "rows": [
                {"candidate": k, "pdb": Path(v).name}
                for k, v in list(artifacts.mutant_complex_pdbs.items())[:20]
            ],
        }
    return None


def _final_library_table(artifacts: ReportArtifacts) -> Optional[Dict[str, Any]]:
    rows = _final_candidate_rows(artifacts)
    if not rows:
        return None
    keep = [
        c for c in (
            "candidate_id",
            "mutation",
            "evidence_class",
            "final_score",
            "ml_score",
            "ddg_fold",
            "docking_score",
            "md_lite_score",
            "plif_recovery",
        )
        if any(c in r for r in rows)
    ]
    if not keep:
        keep = list(rows[0].keys())
    return {
        "id": "final_library",
        "columns": keep,
        "rows": [{c: r.get(c, "") for c in keep} for r in rows],
    }


def _read_target_sequence(fasta_path: Optional[Path]) -> str:
    """Best-effort FASTA reader - returns the joined sequence string.
    Handles multi-line records; ignores comments and the header line.
    Returns '' on any failure (caller short-circuits the sequence panel).
    """
    if fasta_path is None or not Path(fasta_path).exists():
        return ""
    try:
        text = Path(fasta_path).read_text()
    except OSError:
        return ""
    out: list = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith((">", "#", ";")):
            continue
        out.append(line)
    return "".join(out)


def _sequence_track_table(artifacts: ReportArtifacts) -> Optional[Dict[str, Any]]:
    """Compact "what was given" summary for section 01.

    Two-row table: first column is the field name, second is the value.
    More readable than the wide one-row layout when the SMILES is long.
    Includes sequence length here (the FULL sequence text gets its own
    monospace panel below the table - too long for a cell).
    """
    bits: List[tuple] = []
    if artifacts.target_fasta is not None and Path(artifacts.target_fasta).exists():
        bits.append(("target_fasta", Path(artifacts.target_fasta).name))
        seq = _read_target_sequence(artifacts.target_fasta)
        if seq:
            bits.append(("sequence_length_aa", str(len(seq))))
    if artifacts.ligand_smiles:
        bits.append(("ligand_smiles", artifacts.ligand_smiles))
    if not bits:
        return None
    return {
        "id": "sequence_track",
        "columns": ["field", "value"],
        "rows": [{"field": k, "value": v} for k, v in bits],
    }


def _sequence_panel(artifacts: ReportArtifacts) -> Optional[Dict[str, Any]]:
    """Full protein sequence as a monospace panel for section 01.

    Wraps to 60 chars per line with 10-residue numbering so users can
    visually find catalytic / binding positions (e.g. R285, D222) in the
    actual sequence. Without this panel the input section only shows
    `target_fasta: target.fasta` (the filename!) - the user-reported
    "input에 단백질 seq가 안나오는데" bug.
    """
    seq = _read_target_sequence(artifacts.target_fasta)
    if not seq:
        return None
    # 60 chars per line is the FASTA convention; group in 10s for
    # eyeball position counting.
    LINE = 60
    GROUP = 10
    lines: List[str] = []
    for i in range(0, len(seq), LINE):
        chunk = seq[i:i + LINE]
        groups = " ".join(chunk[j:j + GROUP] for j in range(0, len(chunk), GROUP))
        # leading 1-based residue number (4 chars right-aligned)
        lines.append(f"{i + 1:>4}  {groups}")
    return {
        "id": "target_sequence",
        "title": "Target protein sequence",
        "length_aa": len(seq),
        "text": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------


def build_report(
    run_dir: Path,
    *,
    output_dir: Optional[Path] = None,
    style: str = "presentation",
    html_mode: str = "linked",
    skip_3d: bool = False,
    benchmark_csv: Optional[Path] = None,
    mock_backend: bool = False,
    output_zip: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a portable HTML report package for a finished pipeline run.

    See module docstring for the full workflow; in short:

    1. ``discover()`` the artifacts.
    2. Apply matplotlib style.
    3. Render each registered 2D plot (and best-effort PyMOL PNG).
    4. Build 3Dmol inline viewers for PDB artifacts.
    5. Render the jinja2 templates against the resulting sections dict.
    6. Write ``visual_report.html`` + ``visual_manifest.json``.
    7. Mock watermark + zip / self-contained packaging.

    The function never raises for missing optional inputs - missing data
    becomes a warning in the manifest and a skipped figure in the HTML.
    """
    run_dir = Path(run_dir).resolve()
    if output_dir is None:
        output_dir = run_dir / "reports"
    output_dir = Path(output_dir).resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if html_mode not in ("linked", "self-contained"):
        raise ValueError(
            f"html_mode must be 'linked' or 'self-contained' (got {html_mode!r})"
        )

    # ---- 1. discover ----------------------------------------------------
    artifacts = discover(run_dir)
    if benchmark_csv is not None:
        artifacts.benchmark_csv = Path(benchmark_csv)

    # ---- 2. style -------------------------------------------------------
    try:
        apply_mpl_style(style)
    except Exception as exc:  # noqa: BLE001 - matplotlib optional
        _LOG.warning("could not apply matplotlib style %s: %s", style, exc)

    # ---- 3. manifest ----------------------------------------------------
    manifest = ManifestBuilder(
        run_dir=run_dir, style=style, mock_backend=mock_backend
    )

    # ---- 4. sections skeleton ------------------------------------------
    sections = _empty_sections()

    # ---- 5. 2D plot renderers ------------------------------------------
    figures_generated = 0
    figures_skipped = 0
    dispatch = _plot_dispatch()
    for fid, (section_key, render_fn) in dispatch.items():
        out_path = figures_dir / f"{fid}.png"
        spec: Optional[FigureSpec] = None
        try:
            spec = render_fn(artifacts, out_path, style=style)
        except Exception as exc:  # noqa: BLE001 - any plot failure
            _LOG.warning("plot %s failed: %s", fid, exc)
            manifest.add_warning(f"{fid}: {exc}")
            spec = None

        if spec is None or not Path(spec.path).exists():
            figures_skipped += 1
            sections[section_key]["notes"].append(
                f"Figure {fid} skipped (missing input or renderer)."
            )
            continue
        # Force a consistent figure_id (some renderers default to their own).
        spec.figure_id = fid
        spec = _rebase_figure_path(spec, output_dir)
        manifest.add_figure(spec)
        sections[section_key]["figures"].append(_spec_to_template_dict(spec))
        figures_generated += 1

    # ---- 6. 3D viewers + PyMOL ------------------------------------------
    viewer_buckets, pymol_specs, inline_fallback_specs = _build_3d_section_assets(
        artifacts, figures_dir, style=style, skip_3d=skip_3d
    )
    for sec_key, viewers in viewer_buckets.items():
        sections[sec_key]["viewers"].extend(viewers)
    pymol_section_map = {
        "03_boltz_binding_pocket": "03_boltz_complex",
        "04_boltz_pose_ensemble": "04_pose_ensemble",
        "09_final_library_structure": "09_final_library",
    }
    for spec in pymol_specs:
        if spec is None or not Path(spec.path).exists():
            figures_skipped += 1
            continue
        spec = _rebase_figure_path(spec, output_dir)
        manifest.add_figure(spec)
        section_key = pymol_section_map.get(spec.figure_id, "03_boltz_complex")
        sections[section_key]["figures"].append(_spec_to_template_dict(spec))
        figures_generated += 1

    # 3Dmol inline fallback FigureSpecs: provenance-only - ``path``
    # is ``"."`` (the report root) because the actual viewer lives in
    # the rendered HTML's ``<script type="text/plain">`` block, not on
    # disk.  ``params["inline"] = True`` is the real marker.  We DON'T
    # rebase the path against ``output_dir`` (it's already a root-relative
    # sentinel) and we DON'T add the spec to ``sections[..]["figures"]``
    # (would render as a broken ``<img>``).  The viewer itself was
    # appended to ``sections[..]["viewers"]`` above.
    for spec in inline_fallback_specs:
        manifest.add_figure(spec)

    # ---- 7. tables ------------------------------------------------------
    seq_table = _sequence_track_table(artifacts)
    if seq_table is not None:
        sections["01_input"]["tables"].append(seq_table)
    # User-reported bug ("input에 단백질 seq가 안나오는데"): the input
    # section showed only the FASTA filename, not the actual sequence.
    # Add a monospace sequence panel under the metadata table so the
    # reader can verify catalytic / binding positions visually.
    seq_panel = _sequence_panel(artifacts)
    if seq_panel is not None:
        sections["01_input"].setdefault("panels", []).append(seq_panel)
    pose_table = _per_sample_table(artifacts)
    if pose_table is not None:
        sections["04_pose_ensemble"]["tables"].append(pose_table)
    final_table = _final_library_table(artifacts)
    if final_table is not None:
        sections["09_final_library"]["tables"].append(final_table)

    # ---- 8. section 05 contact-lines viewer ----------------------------
    # The 2D figures (fingerprint heatmap + top contacts table) are wired
    # via the standard dispatch above; here we attach the 3Dmol viewer
    # that overlays family-consensus dashed contact lines on the WT
    # complex.  ``contact_lines.render`` returns ``None`` when WT PDB or
    # edge_level.csv is missing - in that case we re-attach the legacy
    # placeholder note so the section explains why it's empty rather
    # than just rendering a blank pane.
    if not skip_3d:
        try:
            from evoliez.figures.three_d.contact_lines import (
                render as render_contact_lines,
            )

            ctx = render_contact_lines(artifacts)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("contact_lines viewer failed: %s", exc)
            ctx = None
        if ctx is not None:
            sections["05_fingerprint"]["viewers"].append(ctx)
    if (
        not sections["05_fingerprint"]["figures"]
        and not sections["05_fingerprint"]["viewers"]
    ):
        sections["05_fingerprint"]["notes"].append(
            "Interaction-fingerprint outputs not yet available "
            "(stage s06b / s11 edge_level.csv missing); section deferred."
        )

    # ---- 8b. section 08 representative frames + trajectory movie -------
    # The 2D plots (RMSD time series + key distances) are wired via the
    # standard dispatch above.  Here we add the interactive
    # representative-frame viewers and (best-effort) the trajectory mp4.
    # md_frames returns a list of up to 3 viewer dicts (start/mid/end);
    # md_trajectory returns a FigureSpec when PyMOL + ffmpeg + a real DCD
    # are all present, else None (the static frames are the fallback).
    if not skip_3d:
        try:
            from evoliez.figures.three_d.md_frames import (
                render as render_md_frames,
            )

            frame_viewers = render_md_frames(artifacts)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("md_frames viewer failed: %s", exc)
            frame_viewers = []
        if frame_viewers:
            sections["08_md"]["viewers"].extend(frame_viewers)

        try:
            from evoliez.figures.movies.md_trajectory import (
                render as render_md_movie,
            )

            movies_dir = output_dir / "movies"
            movies_dir.mkdir(parents=True, exist_ok=True)
            movie_out = movies_dir / "08_md_trajectory.mp4"
            movie_spec = render_md_movie(artifacts, movie_out, style=style)
        except Exception as exc:  # noqa: BLE001 - movie path is best-effort
            _LOG.warning("md_trajectory movie failed: %s", exc)
            movie_spec = None
        if movie_spec is not None and Path(movie_spec.path).exists():
            movie_spec = _rebase_figure_path(movie_spec, output_dir)
            manifest.add_figure(movie_spec)
            sections["08_md"]["figures"].append(_spec_to_template_dict(movie_spec))
            figures_generated += 1

    # ---- 9. metadata for overview --------------------------------------
    state = _read_state(artifacts)
    provenance = _read_provenance(artifacts)
    evidence_summary = _evidence_summary(artifacts)
    sections["00_overview"]["evidence_summary"] = evidence_summary
    sections["00_overview"]["benchmark"] = (
        {"path": str(artifacts.benchmark_csv)}
        if artifacts.benchmark_csv is not None
        and Path(artifacts.benchmark_csv).exists()
        else None
    )
    catalytic = _catalytic_targets(state, provenance)

    # ---- 10. static assets ---------------------------------------------
    _copy_static_assets(output_dir)
    _copy_pdb_assets(artifacts, output_dir)

    # ---- 11. render templates ------------------------------------------
    html_path = _render_html(
        output_dir=output_dir,
        run_dir=run_dir,
        artifacts=artifacts,
        state=state,
        sections=sections,
        manifest=manifest,
        style=style,
        mock_backend=mock_backend,
        catalytic_targets=catalytic,
    )

    # ---- 12. write manifest --------------------------------------------
    manifest_path = output_dir / "visual_manifest.json"
    manifest.write(manifest_path)

    # ---- 13. mock watermark --------------------------------------------
    if mock_backend:
        _apply_watermark_safely(html_path)
        try:
            stamp_figures_with_mock(figures_dir)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("stamp_figures_with_mock failed: %s", exc)

    # ---- 14. packaging --------------------------------------------------
    zip_path: Optional[Path] = None
    if html_mode == "self-contained":
        sc_path = output_dir / "visual_report.selfcontained.html"
        try:
            build_self_contained(output_dir, sc_path)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("self-contained packaging failed: %s", exc)
            manifest.add_warning(f"self-contained packaging failed: {exc}")
    # Default zip path: <run_dir>/report_package.zip when caller didn't
    # specify - the whole point of `evoliez figures` is portability, so
    # a ZIP should exist by default. The CLI sets an explicit path; the
    # Python API gets this convenience default. (build_zip is also
    # called for self-contained mode so the user gets ONE archive
    # whether they chose linked or self-contained.)
    zip_path = (
        Path(output_zip)
        if output_zip is not None
        else (run_dir / "report_package.zip")
    )
    try:
        build_zip(output_dir, zip_path)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("zip packaging failed: %s", exc)
        manifest.add_warning(f"zip packaging failed: {exc}")
        zip_path = None

    # Rewrite the manifest one more time so any packaging warnings land.
    manifest.write(manifest_path)

    # ---- 15. return summary --------------------------------------------
    total_bytes = _dir_size_bytes(output_dir)
    if zip_path is not None and zip_path.exists():
        total_bytes = max(total_bytes, zip_path.stat().st_size)
    return {
        "html_path": html_path,
        "zip_path": zip_path,
        "manifest_path": manifest_path,
        "figures_generated": figures_generated,
        "figures_skipped": figures_skipped,
        "warnings": list(manifest.warnings),
        "mode": html_mode,
        "total_size_mb": total_bytes / (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Jinja2 rendering.
# ---------------------------------------------------------------------------


def _render_html(
    *,
    output_dir: Path,
    run_dir: Path,
    artifacts: ReportArtifacts,
    state: Dict[str, Any],
    sections: Dict[str, Dict[str, Any]],
    manifest: ManifestBuilder,
    style: str,
    mock_backend: bool,
    catalytic_targets: List[str],
) -> Path:
    """Render base.html against the sections dict; write visual_report.html."""
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:
        raise RuntimeError(
            "jinja2 is required to build the HTML report. "
            "Install with `pip install evoliez[figures]`."
        ) from exc

    # The base template uses ``{% include %}`` for partials, components,
    # and the static CSS/JS - so the loader needs both the templates/
    # directory and the package root (so 'static/report.css' resolves).
    loader = FileSystemLoader(
        [str(TEMPLATES_DIR), str(TEMPLATES_DIR.parent)]
    )
    env = Environment(
        loader=loader,
        autoescape=select_autoescape(enabled_extensions=("html", "htm"), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template("base.html")

    backend_label = "mock" if mock_backend else (state.get("backend") or "real")
    run_meta: Dict[str, Any] = {
        "title": _run_title(artifacts, state),
        "subtitle": state.get("subtitle"),
        "target": state.get("target") or state.get("target_id"),
        "ligand": artifacts.ligand_smiles,
        "backend": backend_label,
        "started": state.get("started") or state.get("start_time"),
        "finished": state.get("finished") or state.get("end_time"),
        "git_sha": manifest.git_sha,
        "config_sha1": manifest.config_sha1,
        "evoliez_version": _evoliez_version(),
        "generated_at": manifest.generated_at,
        "mock_backend": mock_backend,
        "catalytic_targets": catalytic_targets,
    }

    # Set the section variable per partial by rendering each partial in a
    # macro-free way: the base template just `{% include %}`s every
    # partial, and each partial reads ``section`` from the surrounding
    # scope.  jinja2 doesn't auto-bind the right section per include, so
    # we render each partial individually then build a single big HTML
    # blob that replaces the base.html's content block.
    # ...but base.html already does the include orchestration, so we
    # instead pass a *single* mapping ``sections`` and override base's
    # content block via a thin wrapper template.
    wrapper_src = _WRAPPER_TEMPLATE
    wrapper = env.from_string(wrapper_src)

    rendered_sections: List[str] = []
    section_partials = [
        ("00_overview", "partials/00_overview.html"),
        ("01_input", "partials/01_input.html"),
        ("02_msa", "partials/02_msa.html"),
        ("03_boltz_complex", "partials/03_boltz_complex.html"),
        ("04_pose_ensemble", "partials/04_pose_ensemble.html"),
        ("05_fingerprint", "partials/05_fingerprint.html"),
        ("06_mutation", "partials/06_mutation.html"),
        ("07_reranking", "partials/07_reranking.html"),
        ("08_md", "partials/08_md.html"),
        ("09_final_library", "partials/09_final_library.html"),
    ]
    for sec_key, partial_path in section_partials:
        try:
            partial = env.get_template(partial_path)
            rendered_sections.append(
                partial.render(
                    section=sections[sec_key],
                    run=run_meta,
                    manifest={"warnings": manifest.warnings},
                )
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("partial %s render failed: %s", partial_path, exc)
            manifest.add_warning(f"{partial_path}: {exc}")
            rendered_sections.append(
                f"<section id=\"{sec_key.replace('_', '-')}\" class=\"report-section\">"
                f"<h2>{sections[sec_key]['title']}</h2>"
                f"<p class=\"section-empty\">Render error: {exc}</p>"
                "</section>"
            )

    html_blob = "\n".join(rendered_sections)
    rendered = wrapper.render(
        run=run_meta,
        manifest={"warnings": manifest.warnings},
        prebuilt_sections=html_blob,
    )

    html_path = output_dir / "visual_report.html"
    html_path.write_text(rendered, encoding="utf-8")
    return html_path


# A thin wrapper around base.html that lets us pre-render the section
# partials with their own ``section`` context (base.html's plain
# ``{% include %}`` would re-use the same outer scope for all 10 partials).
_WRAPPER_TEMPLATE = """\
{% extends "base.html" %}
{% block content %}
{{ prebuilt_sections|safe }}
{% endblock %}
"""


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _apply_watermark_safely(html_path: Path) -> None:
    """Call ``apply_mock_watermark`` adapting to both possible signatures.

    The sibling packaging agent ships ``apply_mock_watermark(html_text:
    str) -> str``; our local fallback (and earlier prototypes) take a
    Path and mutate the file in place.  We try the text-in/text-out form
    first since that's what the released sibling uses.
    """
    if not html_path.exists():
        return
    try:
        original = html_path.read_text(encoding="utf-8")
    except OSError as exc:
        _LOG.warning("could not read %s for watermark: %s", html_path, exc)
        return
    try:
        stamped = apply_mock_watermark(original)
        if isinstance(stamped, str):
            if stamped != original:
                html_path.write_text(stamped, encoding="utf-8")
            return
    except (TypeError, AttributeError):
        # Signature didn't accept the string - fall through to the
        # path-based form below.
        pass
    except Exception as exc:  # noqa: BLE001 - never block on cosmetic
        _LOG.warning("apply_mock_watermark (text form) failed: %s", exc)
        return
    # Path-based fallback (older / stub implementations).
    try:
        apply_mock_watermark(html_path)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("apply_mock_watermark (path form) failed: %s", exc)


def _dir_size_bytes(d: Path) -> int:
    total = 0
    for p in Path(d).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _evoliez_version() -> str:
    try:
        from evoliez import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


__all__ = ["build_report"]
