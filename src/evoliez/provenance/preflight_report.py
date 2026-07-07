"""V4-1 preflight report generator (ROADMAP_V4 §7.2).

Runs :func:`evoliez.provenance.preflight.classify_artifact` over the geometry/structure
artifacts the reference-ensemble stage (V4-2) consumes, and renders a markdown report
that lists each as ``valid`` / ``usable_with_warning`` / ``skipped_not_validated`` /
``invalid`` **with a reason**. This is the report the roadmap names ``v4_preflight_report.md``
and the next-phase gate reads before V4-2.

Honest by construction: production WT/reference structures and v2/v3 MD trajectories live
on the compute server, so in a local checkout they are reported as
``skipped_not_validated`` (reason: not present locally) rather than silently passing. The
reactive atom map derived from the mechanism template is declared but *unverified* against
a real structure, so geometry artifacts are not marked ``valid`` until a structure-verified
atom map exists — which is precisely why V4-2 stays on the v0 (recorded-evidence) path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .artifacts import ArtifactProvenance
from .atom_map import ReactiveAtomMap
from .preflight import (
    SKIPPED_NOT_VALIDATED,
    PreflightResult,
    classify_artifact,
)


@dataclass
class ReferenceArtifactSpec:
    """One geometry artifact V4-2 needs, and where it is expected to live."""

    name: str
    role: str  # wt_reference_complex | v2_md_snapshot | v3_md_snapshot | redock_pose | example
    path: str
    backend: str = "unknown"
    source_stage: str = "unknown"
    requires_pose_rmsd: bool = False


def _scan_structure(path: Path) -> tuple[bool, int]:
    """Cheap full-atom check for a PDB/CIF: (is_full_atom, atom_count).

    Full-atom means the structure has protein atoms beyond the CA backbone trace. We scan
    ATOM/HETATM records and look for any non-CA protein atom name; a CA-only trace scores
    ``is_full_atom=False`` so preflight hard-fails it (no silent CA-only pass).
    """
    atom_count = 0
    has_non_ca = False
    try:
        with path.open("r", errors="ignore") as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")):
                    atom_count += 1
                    atom_name = line[12:16].strip() if len(line) >= 16 else ""
                    if atom_name and atom_name != "CA":
                        has_non_ca = True
    except OSError:
        return False, 0
    return has_non_ca, atom_count


def _classify_spec(
    spec: ReferenceArtifactSpec, atom_map: Optional[ReactiveAtomMap]
) -> PreflightResult:
    path = Path(spec.path)
    if not path.exists():
        # Not present in this environment (server-resident): honest skip, never a silent pass.
        placeholder = ArtifactProvenance(
            structure_path=spec.path,
            source_stage=spec.source_stage,
            backend=spec.backend,
        )
        return PreflightResult(
            artifact=placeholder,
            status=SKIPPED_NOT_VALIDATED,
            reasons=["artifact not present locally (server-resident); cannot validate"],
        )
    is_full_atom, atom_count = _scan_structure(path)
    artifact = ArtifactProvenance(
        structure_path=spec.path,
        source_stage=spec.source_stage,
        backend=spec.backend,
        atom_count=atom_count,
        is_full_atom=is_full_atom,
    )
    return classify_artifact(
        artifact, atom_map=atom_map, requires_pose_rmsd=spec.requires_pose_rmsd
    )


def collect_reference_artifacts(root: Path) -> List[ReferenceArtifactSpec]:
    """The geometry artifacts V4-2 depends on, keyed to their expected (production) paths.

    Production structures/trajectories live on the compute server; a local example
    structure is included when present so the classifier is exercised on a real file.
    """
    specs = [
        ReferenceArtifactSpec(
            name="wt_reference_complex",
            role="wt_reference_complex",
            path=str(root / "runs" / "fdh_prod_v3" / "complexes" / "wt_reference_complex.pdb"),
            backend="boltz",
            source_stage="s04_reference",
        ),
        ReferenceArtifactSpec(
            name="v2_md_snapshot",
            role="v2_md_snapshot",
            path=str(root / "runs" / "fdh_v2" / "md" / "wt" / "wt_snapshot.pdb"),
            backend="openmm",
            source_stage="s10_md",
        ),
        ReferenceArtifactSpec(
            name="v3_md_snapshot",
            role="v3_md_snapshot",
            path=str(root / "runs" / "fdh_prod_v3" / "md" / "wt" / "wt_snapshot.pdb"),
            backend="openmm",
            source_stage="s10_md",
        ),
        ReferenceArtifactSpec(
            name="redock_pose",
            role="redock_pose",
            path=str(root / "runs" / "fdh_prod_v3" / "validation" / "redock" / "wt_lig.pdb"),
            backend="gnina",
            source_stage="s06b_redock",
            requires_pose_rmsd=True,
        ),
    ]
    # A local, real full-atom structure exercises the classifier end-to-end when available.
    example = root / "runs" / "smoke" / "complexes" / "boltz" / "wt_complex.pdb"
    if example.exists():
        specs.append(
            ReferenceArtifactSpec(
                name="example_local_full_atom_structure",
                role="example",
                path=str(example),
                backend="boltz",
                source_stage="s04_reference",
            )
        )
    return specs


@dataclass
class PreflightSummary:
    results: List[tuple[ReferenceArtifactSpec, PreflightResult]] = field(default_factory=list)
    atom_map_validated: bool = False

    @property
    def ready_for_v4_2(self) -> bool:
        """The next-phase gate: every reference artifact must be valid or usable_with_warning."""
        return all(
            r.status in ("valid", "usable_with_warning") for _, r in self.results
        ) and bool(self.results)


def run_reference_preflight(
    root: Path, atom_map: Optional[ReactiveAtomMap] = None
) -> PreflightSummary:
    """Classify every reference artifact. ``atom_map`` defaults to an *unverified* template
    map (validated=False) — the honest local state, which keeps geometry artifacts out of
    ``valid`` until a structure-verified atom map is supplied."""
    if atom_map is None:
        # Declared from the hydride_transfer template SMARTS but not yet verified against a
        # structure -> validated=False on purpose (no silent full-atom/atom-map pass).
        atom_map = ReactiveAtomMap(source="hydride_transfer_template_smarts_unverified")
    specs = collect_reference_artifacts(root)
    results = [(spec, _classify_spec(spec, atom_map)) for spec in specs]
    return PreflightSummary(results=results, atom_map_validated=atom_map.validated)


def render_preflight_report(summary: PreflightSummary) -> str:
    lines = [
        "# V4 real-artifact preflight report",
        "",
        "Classifies the geometry/structure artifacts the ReferenceEnsemble stage (V4-2) "
        "consumes. Real-artifact failures are reported as `invalid` or "
        "`skipped_not_validated`, never converted into a neutral pass.",
        "",
        f"Reactive atom map validated: {summary.atom_map_validated}",
        "",
        "| artifact | role | status | reasons |",
        "|---|---|---|---|",
    ]
    counts: dict[str, int] = {}
    for spec, result in summary.results:
        counts[result.status] = counts.get(result.status, 0) + 1
        reasons = "; ".join(result.reasons) if result.reasons else "—"
        lines.append(
            f"| `{spec.name}` | {spec.role} | **{result.status}** | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
        ]
    )
    for status in ("valid", "usable_with_warning", "skipped_not_validated", "invalid"):
        lines.append(f"- {status}: {counts.get(status, 0)}")
    lines.extend(
        [
            "",
            "## Next-phase gate",
            "",
        ]
    )
    if summary.ready_for_v4_2:
        lines.append(
            "All reference artifacts are `valid` or `usable_with_warning`; V4-2 may consume "
            "real geometry."
        )
    else:
        lines.append(
            "Not all reference artifacts are validated in this environment. V4-2 must stay on "
            "the v0 recorded-evidence path until a structure-verified atom map and full-atom "
            "reference structures are available (server-resident). This is the honest, "
            "no-silent-fallback state, not a failure of the preflight."
        )
    return "\n".join(lines) + "\n"


def write_preflight_report(root: Path, out_path: Optional[Path] = None) -> PreflightSummary:
    summary = run_reference_preflight(root)
    out = out_path or (root / "reports" / "v4_preflight_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_preflight_report(summary))
    return summary
