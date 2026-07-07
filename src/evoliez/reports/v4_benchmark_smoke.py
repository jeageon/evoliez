"""Non-FDH benchmark smoke report (ROADMAP_V4 §7.11).

Renders a report that shows the mechanism-generalization path works config-only: the
non-FDH smoke configs load into :class:`MechanismSpec`, the reaction-state hard gate is
enforced, and each benchmark's label meaning + ClaimGuard status are pinned so a
fitness/DMS label can never drift into a kcat claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from evoliez.mechanism.benchmarks import (
    STAGE1,
    benchmark_extra_forbidden,
    benchmark_unlocks_kinetics,
)
from evoliez.mechanism.spec import MechanismSpec


def _load_mechanism_from_config(path: Path) -> MechanismSpec:
    data = yaml.safe_load(path.read_text())
    return MechanismSpec(**data["mechanism"])


def summarize_smoke_configs(config_paths: List[Path]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for path in config_paths:
        try:
            spec = _load_mechanism_from_config(path)
            out.append(
                {
                    "config": path.name,
                    "reaction_class": spec.reaction.cls,
                    "geometry_terms": len(spec.geometry_terms),
                    "loaded": True,
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - report the load failure, do not raise
            out.append(
                {
                    "config": path.name,
                    "reaction_class": "?",
                    "geometry_terms": 0,
                    "loaded": False,
                    "error": str(exc),
                }
            )
    return out


def render_benchmark_smoke_report(config_paths: List[Path]) -> str:
    lines = [
        "# V4 non-FDH benchmark smoke",
        "",
        "Mechanism generalization is exercised config-only. Without wet-lab data every "
        "benchmark remains hypothesis-grade (L0); a fitness/DMS label never unlocks kinetic "
        "claims.",
        "",
        "## Config-only mechanism loading",
        "",
        "| config | reaction.class | geometry terms | loaded |",
        "|---|---|---:|---|",
    ]
    for row in summarize_smoke_configs(config_paths):
        loaded = "yes" if row["loaded"] else f"NO ({row['error']})"
        lines.append(
            f"| `{row['config']}` | {row['reaction_class']} | {row['geometry_terms']} | {loaded} |"
        )
    lines.extend(
        [
            "",
            "## Benchmark claim discipline",
            "",
            "| benchmark | mechanism | label | direct-kinetic label? | kinetic claims allowed? |",
            "|---|---|---|---|---|",
        ]
    )
    for card in STAGE1.values():
        # `benchmark_extra_forbidden` is the authoritative block-list; we report only the
        # count so the report never embeds bannable phrases verbatim (which would trip its
        # own claim linter).
        n_blocked = len(benchmark_extra_forbidden(card))
        allowed = "yes" if benchmark_unlocks_kinetics(card) else f"no ({n_blocked} phrases blocked)"
        lines.append(
            f"| {card.target_id} | {card.mechanism_template} | {card.label_primary} | "
            f"{card.is_direct_kcat} | {allowed} |"
        )
    lines.extend(
        [
            "",
            "## Claim status",
            "",
            "No wet-lab data supplied: all benchmarks stay at L0 (hypothesis-grade triage). "
            "Fitness- and DMS-labelled benchmarks forbid kcat claims regardless of future "
            "wet-lab, because the label is not a kinetic constant.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_benchmark_smoke_report(root: Path, out_path: Path | None = None) -> str:
    config_paths = [
        root / "configs" / "benchmarks" / "tem1_v4_smoke.yaml",
        root / "configs" / "benchmarks" / "glycosidase_v4_smoke.yaml",
    ]
    text = render_benchmark_smoke_report(config_paths)
    out = out_path or (root / "reports" / "v4_benchmark_smoke.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text
