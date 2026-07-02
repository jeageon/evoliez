"""V4-10: ClaimGuard keeps non-FDH benchmarks honest (ROADMAP_V4 §7.11)."""

from pathlib import Path

from evoliez.mechanism.benchmarks import (
    TEM1,
    benchmark_extra_forbidden,
    benchmark_unlocks_kinetics,
)
from evoliez.mechanism.vocab import UNCALIBRATED
from evoliez.ranking.claim_guard import ClaimProvenance, evaluate, lint_text
from evoliez.reports.v4_benchmark_smoke import render_benchmark_smoke_report

ROOT = Path(__file__).resolve().parents[2]


def test_fitness_benchmark_never_unlocks_kinetics():
    assert benchmark_unlocks_kinetics(TEM1) is False
    extra = benchmark_extra_forbidden(TEM1)
    assert any("kcat" in phrase for phrase in extra)


def test_kcat_claim_on_fitness_benchmark_is_caught():
    extra = benchmark_extra_forbidden(TEM1)
    text = "This variant predicts kcat for the beta-lactamase substrate."
    assert lint_text(text, extra_forbidden=extra)


def test_weak_provenance_stays_uncalibrated():
    verdict = evaluate(ClaimProvenance())  # no controls, no wet-lab
    assert verdict.claim_ceiling == UNCALIBRATED
    assert not verdict.allowed_categories


def test_smoke_report_loads_configs_and_is_clean():
    text = render_benchmark_smoke_report([
        ROOT / "configs" / "benchmarks" / "tem1_v4_smoke.yaml",
        ROOT / "configs" / "benchmarks" / "glycosidase_v4_smoke.yaml",
    ])
    # both non-FDH mechanisms load config-only
    assert "nucleophilic_acyl_substitution" in text
    assert "glycosidic_bond_cleavage" in text
    assert "| yes |" in text  # at least one config loaded
    # the report itself must not over-claim
    assert lint_text(text) == []
