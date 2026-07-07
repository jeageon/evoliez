"""V3-8: BenchmarkCard registry + claim integration (ROADMAP_V3 D9)."""
from evoliez.mechanism import benchmarks as bm
from evoliez.mechanism.spec import list_template_keys
from evoliez.ranking import claim_guard as cg


def test_stage1_registry():
    assert set(bm.STAGE1) == {"fdh_nadp", "TEM1_beta_lactamase", "glycosidase"}
    # every benchmark names a registered mechanism template
    keys = set(list_template_keys())
    for card in bm.ALL_BENCHMARKS.values():
        assert card.mechanism_template in keys


def test_fitness_benchmark_never_unlocks_kinetics():
    assert bm.benchmark_unlocks_kinetics(bm.TEM1) is False    # antibiotic fitness != kcat
    assert bm.benchmark_unlocks_kinetics(bm.FDH) is True      # direct A340 endpoint


def test_benchmark_forbidden_phrases_enforced():
    forbidden = bm.benchmark_extra_forbidden(bm.TEM1)
    assert "direct kcat prediction" in forbidden
    # even a sentence with no generic over-claim token is caught by the benchmark phrase
    v = cg.lint_text("The model provides direct kcat prediction for TEM-1.",
                     extra_forbidden=forbidden)
    assert any(x.category == cg.BENCHMARK_CATEGORY for x in v)
    # negation still safe
    assert cg.lint_text("This benchmark does not provide direct kcat prediction.",
                        extra_forbidden=forbidden) == []


def test_p450_is_stress_test_only():
    assert "validated catalytic state" in bm.P450_BM3.prohibited_claims
    assert bm.benchmark_unlocks_kinetics(bm.P450_BM3) is False
