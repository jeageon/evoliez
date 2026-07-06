"""ROADMAP_V5 generality (commercial-platform criterion) — the SAME framework runs three distinct
mechanism classes from CONFIG ONLY: FDH hydride_transfer, a non-redox nucleophilic_acyl_substitution
(metalloenzyme), and CAR adenylation_phosphoryl_transfer. Each declares only a `mechanism:` block and
resolves to mechanism_spec mode with its template — no per-enzyme core code. (test_v5_generality proves
the spec-level contract; this proves the shipped RUN configs actually resolve it, and the mock e2e +
ClaimGuard-clean reports are demonstrated in docs/car_v5/generality_demonstration.md.)"""
import pytest

from evoliez.config import load_config
from evoliez.mechanism.mode import mechanism_mode

CASES = [
    ("configs/gen_fdh_hydride.yaml", "hydride_transfer"),
    ("configs/gen_metallo_acyl.yaml", "nucleophilic_acyl_substitution"),
    ("configs/car_srcar_3hp_v5_smoke.yaml", "adenylation_phosphoryl_transfer"),
]


@pytest.mark.parametrize("path,expected_class", CASES)
def test_run_config_resolves_mechanism_spec(path, expected_class):
    cfg = load_config(path)
    assert mechanism_mode(cfg) == "mechanism_spec", f"{path} did not resolve mechanism_spec"
    ri = cfg.mechanism.reaction
    got = ri.model_dump().get("cls")
    assert got == expected_class, f"{path} class={got}, expected {expected_class}"


def test_three_distinct_mechanism_classes_one_framework():
    """The point of the criterion: the three configs cover THREE distinct reaction classes, all
    under the same mechanism_spec resolution — redox (hydride), non-redox (acyl), and metal-bridged
    phosphoryl transfer (adenylation)."""
    classes = set()
    for path, _ in CASES:
        ri = load_config(path).mechanism.reaction
        classes.add(ri.model_dump().get("cls"))
    assert classes == {"hydride_transfer", "nucleophilic_acyl_substitution",
                       "adenylation_phosphoryl_transfer"}
    assert len(classes) == 3
