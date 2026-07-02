import pytest

from evoliez.guided.feasibility import FeasibilityConfig


def test_feasibility_config_schema():
    cfg = FeasibilityConfig()
    assert cfg.max_gpu_memory_fraction == 0.75
    assert "mut_00479" in cfg.cases
    with pytest.raises(ValueError):
        FeasibilityConfig(max_gpu_memory_fraction=0.9)
