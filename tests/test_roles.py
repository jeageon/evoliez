"""Unit tests for the canonical ligand-role vocabulary (v2 Phase A foundation)."""
from evoliez.roles import (
    CANONICAL_ROLES, COFACTOR, CONTEXT, DESIGN_LIGAND, METAL, ROLE_OBJECTIVES,
    SUBSTRATE, TS_PROXY, normalize_role, objective_for, resolve_role,
)


def test_normalize_canonical_and_aliases():
    assert normalize_role("cofactor") == COFACTOR
    assert normalize_role("ion") == METAL               # alias
    assert normalize_role("TS") == TS_PROXY             # alias
    assert normalize_role("Design Ligand") == DESIGN_LIGAND  # spaces/case
    assert normalize_role("transition-state") == TS_PROXY    # hyphen
    assert normalize_role(None) is None
    assert normalize_role("nonsense") is None
    # a ligand NAME is not a role (role comes from the role field, not the identity)
    assert normalize_role("NADP") is None


def test_resolve_positional_default():
    assert resolve_role(None, is_primary=True) == DESIGN_LIGAND
    assert resolve_role(None, is_primary=False) == COFACTOR
    assert resolve_role("substrate", is_primary=False) == SUBSTRATE   # explicit wins
    assert resolve_role("metal", is_primary=True) == METAL            # explicit wins over primary


def test_objective_table_complete_and_role_specific():
    for r in CANONICAL_ROLES:
        assert r in ROLE_OBJECTIVES
    assert objective_for(None, is_primary=True).optimize_affinity      # design ligand
    assert objective_for("substrate").optimize_catalytic_geometry
    assert objective_for("transition_state_proxy").optimize_catalytic_geometry
    assert objective_for("cofactor").optimize_orientation_retention
    assert objective_for("metal").optimize_coordination
    assert objective_for("context").preserve_contacts
    # roles are mutually distinct objectives, not all-the-same
    assert not objective_for("context").optimize_affinity


def test_compute_config_defaults():
    from evoliez.config import ComputeConfig
    c = ComputeConfig()
    assert c.cpu_core_budget == 16 and c.gpu_pool == [] and c.fail_loud_on_cpu_md is True
