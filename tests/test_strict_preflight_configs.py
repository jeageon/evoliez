"""L (post-expert-audit) — strict_preflight enabled in NADP-cofactor configs.

H added the gate and K added the pipeline-halt; neither helps until a
config actually turns the flag on. The expert flagged: "현재 cheap
config들에는 ... strict_preflight는 명시되어 있지 않습니다."

This file pins which configs enable strict_preflight:

  - Cofactor-class enzymes (NADP+/NADPH ligand) → strict_preflight:
    true. Without curated AMBER files the preflight is the ONLY
    honest signal; running Boltz for hours when MD will skip is the
    expert's wasted-time scenario.

  - Drug-like substrate enzymes (TEM-1 β-lactam, Bgl3 glucose,
    P450 BM3 lauric acid) → strict_preflight stays at the default
    (False). Their preflight returns "ok" so strict mode would never
    trigger anyway; leaving it off keeps cheap-run wall-time savings
    available even on transient preflight glitches (test mocks etc).

If you add a new cofactor-class enzyme config, add it to the
COFACTOR_STRICT_CONFIGS list below so this test fails until
strict_preflight is turned on.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# Configs that MUST enable strict_preflight because MD is the primary
# scientific signal and the ligand is a cofactor without curated AMBER
# files (preflight will return "unsupported" -> wasted Boltz time
# without strict mode).
COFACTOR_STRICT_CONFIGS = [
    "server_xr_cheap.yaml",
    "server_fdh_nadp_cheap.yaml",
]

# Configs that LEAVE strict_preflight at its default (False) because
# their ligand is drug-like (Sage / GAFF / espaloma can parameterise).
DRUG_LIKE_NON_STRICT_CONFIGS = [
    "server_tem1_cheap.yaml",
    "server_bgl3_cheap.yaml",
    "server_p450_bm3_cheap.yaml",
]


def _load_md_section(name: str) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / name).read_text())
    return (cfg.get("validation") or {}).get("md") or {}


@pytest.mark.parametrize("name", COFACTOR_STRICT_CONFIGS)
def test_cofactor_config_has_strict_preflight_true(name):
    md = _load_md_section(name)
    assert md.get("strict_preflight") is True, (
        f"{name}: NADP-class cofactor — must enable strict_preflight "
        "so the pipeline halts cleanly at s01 when curated AMBER files "
        "aren't on the server (instead of burning Boltz hours)"
    )


@pytest.mark.parametrize("name", DRUG_LIKE_NON_STRICT_CONFIGS)
def test_drug_like_config_strict_preflight_default(name):
    """Drug-like configs SHOULD leave strict_preflight at its default
    (absent or False). If you opt-in deliberately, change the test."""
    md = _load_md_section(name)
    # Either explicitly False, or absent (= the MDConfig default).
    val = md.get("strict_preflight", False)
    assert val is False, (
        f"{name} sets strict_preflight={val}. If this is deliberate "
        "(e.g. lockdown mode for the drug-like benchmark), move it to "
        "COFACTOR_STRICT_CONFIGS instead."
    )


def test_default_md_config_strict_preflight_is_off():
    """Pydantic default: when a config doesn't mention strict_preflight,
    we get the safe default (off). Mirrors test_strict_md_preflight.py
    but specifically for the L-config-coverage angle."""
    from evoliez.config import MDConfig
    assert MDConfig().strict_preflight is False
