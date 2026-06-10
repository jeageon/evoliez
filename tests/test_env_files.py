"""Guard the conda env files against dropping deps that the real pipeline
needs. Pure file parsing - no tool/GPU import, runs on the light venv.

Why this exists: pdbfixer is required by the real-MD terminal/heavy-atom
repair (openmm_engine._protein_only_topology) and the `doctor` preflight, and
it lives in pyproject's [md] extra. But the env files pip-install only `-e .`
(base deps), so a server env built from environment-gpu.yml silently lacks
pdbfixer and every real-MD candidate hits the uncapped-terminus failure the
repair step was written to prevent.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _conda_deps(env_path: Path):
    """(conda_package_names, pip_requirement_strings) from a conda env yaml."""
    spec = yaml.safe_load(env_path.read_text())
    conda, pip = [], []
    for dep in spec.get("dependencies", []):
        if isinstance(dep, str):
            conda.append(dep)
        elif isinstance(dep, dict) and "pip" in dep:
            pip.extend(dep["pip"])
    return conda, pip


def _provides_pdbfixer(env_path: Path) -> bool:
    conda, pip = _conda_deps(env_path)
    # satisfied by a direct conda package OR a pip install of the [md] extra
    if any(c.split("=")[0].strip() == "pdbfixer" for c in conda):
        return True
    return any(".[md]" in p or ".[all]" in p for p in pip)


def test_gpu_env_provides_pdbfixer():
    """The SERVER env (where real MD runs) must provide pdbfixer."""
    env = ROOT / "environment-gpu.yml"
    assert env.exists(), "environment-gpu.yml missing"
    assert _provides_pdbfixer(env), (
        "environment-gpu.yml does not provide pdbfixer; real MD will hit "
        "'No template for residue' on Boltz chain termini"
    )


def test_local_env_provides_pdbfixer():
    """The local env runs MD-lite smoke tests, which also repair termini."""
    env = ROOT / "environment.yml"
    assert env.exists(), "environment.yml missing"
    assert _provides_pdbfixer(env), (
        "environment.yml does not provide pdbfixer for local MD-lite smoke"
    )
