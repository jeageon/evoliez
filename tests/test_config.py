import re
from pathlib import Path

import pytest

from evoliez.config import Backend, Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def _seq_from_fasta(p: Path) -> str:
    return "".join(l.strip() for l in p.read_text().splitlines()
                   if l and not l.startswith(">")).upper()


@pytest.mark.parametrize(
    "cfg_path",
    sorted((ROOT / "configs").glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_residue_tokens_consistent_with_bundled_fasta(cfg_path):
    """Catalytic/fixed/binding-site residue tokens must match the WT letter at
    that position in the config's bundled target fasta. server_fdh_nadp.yaml had
    stale H155/G10/G12 (the fasta has V/D/P there) while its siblings were
    corrected to H154/G15/G33; validate_residue_tokens only warns, so nothing
    caught the drift."""
    cfg = load_config(cfg_path)
    ic = cfg.input
    fasta = getattr(ic, "target_fasta", None)
    if not (fasta and (ROOT / fasta).exists()):
        pytest.skip("no bundled fasta")
    seq = _seq_from_fasta(ROOT / fasta)
    bad = []
    for kind in ("catalytic_residues", "fixed_residues", "known_binding_site"):
        for t in (getattr(ic, kind, None) or []):
            m = re.match(r"^([A-Z])(\d+)$", str(t))
            if m and int(m.group(2)) <= len(seq):
                wt, pos = m.group(1), int(m.group(2))
                if seq[pos - 1] != wt:
                    bad.append(f"{t}: fasta[{pos}]={seq[pos - 1]}")
    assert not bad, f"{cfg_path.name}: {bad}"


def test_example_config_loads():
    cfg = load_config(ROOT / "configs" / "example_fdh_nadp.yaml")
    assert cfg.project.objective == "cofactor_switching"
    assert cfg.backend is Backend.mock
    assert "ligandmpnn" in cfg.mutation_generation.methods
    assert cfg.input.ligand.type == "smiles"


def test_default_config_loads():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.validation.md.protocol_level == 1
    assert cfg.backend_for("s04_complex") is Backend.mock


def test_overrides_and_per_stage_backend():
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {"backend": "real", "backends": {"s10_md": "mock"}},
    )
    assert cfg.backend is Backend.real
    assert cfg.backend_for("s10_md") is Backend.mock
    assert cfg.backend_for("s04_complex") is Backend.real


def test_unknown_key_rejected():
    with pytest.raises(Exception):
        Config.model_validate({"input": {"ligand": {"value": "C"}}, "bogus": 1})


def test_input_requires_sequence_source():
    with pytest.raises(Exception):
        Config.model_validate({"input": {"ligand": {"value": "C"}}})


def test_no_dead_data_scale_knob():
    """The data_scale config block was inert (no code read cfg.data_scale.*)
    yet was serialized into provenance, advertising a scale the run never
    honored. It must not exist on the schema."""
    cfg = load_config(ROOT / "configs" / "server_fdh_nadp.yaml")
    assert not hasattr(cfg, "data_scale")
    assert "data_scale" not in cfg.model_dump(mode="json")


@pytest.mark.parametrize(
    "cfg_path",
    sorted((ROOT / "configs").glob("*.yaml")),
    ids=lambda p: p.name,
)
def test_all_shipped_configs_validate(cfg_path):
    """Every configs/*.yaml must pass Config schema validation (extra keys
    are forbidden). Regression guard: pytest never loaded smoke.yaml, so a
    stray top-level key (e.g. a misplaced `docking:`) only blew up on the
    server. This makes any shipped config a CI citizen."""
    load_config(cfg_path)
