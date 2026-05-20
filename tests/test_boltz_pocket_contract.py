"""P0.2: pocket-steering contract + per-sample ensemble disagreement.

`complex_prediction.pocket_constraints: true` was previously a no-op
metadata flag. The plan demands a contract test that the flag actually
maps to a real Boltz `constraints` section pointing at the configured
catalytic / known-binding-site residues; if it ever silently regresses
to metadata, this test fails.

Per-sample ensemble disagreement (`affinity_ensemble_std`,
`confidence_ensemble_std`) is also exercised here so the reranker /
final report layer can detect internally-noisy predictions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evoliez.adapters.boltz import _finalize, _predict_real, predict_complex
from evoliez.config import Backend, ComplexPredictionConfig
from evoliez.types import BoltzSample, Complex, Ligand


def _cfg(pocket_constraints: bool, predict_affinity: bool = True) -> ComplexPredictionConfig:
    return ComplexPredictionConfig(
        primary_method="boltz2",
        use_msa_server=True,
        representative_homologs=4,
        use_templates=True,
        pocket_constraints=pocket_constraints,
        predict_affinity=predict_affinity,
        diffusion_samples=3,
        use_kernels=False,
    )


def _read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def test_pocket_constraints_flag_emits_real_yaml_block(tmp_path):
    # _predict_real(dry_run=True) writes the Boltz input YAML but never
    # invokes the binary. Stop after the YAML is written by inspecting
    # `<label>_boltz_input.yaml` in `outdir`.
    cfg = _cfg(pocket_constraints=True)
    lig = Ligand(id="L", smiles="CCO")
    _predict_real(
        "wt", "A" * 30, lig, cfg, tmp_path,
        dry_run=True, msa_path=None,
        pocket_residues=[15, 28, 154, 285],
    )
    yml = _read_yaml(tmp_path / "wt_boltz_input.yaml")
    # Real Boltz YAML schema: constraints[].pocket.{binder,contacts}.
    assert "constraints" in yml, (
        "pocket_constraints=true but no `constraints` block emitted - the "
        "flag is back to being metadata-only"
    )
    pocket = yml["constraints"][0]["pocket"]
    assert pocket["binder"] == "B"
    assert pocket["contacts"] == [
        ["A", 15], ["A", 28], ["A", 154], ["A", 285],
    ]


def test_pocket_flag_off_emits_no_constraints(tmp_path):
    cfg = _cfg(pocket_constraints=False)
    lig = Ligand(id="L", smiles="CCO")
    _predict_real(
        "wt", "A" * 30, lig, cfg, tmp_path,
        dry_run=True, msa_path=None,
        pocket_residues=[154, 285],
    )
    yml = _read_yaml(tmp_path / "wt_boltz_input.yaml")
    assert "constraints" not in yml


def test_pocket_flag_on_but_no_residues_emits_no_constraints(tmp_path):
    # Steering on but the upstream config didn't supply residues - we must
    # NOT fabricate a pocket from thin air, or Boltz would be steered at
    # arbitrary indices. Test guards against that silent failure.
    cfg = _cfg(pocket_constraints=True)
    lig = Ligand(id="L", smiles="CCO")
    _predict_real(
        "wt", "A" * 30, lig, cfg, tmp_path,
        dry_run=True, msa_path=None,
        pocket_residues=None,
    )
    yml = _read_yaml(tmp_path / "wt_boltz_input.yaml")
    assert "constraints" not in yml


def test_predict_complex_forwards_pocket_residues_to_real(tmp_path, monkeypatch):
    # The public entry point must thread pocket_residues through to the
    # real backend - else s04_complex's wiring (catalytic + known site)
    # would never reach the YAML.
    captured = {}

    def fake_real(label, sequence, ligand, cfg, outdir, *, dry_run, msa_path,
                  pocket_residues=None):
        captured["pocket_residues"] = pocket_residues
        return Complex(structure=None, ligand=ligand, method="boltz2")  # type: ignore

    monkeypatch.setattr("evoliez.adapters.boltz._predict_real", fake_real)
    predict_complex(
        "wt", "AAA", Ligand(id="L", smiles="CCO"), _cfg(True),
        tmp_path, backend=Backend.real, dry_run=True,
        pocket_residues=[154, 285],
    )
    assert captured["pocket_residues"] == [154, 285]


# --------------------------------------------------------------------------- #
# Per-sample ensemble disagreement
# --------------------------------------------------------------------------- #
def test_finalize_computes_affinity_and_confidence_std_per_sample():
    samples = [
        BoltzSample(idx=0, metrics={
            "affinity_pred_value": -7.0, "confidence_score": 0.80,
        }),
        BoltzSample(idx=1, metrics={
            "affinity_pred_value": -6.5, "confidence_score": 0.78,
        }),
        BoltzSample(idx=2, metrics={
            "affinity_pred_value": -7.5, "confidence_score": 0.82,
        }),
    ]
    cx = Complex(structure=None, ligand=None, samples=samples)  # type: ignore
    _finalize(cx)
    # std across 3 samples is nontrivial; sanity: must be > 0 and finite.
    assert cx.metrics["affinity_ensemble_std"] > 0
    assert cx.metrics["confidence_ensemble_std"] > 0
    assert cx.metrics["affinity_ensemble_mean"] == pytest.approx(-7.0, abs=0.1)


def test_finalize_handles_single_sample_no_std():
    samples = [BoltzSample(idx=0, metrics={"affinity_pred_value": -7.0})]
    cx = Complex(structure=None, ligand=None, samples=samples)  # type: ignore
    _finalize(cx)
    assert cx.metrics.get("affinity_ensemble_std", 0.0) == 0.0


# --------------------------------------------------------------------------- #
# Shipped default config carries the bumped WT sample count + steering on
# --------------------------------------------------------------------------- #
def test_default_config_diffusion_samples_at_least_25():
    from evoliez.config import load_config
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.complex_prediction.diffusion_samples >= 25, (
        "P0.2 plan: real-backend WT samples bumped 5 -> >=25"
    )
    assert cfg.complex_prediction.pocket_constraints is True
