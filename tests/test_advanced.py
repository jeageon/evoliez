from pathlib import Path

import pytest

from evoliez.adapters.boltz import predict_complex
from evoliez.adapters.plip import fingerprint, type_counts
from evoliez.config import Backend, ComplexPredictionConfig, LigandInput
from evoliez.features.evolutionary import compute_position_features
from evoliez.features.ligand import parse_ligand
from evoliez.features.ligand_importance import ligand_atom_importance
from evoliez.features.mechanism import annotate, catalytic_geometry_deviation
from evoliez.features.subfamily import annotate_subfamilies
from evoliez.io.provenance import build_provenance
from evoliez.ml.active_learning import select_focused_library
from evoliez.ml.benchmark import run_benchmark
from evoliez.ml.calibration import expected_calibration_error, recommendation
from evoliez.ml.labels import LabelPolicyError, assert_supervised_label_allowed
from evoliez.ranking.negative_design import negative_penalties
from evoliez.types import Candidate, Mutation

ROOT = Path(__file__).resolve().parents[1]


def _cx(tmp_path):
    lig = parse_ligand(LigandInput(id="L", type="smiles",
                                   value="NC(=O)c1ccnc1OP([O-])=O"))
    return predict_complex("t", "ACDEFGHIKLMNPQRSTVWY" * 3, lig,
                           ComplexPredictionConfig(diffusion_samples=4),
                           tmp_path, backend=Backend.mock)


def test_mechanism_annotation(tmp_path):
    cx = _cx(tmp_path)
    m = annotate(cx, catalytic_positions=[15, 28], cofactor="NADP")
    assert m.roles.get(15, {}).get("catalytic")
    assert m.reactive_ligand_atoms
    assert 0.0 <= m.ts_geometry_score <= 1.0
    assert catalytic_geometry_deviation(m, m) == 0.0


def test_ligand_importance_orders_reactive_highest(tmp_path):
    cx = _cx(tmp_path)
    m = annotate(cx, catalytic_positions=[15], cofactor="NADP")
    imp = ligand_atom_importance(cx.ligand, m)
    assert imp
    assert max(imp.values()) <= 1.0 and min(imp.values()) >= 0.0
    for aid in m.reactive_ligand_atoms:
        assert imp[aid] >= 0.9


def test_plip_fingerprint_types(tmp_path):
    cx = _cx(tmp_path)
    fp = fingerprint(cx.structure, cx.ligand.atoms)
    tc = type_counts(fp)
    assert set(tc).issuperset({"hbond", "salt_bridge", "pi_stack"})
    assert sum(tc.values()) == len(fp)


def test_negative_design_penalizes_catalytic(tmp_path):
    cx = _cx(tmp_path)
    m = annotate(cx, catalytic_positions=[15], cofactor="NADP")
    cand = Candidate("c0", [Mutation("H", 15, "A")], "test")
    p = negative_penalties(
        cand, wt_mech=m, mut_mech=m, position_features=[],
        catalytic_positions=[15], buried_fraction=0.3,
        docking_score=-8.0, redocking_consistency=0.9,
    )
    assert p["neg_catalytic_mut"] >= 1.0


def test_subfamily_specificity_divergence():
    # subfamily A fixes K at col1, subfamily B fixes E -> divergent
    msa = [
        ("target", "AK"), ("a1", "AK"), ("a2", "AK"),
        ("b1", "AE"), ("b2", "AE"), ("b3", "AE"),
    ]
    feats = compute_position_features(msa)
    annotate_subfamilies(
        msa, feats,
        {"target": 0, "a1": 0, "a2": 0, "b1": 1, "b2": 1, "b3": 1},
    )
    by = {f.target_position: f for f in feats}
    assert by[2].specificity_divergence >= by[1].specificity_divergence


def test_benchmark_metrics():
    ranked = [
        Candidate("c1", [Mutation("A", 85, "K")], "g"),
        Candidate("c2", [Mutation("H", 155, "A")], "g"),
    ]
    ranked[0].scores["final_score"] = 5.0
    ranked[1].scores["final_score"] = -2.0
    bench = [
        {"mutation": "A85K", "label": "beneficial", "activity": 2.0},
        {"mutation": "H155A", "label": "deleterious", "activity": 0.05},
    ]
    r = run_benchmark(ranked, bench, k=1, catalytic_positions=[155],
                      known_site=[85])
    assert r["beneficial_recall_at_k"] == 1.0
    assert r["auroc_beneficial"] == 1.0
    # catalytic single-mutant H155A should be protected (ranked bottom half)
    assert r["catalytic_protection_rate"] == 1.0
    assert r["binding_site_enrichment"] == 1.0  # top-1 targets site res 85
    assert 0.0 <= r["calibration"]["ece"] <= 1.0


def test_run_ablation_reports_layer_deltas(tmp_path):
    from evoliez.ml.benchmark import run_ablation

    cfg = tmp_path / "small.yaml"
    cfg.write_text(
        "project: {name: abl, output_dir: %s/run}\n"
        "input:\n"
        "  target_id: fdh\n"
        "  target_fasta: %s\n"
        "  ligand: {id: L, type: smiles, value: 'CC(=O)O'}\n"
        "mutation_generation: {methods: [chemistry_rules], "
        "max_candidates: 25, design_radius_angstrom: 9.0}\n"
        "reranking: {top_for_redocking: 8, top_for_md: 3}\n"
        "validation: {md: {top_candidates: 3}}\n"
        "gnn: {build_dataset: false}\n"
        "backend: mock\n"
        % (tmp_path, ROOT / "examples" / "fdh" / "target.fasta")
    )
    bench = [{"mutation": "A85K", "label": "beneficial", "activity": 2.0},
             {"mutation": "F84L", "label": "neutral", "activity": 1.0}]
    res = run_ablation(str(cfg), bench, k=5, toggles=("md",),
                       base_output=str(tmp_path / "abl"))
    tags = [r["ablation"] for r in res["ablation"]]
    assert tags == ["full_model", "no_md"]
    assert "delta_auroc_beneficial" in res["ablation"][1]


def test_calibration_and_ece():
    # The top confident + positive-geometry label is claim-safe (V5-5): no bare "strong candidate".
    assert recommendation(5.0, 0.1, 1, 100, geometry=0.3) == "priority screening candidate"
    assert recommendation(0.0, 0.9, 50, 100) == "reject"
    ece = expected_calibration_error([0.9, 0.1, 0.8], [1, 0, 1])
    assert 0.0 <= ece <= 1.0


def test_active_learning_diverse_library():
    cands = []
    for i in range(20):
        c = Candidate(f"c{i}", [Mutation("A", 10 + i, "KDLS"[i % 4])], "g")
        c.scores["final_score"] = 5.0 - i * 0.1
        cands.append(c)
    lib = select_focused_library(cands, 8)
    assert len(lib) == 8
    assert all("acquisition_score" in c.scores for c in lib)


def test_provenance_is_deterministic():
    a = build_provenance(sequence="ACDE", ligand_smiles="CCO",
                          config_dict={"x": 1}, seed=1, backend="mock")
    b = build_provenance(sequence="ACDE", ligand_smiles="CCO",
                         config_dict={"x": 1}, seed=1, backend="mock")
    assert a["input_sequence_sha1"] == b["input_sequence_sha1"]
    assert a["config_sha1"] == b["config_sha1"]


def test_advanced_features_never_labels():
    for bad in ("neg_catalytic_mut", "mechanism_ts", "ifp_hbond",
                "specificity_divergence",
                "acquisition_score", "uncertainty"):
        with pytest.raises(LabelPolicyError):
            assert_supervised_label_allowed(bad)


def test_dead_subfamily_conservation_feature_removed():
    """subfamily_conservation was always 1.0 (target is a singleton cluster) and
    never read by any ranking/ML code. The silently-constant feature is removed;
    the consumed specificity_divergence stays."""
    from dataclasses import fields

    from evoliez.features.evolutionary import PositionFeature
    names = {f.name for f in fields(PositionFeature)}
    assert "subfamily_conservation" not in names
    assert "specificity_divergence" in names
