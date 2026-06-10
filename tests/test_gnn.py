from pathlib import Path

import numpy as np

from evoliez.adapters.boltz import predict_complex
from evoliez.config import (
    Backend,
    Config,
    ComplexPredictionConfig,
    LigandInput,
    ScoreWeights,
)
from evoliez.features.boltz_features import ensemble_contacts
from evoliez.features.evolutionary import compute_position_features
from evoliez.features.ligand import parse_ligand
from evoliez.ml import egnn
from evoliez.ml.gnn_scorer import EvoLigandGNNScorer
from evoliez.ml.graph_dataset import (
    EDGE_DIM,
    NODE_DIM,
    build_graph_sample,
    load_graph_dataset,
    save_graph_dataset,
)
from evoliez.utils.gpu import GpuPool, free_gpu_indices

ROOT = Path(__file__).resolve().parents[1]


def _ctx():
    lig = parse_ligand(LigandInput(id="L", type="smiles", value="CC(=O)OP(O)=O"))
    seq = "ACDEFGHIKLMNPQRSTVWY" * 3
    cx = predict_complex("t", seq, lig, ComplexPredictionConfig(diffusion_samples=6),
                         Path("/tmp/ez_gnn"), backend=Backend.mock)
    msa = [("t", seq)] + [("h%d" % i, seq) for i in range(5)]
    pf = compute_position_features(msa)
    econ = ensemble_contacts(cx.structure, cx.samples, cutoff=6.0,
                             ligand_iptm=cx.metrics["ligand_iptm"])
    return cx, pf, econ


def test_build_graph_sample_shapes():
    cx, pf, econ = _ctx()
    s = build_graph_sample(cx, pf, econ, catalytic_positions=[5, 10])
    assert s is not None
    assert s["node_feat"].shape[1] == NODE_DIM
    assert s["edge_attr"].shape[1] == EDGE_DIM
    assert s["edge_index"].shape[0] == 2
    assert s["lr_edge_index"].shape[0] == 2
    assert len(s["contact_label"]) == s["lr_edge_index"].shape[1]
    # node count = ligand atoms + nearby residues
    assert s["node_feat"].shape[0] == len(s["node_type"])
    assert set(np.unique(s["node_type"]).tolist()).issubset({0, 1})
    # confidence-aware additions
    for k in ("edge_conf", "plddt_norm", "relia_label", "risk_label"):
        assert k in s
    assert s["edge_conf"].shape[0] == s["edge_index"].shape[1]
    assert s["plddt_norm"].shape[0] == s["node_feat"].shape[0]
    assert (s["edge_conf"] >= 0.0).all()


def test_graph_dataset_roundtrip(tmp_path):
    cx, pf, econ = _ctx()
    s = build_graph_sample(cx, pf, econ)
    save_graph_dataset([s, s], tmp_path)
    back = load_graph_dataset(tmp_path)
    assert len(back) == 2
    assert np.allclose(back[0]["pos"], s["pos"])


def test_scorer_falls_back_without_checkpoint(tmp_path):
    assert EvoLigandGNNScorer.load(tmp_path / "missing.pt") is None


def test_gpu_pool_degrades_on_laptop():
    pool = GpuPool()
    assert len(pool) >= 1
    assert isinstance(pool.next(), int)
    assert isinstance(free_gpu_indices(), list)


def test_server_config_and_defaults():
    cfg = Config.model_validate(
        {"input": {"target_sequence": "MKTAYIAKQR" * 3,
                   "ligand": {"value": "CCO"}}}
    )
    assert cfg.gnn.enabled is False           # optional by default
    assert ScoreWeights().gnn == 0.0          # no effect unless trained
    srv = ROOT / "configs" / "server_fdh_nadp.yaml"
    from evoliez.config import load_config

    sc = load_config(srv)
    assert sc.backend is Backend.real
    assert sc.gnn.enabled is True
    assert sc.project.output_dir.startswith("/mnt/data2")
    assert sc.scoring.gnn > 0.0


def test_egnn_forward_if_torch_available():
    if not egnn.is_available():
        return  # torch is server-only; skip on the laptop
    import torch

    cx, pf, econ = _ctx()
    s = build_graph_sample(cx, pf, econ)
    from evoliez.ml.graph_dataset import to_torch

    model = egnn.EvoLigandGNN(NODE_DIM, EDGE_DIM, hidden=32, layers=2)
    out = model(to_torch(s))
    assert out["contact_logit"].shape[0] == s["lr_edge_index"].shape[1]
    loss = egnn.multitask_loss(out, to_torch(s))
    assert torch.isfinite(loss)


def test_inference_score_ignores_untrained_graph_head():
    """The graph_score head gets NO gradient in multitask_loss (no graph-level
    label), so it stays at random init. It must NOT influence the inference
    family-consistency score - otherwise ~50% of gnn_score is init-noise. Force
    the head to large +/- outputs and assert the score is unchanged."""
    if not egnn.is_available():
        return
    import torch

    cx, pf, econ = _ctx()
    model = egnn.EvoLigandGNN(NODE_DIM, EDGE_DIM, hidden=32, layers=2).eval()
    scorer = EvoLigandGNNScorer(model, {})
    base = scorer.score_complex(cx, pf, econ)
    with torch.no_grad():
        for p in model.score_head.parameters():
            p.copy_(p * 0 + 50.0)            # graph_score -> +inf, sigmoid ~1
        hi = scorer.score_complex(cx, pf, econ)
        for p in model.score_head.parameters():
            p.copy_(p * 0 - 50.0)            # graph_score -> -inf, sigmoid ~0
        lo = scorer.score_complex(cx, pf, econ)
    assert hi == lo == base, (
        "inference score depends on the untrained graph_score head "
        f"(base={base}, hi={hi}, lo={lo})"
    )


def test_disorder_changes_last_two_node_columns():
    """Training passes a DisorderTrack; inference must too. A graph built WITH
    disorder differs from one WITHOUT it in exactly the last 2 node-feature
    columns (dis_iup, dis_lc) - the train/inference skew this guards. Torch-free
    (build_graph_sample is numpy)."""
    from evoliez.adapters.disorder import predict_disorder

    cx, pf, econ = _ctx()
    dis = predict_disorder(cx.structure.sequence, Path("/tmp/ez_dis"),
                           backend=Backend.mock)
    s_no = build_graph_sample(cx, pf, econ)                 # disorder=None
    s_yes = build_graph_sample(cx, pf, econ, disorder=dis)
    a, b = s_no["node_feat"], s_yes["node_feat"]
    assert a.shape == b.shape
    assert not np.allclose(a[:, -2:], b[:, -2:])            # disorder cols differ
    assert np.allclose(a[:, :-2], b[:, :-2])                # other cols identical


def test_score_complex_accepts_disorder():
    """gnn_scorer.score_complex must accept (and thread) a disorder track so
    inference matches training - it previously did not, zeroing 2/21 features."""
    if not egnn.is_available():
        return
    from evoliez.adapters.disorder import predict_disorder

    cx, pf, econ = _ctx()
    dis = predict_disorder(cx.structure.sequence, Path("/tmp/ez_dis2"),
                           backend=Backend.mock)
    model = egnn.EvoLigandGNN(NODE_DIM, EDGE_DIM, hidden=32, layers=2).eval()
    scorer = EvoLigandGNNScorer(model, {})
    v = scorer.score_complex(cx, pf, econ, disorder=dis)
    assert 0.0 <= v <= 1.0


def test_train_persists_rbf_n_and_graph_geom(tmp_path):
    """gnn.rbf and the graph geometry must round-trip through the checkpoint so
    they are not silently ignored (rbf inert) / defaulted (geometry skew)."""
    if not egnn.is_available():
        return
    import torch
    from evoliez.ml.train_gnn import train

    cx, pf, econ = _ctx()
    s = build_graph_sample(cx, pf, econ)
    save_graph_dataset([s, s], tmp_path / "ds")
    ckpt = tmp_path / "m.pt"
    geom = {"radius_lr": 3.0, "radius_rr": 3.0, "low_plddt_cutoff": 40.0,
            "drop_far_low_plddt": False, "use_disorder": True}
    train(tmp_path / "ds", ckpt, epochs=1, hidden=16, layers=2, amp=False,
          rbf_n=8, graph_geom=geom)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert blob["rbf_n"] == 8
    assert blob["graph_geom"]["radius_lr"] == 3.0
    scorer = EvoLigandGNNScorer.load(ckpt)        # weights_only load must work
    assert scorer is not None
    assert scorer.meta["rbf_n"] == 8
    assert scorer.meta["graph_geom"]["radius_lr"] == 3.0


def test_equivariant_readout_is_invariant_if_torch():
    """E(3)-equivariant message passing -> rotation/translation-invariant
    contact/score readout (expert review #6 / terminology)."""
    if not egnn.is_available():
        return
    import torch

    from evoliez.ml.graph_dataset import to_torch

    cx, pf, econ = _ctx()
    s = build_graph_sample(cx, pf, econ)
    model = egnn.EvoLigandGNN(NODE_DIM, EDGE_DIM, hidden=32, layers=3,
                              equivariant=True).eval()
    b = to_torch(s)
    with torch.no_grad():
        o1 = model(b)
        # random rotation + translation of all coordinates
        q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        b2 = dict(b)
        b2["pos"] = b["pos"].float() @ q.T + torch.tensor([5.0, -2.0, 1.0])
        o2 = model(b2)
    assert torch.allclose(o1["contact_logit"], o2["contact_logit"], atol=1e-4)
    assert torch.allclose(o1["graph_score"], o2["graph_score"], atol=1e-4)
