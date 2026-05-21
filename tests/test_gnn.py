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
    # Same "safe mount" criteria as doctor (/_check_config): either of the
    # bulk-storage roots is fine. After the PseFDH real-target switch the
    # server config writes to /mnt/data/<user>/runs/ (writable in our
    # actual deployment; /mnt/data2 root isn't on this host).
    assert sc.project.output_dir.startswith(("/mnt/data2", "/mnt/data"))
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
