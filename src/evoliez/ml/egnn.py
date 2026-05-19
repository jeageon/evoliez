"""EvoLigand-GNN: E(3)-invariant heterogeneous graph model (server-grade).

Ligand-atom + protein-residue nodes; ligand-residue and residue-residue edges
carrying the relative vector r_ij and an RBF distance encoding. Message passing
uses only pairwise distances, so node embeddings are E(3)-invariant (Satorras
et al. EGNN style, coordinates kept fixed). Multi-task heads:

  - edge contact probability        (weak label: ensemble contact frequency)
  - edge interaction-type (6-class) (geometry-derived weak label)
  - residue mutation permissiveness (MSA-derived, regression)
  - residue native-residue recovery (self-supervised, 20-class)
  - graph-level mutation score      (ranking head)

Pure PyTorch (no torch_geometric). torch is server-only: this module imports
without torch; ``is_available()`` is False on the laptop and callers fall back
to the heuristic family model.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAVE_TORCH = True
except Exception:  # torch is a server-only (gpu/gnn extra) dependency
    _HAVE_TORCH = False


def is_available() -> bool:
    return _HAVE_TORCH


N_ITYPES = 6  # hbond, salt_bridge, aromatic, hydrophobic, vdw, none
N_AA = 20


if _HAVE_TORCH:

    def rbf(d: "torch.Tensor", n: int = 16, dmax: float = 12.0) -> "torch.Tensor":
        centers = torch.linspace(0.0, dmax, n, device=d.device)
        width = (dmax / n)
        return torch.exp(-((d.unsqueeze(-1) - centers) ** 2) / (2 * width**2))

    class EGNNLayer(nn.Module):
        def __init__(self, hidden: int, edge_dim: int):
            super().__init__()
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden + 1 + edge_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )

        def forward(self, h, pos, edge_index, edge_attr, edge_w=None):
            src, dst = edge_index[0], edge_index[1]
            d2 = ((pos[src] - pos[dst]) ** 2).sum(-1, keepdim=True)
            m = self.edge_mlp(
                torch.cat([h[src], h[dst], d2, edge_attr], dim=-1)
            )
            if edge_w is not None:
                # confidence-aware: low-pLDDT / high-PDE edges contribute less
                m = m * edge_w.unsqueeze(-1)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            return h + self.node_mlp(torch.cat([h, agg], dim=-1)), pos

    class EquivariantEGNNLayer(nn.Module):
        """True E(3)-equivariant layer (Satorras et al. EGNN): scalar features
        stay invariant, coordinates are updated equivariantly via the
        normalised relative vector. Readout from h remains E(3)-invariant."""

        def __init__(self, hidden: int, edge_dim: int):
            super().__init__()
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * hidden + 1 + edge_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),
            )

        def forward(self, h, pos, edge_index, edge_attr, edge_w=None):
            src, dst = edge_index[0], edge_index[1]
            rel = pos[src] - pos[dst]
            d2 = (rel ** 2).sum(-1, keepdim=True)
            m = self.edge_mlp(
                torch.cat([h[src], h[dst], d2, edge_attr], dim=-1)
            )
            if edge_w is not None:
                m = m * edge_w.unsqueeze(-1)
            # equivariant coordinate update (vector channel)
            coeff = self.coord_mlp(m)
            rel_n = rel / (d2.sqrt() + 1.0)
            dx = torch.zeros_like(pos)
            dx.index_add_(0, dst, rel_n * coeff)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
            return h, pos + dx

    class EvoLigandGNN(nn.Module):
        def __init__(
            self,
            node_in: int,
            edge_in: int,
            hidden: int = 128,
            layers: int = 4,
            rbf_n: int = 16,
            equivariant: bool = False,
        ):
            super().__init__()
            self.rbf_n = rbf_n
            self.equivariant = equivariant
            self.type_emb = nn.Embedding(2, hidden)  # 0 ligand atom, 1 residue
            self.node_enc = nn.Linear(node_in, hidden)
            self.edge_enc = nn.Linear(edge_in + rbf_n, hidden)
            Layer = EquivariantEGNNLayer if equivariant else EGNNLayer
            self.blocks = nn.ModuleList(
                [Layer(hidden, hidden) for _ in range(layers)]
            )
            self.contact_head = nn.Linear(2 * hidden, 1)
            self.itype_head = nn.Linear(2 * hidden, N_ITYPES)
            self.perm_head = nn.Linear(hidden, 1)
            self.native_head = nn.Linear(hidden, N_AA)
            self.relia_head = nn.Linear(hidden, 1)   # Task F: coord reliability
            self.risk_head = nn.Linear(hidden, 1)    # Task G: flexible-pocket risk
            self.score_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )

        def forward(self, batch: dict) -> dict:
            h = self.node_enc(batch["node_feat"]) + self.type_emb(
                batch["node_type"]
            )
            pos = batch["pos"]
            ei = batch["edge_index"]
            d = ((pos[ei[0]] - pos[ei[1]]) ** 2).sum(-1).clamp_min(1e-8).sqrt()
            ea = self.edge_enc(
                torch.cat([batch["edge_attr"], rbf(d, self.rbf_n)], dim=-1)
            )
            ew = batch.get("edge_conf")
            for blk in self.blocks:
                h, pos = blk(h, pos, ei, ea, ew)

            lr = batch["lr_edge_index"]  # ligand-atom -> residue edges
            ee = torch.cat([h[lr[0]], h[lr[1]]], dim=-1)
            res_mask = batch["node_type"] == 1
            hr = h[res_mask]
            pooled = hr.mean(dim=0, keepdim=True)
            return {
                "contact_logit": self.contact_head(ee).squeeze(-1),
                "itype_logit": self.itype_head(ee),
                "perm": self.perm_head(hr).squeeze(-1),
                "native_logit": self.native_head(hr),
                "relia_logit": self.relia_head(hr).squeeze(-1),
                "risk": torch.sigmoid(self.risk_head(hr).squeeze(-1)),
                "graph_score": self.score_head(pooled).squeeze(-1),
            }

    def multitask_loss(out: dict, batch: dict) -> "torch.Tensor":
        loss = out["graph_score"].new_zeros(())
        if "contact_label" in batch and batch["contact_label"].numel():
            loss = loss + F.binary_cross_entropy_with_logits(
                out["contact_logit"], batch["contact_label"].float(),
                weight=batch.get("contact_weight"),
            )
        if "itype_label" in batch and batch["itype_label"].numel():
            loss = loss + F.cross_entropy(
                out["itype_logit"], batch["itype_label"].long()
            )
        if "perm_label" in batch and batch["perm_label"].numel():
            loss = loss + F.mse_loss(out["perm"], batch["perm_label"].float())
        if "native_label" in batch and batch["native_label"].numel():
            loss = loss + F.cross_entropy(
                out["native_logit"], batch["native_label"].long()
            )
        if "relia_label" in batch and batch["relia_label"].numel():
            loss = loss + F.binary_cross_entropy_with_logits(
                out["relia_logit"], batch["relia_label"].float()
            )
        if "risk_label" in batch and batch["risk_label"].numel():
            loss = loss + F.mse_loss(out["risk"], batch["risk_label"].float())
        return loss
