"""Load a trained EvoLigand-GNN and score a (mutant) complex.

Optional: if torch is missing or no checkpoint exists, ``load`` returns None
and callers fall back to the heuristic family interaction model. The score is
a family-consistency probability in [0, 1] (graph head + mean contact prob).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from evoliez.features.evolutionary import PositionFeature
from evoliez.logging_utils import get_logger
from evoliez.ml import egnn
from evoliez.ml.graph_dataset import build_graph_sample, to_torch
from evoliez.types import Complex

log = get_logger("evoliez.gnn_scorer")


class EvoLigandGNNScorer:
    def __init__(self, model, meta: dict):
        self._model = model
        self.meta = meta

    @classmethod
    def load(cls, ckpt: Path) -> Optional["EvoLigandGNNScorer"]:
        if not egnn.is_available() or not Path(ckpt).exists():
            return None
        import torch

        try:
            blob = torch.load(ckpt, map_location="cpu")
            model = egnn.EvoLigandGNN(
                blob["node_dim"], blob["edge_dim"],
                hidden=blob["hidden"], layers=blob["layers"],
                equivariant=blob.get("equivariant", False),
            )
            model.load_state_dict(blob["state_dict"])
            model.eval()
            return cls(model, blob)
        except Exception as exc:  # never break ranking on a bad ckpt
            log.warning("could not load GNN checkpoint (%s); fallback", exc)
            return None

    def score_complex(
        self,
        cx: Complex,
        position_features: Sequence[PositionFeature],
        ensemble_contacts,
        *,
        catalytic_positions: Sequence[int] = (),
    ) -> float:
        import torch

        from evoliez.features.confidence import residue_confidence

        s = build_graph_sample(
            cx, position_features, ensemble_contacts,
            catalytic_positions=catalytic_positions,
            residue_confidence=residue_confidence(cx.structure),
            ligand_iptm=float(cx.metrics.get("ligand_iptm", 1.0)),
            complex_ipde=float(cx.metrics.get("complex_ipde", 2.0)),
        )
        if s is None:
            return 0.5
        with torch.no_grad():
            out = self._model(to_torch(s))
            graph = torch.sigmoid(out["graph_score"]).mean().item()
            contact = torch.sigmoid(out["contact_logit"]).mean().item()
        return round(float(0.5 * graph + 0.5 * contact), 4)
