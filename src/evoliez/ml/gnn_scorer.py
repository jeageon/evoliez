"""Load a trained EvoLigand-GNN and score a (mutant) complex.

Optional: if torch is missing or no checkpoint exists, ``load`` returns None
and callers fall back to the heuristic family interaction model. The score is
a family-consistency probability in [0, 1] = the mean TRAINED contact-head
probability (the graph_score head is untrained, so it is not blended in).
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
            # weights_only=True: the blob is a state_dict + plain ints/bool/str
            # and a small geom dict (see train_gnn.save) - all torch>=2.0
            # safe-globals; blocks code execution from a tampered checkpoint.
            blob = torch.load(ckpt, map_location="cpu", weights_only=True)
            model = egnn.EvoLigandGNN(
                blob["node_dim"], blob["edge_dim"],
                hidden=blob["hidden"], layers=blob["layers"],
                rbf_n=blob.get("rbf_n", 16),
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
        disorder=None,
    ) -> float:
        import torch

        from evoliez.features.confidence import residue_confidence

        # Match TRAINING's graph construction: the same disorder track (else the
        # last 2 node features are 0 at inference but real at train time) and the
        # geometry the checkpoint was trained with (else inference topology
        # diverges when a user edits gnn radii/cutoffs and retrains).
        geom = self.meta.get("graph_geom", {})
        s = build_graph_sample(
            cx, position_features, ensemble_contacts,
            catalytic_positions=catalytic_positions,
            residue_confidence=residue_confidence(cx.structure),
            disorder=disorder,
            ligand_iptm=float(cx.metrics.get("ligand_iptm", 1.0)),
            complex_ipde=float(cx.metrics.get("complex_ipde", 2.0)),
            radius_lr=geom.get("radius_lr", 6.0),
            radius_rr=geom.get("radius_rr", 8.0),
            low_plddt_cutoff=geom.get("low_plddt_cutoff", 50.0),
            drop_far_low_plddt=geom.get("drop_far_low_plddt", True),
        )
        if s is None:
            return 0.5
        with torch.no_grad():
            out = self._model(to_torch(s))
            # ONLY the trained contact head contributes. The graph_score head
            # receives no gradient in egnn.multitask_loss (there is no
            # graph-level label), so it stays at random init; blending it in
            # injected init-noise into ~50% of gnn_score with no warning.
            contact = torch.sigmoid(out["contact_logit"]).mean().item()
        return round(float(contact), 4)
