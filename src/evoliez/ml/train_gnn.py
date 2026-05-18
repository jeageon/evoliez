"""Self-supervised EvoLigand-GNN training (server-grade).

Single-GPU by default; DDP-aware when launched via ``torchrun`` (reads
LOCAL_RANK / WORLD_SIZE). Mixed precision optional. Trains on the .npz graph
dataset with the multi-task weak-label objective (no experimental labels).
torch is server-only; on the laptop this raises a clear message and the
pipeline keeps using the heuristic family model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from evoliez.logging_utils import get_logger
from evoliez.ml import egnn
from evoliez.ml.graph_dataset import (
    EDGE_DIM,
    NODE_DIM,
    load_graph_dataset,
    to_torch,
)
from evoliez.utils.gpu import apply_gpu_selection

log = get_logger("evoliez.train_gnn")


def train(
    dataset_dir: Path,
    out_ckpt: Path,
    *,
    epochs: int = 20,
    hidden: int = 128,
    layers: int = 4,
    lr: float = 1e-3,
    amp: bool = True,
    seed: int = 1234,
) -> Path:
    if not egnn.is_available():
        raise RuntimeError(
            "torch not installed - EvoLigand-GNN training is server-only. "
            "Install the 'gnn' extra on the GPU server (see SERVER_RUNBOOK)."
        )
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp = rank >= 0
    if not ddp:
        apply_gpu_selection()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_graph_dataset(dataset_dir)
    if not samples:
        raise RuntimeError(f"no graph samples in {dataset_dir}")
    log.info("loaded %d graph samples (device=%s, ddp=%s)",
             len(samples), device, ddp)

    model = egnn.EvoLigandGNN(NODE_DIM, EDGE_DIM, hidden=hidden,
                              layers=layers).to(device)
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank]
        )
    core = model.module if ddp else model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    loader = DataLoader(range(len(samples)), batch_size=1, shuffle=True,
                        collate_fn=lambda b: b[0])
    for ep in range(epochs):
        tot = 0.0
        for i in loader:
            b = {k: v.to(device) for k, v in to_torch(samples[i]).items()}
            if "contact_mask" in b and b["contact_mask"].numel():
                m = b["contact_mask"].bool()
                b["contact_label"] = b["contact_label"][m]
                b["contact_weight"] = b["contact_weight"][m]
                b["_lr_keep"] = m
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(b)
                if "_lr_keep" in b:
                    out["contact_logit"] = out["contact_logit"][b["_lr_keep"]]
                    out["itype_logit"] = out["itype_logit"][b["_lr_keep"]]
                    b["itype_label"] = b["itype_label"][b["_lr_keep"]]
                loss = egnn.multitask_loss(out, b)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
        log.info("epoch %d/%d  loss=%.4f", ep + 1, epochs, tot / len(samples))

    if (not ddp) or rank == 0:
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": core.state_dict(),
                "node_dim": NODE_DIM,
                "edge_dim": EDGE_DIM,
                "hidden": hidden,
                "layers": layers,
            },
            out_ckpt,
        )
        log.info("saved checkpoint: %s", out_ckpt)
    if ddp:
        torch.distributed.destroy_process_group()
    return out_ckpt
