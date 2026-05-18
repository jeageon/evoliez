"""Deterministic seeding so mock runs and tests are reproducible."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:  # torch only present on the GPU server
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def derive_seed(base: int, *tokens: str) -> int:
    """Stable per-entity seed so a candidate's mock output never changes."""
    h = base
    for t in tokens:
        for ch in str(t):
            h = (h * 1099511628211 + ord(ch)) & 0xFFFFFFFF
    return h
