"""ESM2 single-sequence prior (roadmap P1.1, ESM priors).

A protein language model scores how tolerant each position is to substitution
WITHOUT an MSA or any database: feed the target sequence to ESM2, read the
predicted amino-acid distribution per position, and take its (normalised)
entropy as a 0..1 "variability" — high = many residues are acceptable there,
low = constrained. This complements the MSA-derived conservation (which needs
homologs) and is fully local (GPU-optional).

real: fair-esm (`import esm`) or HuggingFace `transformers`, whichever is
present. mock: deterministic per-position pseudo-variability from the sequence,
so the feature has reproducible structure on the dev box / CI.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import List, Optional

from evoliez.config import Backend
from evoliez.logging_utils import get_logger
from evoliez.utils.seeds import derive_seed

log = get_logger("evoliez.esm")
_AA = "ACDEFGHIKLMNPQRSTVWY"
_LOG2_20 = math.log2(len(_AA))


def esm_position_priors(
    sequence: str,
    *,
    model: str,
    backend: Backend,
    dry_run: bool = False,
    workdir: Optional[Path] = None,
) -> List[float]:
    """Per-residue substitution variability in [0,1], length == len(sequence).

    The result is content-keyed on ``sha1(sequence + model)`` and cached as JSON
    under ``workdir`` (when given). A resume with byte-identical inputs then skips
    reloading the 650M model + the forward pass entirely — it just reads the JSON.
    Caching applies to the mock path too, so the behaviour is identical with or
    without torch.
    """
    seq = "".join(c for c in sequence.upper() if not c.isspace())

    cache = _cache_path(workdir, seq, model)
    if cache is not None:
        hit = _cache_read(cache, len(seq))
        if hit is not None:
            log.info("ESM prior cache hit (%s)", cache.name)
            return hit

    if backend is Backend.real and not dry_run:
        out = _esm_real(seq, model)
        if out is not None:
            _cache_write(cache, out)
            return out
    out = _esm_mock(seq)
    _cache_write(cache, out)
    return out


def _cache_path(workdir: Optional[Path], seq: str, model: str) -> Optional[Path]:
    if workdir is None:
        return None
    key = hashlib.sha1(f"{seq}\0{model}".encode()).hexdigest()
    return Path(workdir) / f"esm_prior.{key}.json"


def _cache_read(path: Path, expected_len: int) -> Optional[List[float]]:
    """Return the cached priors iff present, well-formed and the right length."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if (isinstance(data, list) and len(data) == expected_len
                and all(isinstance(x, (int, float)) for x in data)):
            return [float(x) for x in data]
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    return None  # corrupt / stale -> recompute


def _cache_write(path: Optional[Path], priors: List[float]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(priors))
    except OSError as exc:  # caching is best-effort; never fail the prior
        log.warning("could not cache ESM prior (%s)", exc)


def _esm_real(sequence: str, model: str) -> Optional[List[float]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is server-only
        log.warning("ESM real path needs torch (%s); using mock prior", exc)
        return None

    probs = _fair_esm_probs(sequence, model) or _hf_esm_probs(sequence, model)
    if probs is None:
        log.warning("no ESM backend (fair-esm / transformers) available; mock prior")
        return None
    import numpy as np

    p = np.clip(np.asarray(probs, dtype=float), 1e-9, None)  # [L, 20]
    p = p / p.sum(axis=1, keepdims=True)
    ent = -(p * np.log2(p)).sum(axis=1) / _LOG2_20          # normalised 0..1
    return [round(float(x), 4) for x in ent]


def _aa_logit_indices(symbols) -> List[int]:
    return [symbols.index(a) for a in _AA]


def _fair_esm_probs(sequence: str, model: str):
    """[L, 20] predicted AA probabilities from fair-esm, or None."""
    try:
        import esm  # fair-esm
        import torch
    except Exception:
        return None
    try:
        loader = getattr(esm.pretrained, model, None)
        if loader is None:
            loader = esm.pretrained.esm2_t33_650M_UR50D
        mdl, alphabet = loader()
        mdl.eval()
        if torch.cuda.is_available():
            mdl = mdl.cuda()
        bc = alphabet.get_batch_converter()
        _, _, toks = bc([("q", sequence)])
        if torch.cuda.is_available():
            toks = toks.cuda()
        with torch.no_grad():
            logits = mdl(toks)["logits"][0]          # [L+2, vocab]
        aa_idx = [alphabet.get_idx(a) for a in _AA]
        # strip BOS/EOS -> per-residue rows, select the 20 AA columns
        sub = logits[1:1 + len(sequence)][:, aa_idx]
        return torch.softmax(sub.float(), dim=-1).cpu().numpy()
    except Exception as exc:
        log.warning("fair-esm inference failed (%s)", exc)
        return None


def _hf_esm_probs(sequence: str, model: str):
    """[L, 20] predicted AA probabilities from HuggingFace transformers, or None."""
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except Exception:
        return None
    try:
        name = model if "/" in model else f"facebook/{model}"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForMaskedLM.from_pretrained(name).eval()
        if torch.cuda.is_available():
            mdl = mdl.cuda()
        enc = tok(sequence, return_tensors="pt")
        if torch.cuda.is_available():
            enc = {k: v.cuda() for k, v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits[0]            # [L+2, vocab]
        aa_idx = [tok.convert_tokens_to_ids(a) for a in _AA]
        sub = logits[1:1 + len(sequence)][:, aa_idx]
        return torch.softmax(sub.float(), dim=-1).cpu().numpy()
    except Exception as exc:
        log.warning("transformers ESM inference failed (%s)", exc)
        return None


def _esm_mock(sequence: str) -> List[float]:
    """Deterministic per-position variability in [0,1]. Not physical, but stable
    and position-dependent so downstream code sees realistic structure."""
    out: List[float] = []
    L = len(sequence)
    for i, aa in enumerate(sequence):
        s = derive_seed(0xE5A2, aa, str(i % 7))
        v = (s % 1000) / 1000.0
        # bias the ends toward variable, a periodic 'buried' trough toward
        # conserved, so the profile isn't flat noise.
        edge = 1.0 if (i < 3 or i > L - 4) else 0.0
        core = 0.5 + 0.5 * math.cos(2 * math.pi * i / 11.0)
        v = max(0.0, min(1.0, 0.5 * v + 0.3 * edge + 0.2 * (1.0 - core)))
        out.append(round(v, 4))
    return out
