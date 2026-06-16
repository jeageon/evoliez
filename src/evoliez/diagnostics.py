"""Preflight environment check (``evoliez doctor``).

First thing to run on the server: reports what's installed / missing / risky
before a real run, so environment problems surface immediately instead of
mid-pipeline. Informational - never raises, exit code reflects only hard
blockers.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from evoliez import __version__

OK, WARN, MISSING, BLOCK = "ok", "warn", "missing", "block"

# python package -> which capability it unlocks
_PY_DEPS = {
    "rdkit": "ligand 3D / chemistry (else synthetic ligand)",
    "Bio": "FASTA / sequence QC (biopython)",
    "sklearn": "logistic reranker / GNN logistic fallback",
    "xgboost": "family + mutation reranker (else heuristic)",
    "openmm": "real MD (else mock MD-lite)",
    "openmmforcefields": "real MD ligand FF (GAFF; else skipped_parameterization)",
    "openff.toolkit": "real MD ligand topology (else mock MD-lite)",
    "pdbfixer": "real MD terminal/missing-atom repair (else Amber template err)",
    "torch": "EvoLigand-GNN train/score (else heuristic family model)",
    "torch_geometric": "optional GNN backend (not required)",
}

# external binary -> (stage, env override)
_TOOLS = {
    "boltz": ("s04_complex / s06b (real)", None),
    "mmseqs": ("s02_homolog (real)", None),
    "jackhmmer": ("s02_homolog (alt)", None),
    "blastp": ("s02_homolog (alt)", None),
    "mafft": ("s03_msa (real)", None),
    "vina": ("s05/s09 docking (real)", None),
    "gnina": ("s05/s09 docking (real, CUDA)", None),
    "obabel": ("docking prep (real)", None),
    "antechamber": ("s10_md GAFF ligand params (real, ambertools)", None),
    "parmchk2": ("s10_md GAFF ligand params (real, ambertools)", None),
    "foldx": ("s09 stability (real, licensed)", None),
    "iupred2a.py": ("disorder (real, else proxy)", None),
    "snakemake": ("workflow/Snakefile (optional)", None),
}

_ENV_VARS = ["BOLTZ_CACHE", "EVOLIEZ_LIGANDMPNN", "EVOLIEZ_DIFFDOCK",
             "EVOLIEZ_DATA_DIR", "EVOLIEZ_DB_DIR", "CUDA_VISIBLE_DEVICES"]


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    def add(self, name, status, detail=""):
        self.checks.append(Check(name, status, detail))

    @property
    def n_block(self) -> int:
        return sum(1 for c in self.checks if c.status == BLOCK)


def _has_module(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _load_cfg(config_path: Optional[str]):
    """(cfg, error) — load the run config once so the tool/dep checks know what
    this run actually needs. error is the exception if it failed to load."""
    if not config_path:
        return None, None
    try:
        from evoliez.config import load_config

        return load_config(config_path), None
    except Exception as exc:  # surfaced as a BLOCK by _check_config
        return None, exc


def _required_under_real(cfg) -> tuple:
    """(required_tools, required_pydeps) whose ABSENCE must BLOCK — i.e. the
    real tools/deps THIS config will actually invoke. Anything not required is a
    mere MISSING (informational). Makes `doctor` a real production gate without
    false-blocking configs that don't use a given tool (audit P0 #6)."""
    tools: set = set()
    deps: set = set()
    if cfg is None:
        return tools, deps
    from evoliez.config import Backend

    real = cfg.backend is Backend.real

    def sreal(stage: str) -> bool:
        return cfg.backend_for(stage) is Backend.real

    if real:
        deps.add("rdkit")                       # ligand chemistry: no real fallback
    if sreal("s04_complex"):
        tools.add("boltz")                      # the central real stage
    if real and not cfg.msa.remote_server:      # local homolog search + align
        tools.add({"mmseqs2": "mmseqs"}.get(cfg.homologs.method, cfg.homologs.method))
        if cfg.msa.method == "mafft":
            tools.add("mafft")
    if sreal("s05_docking") or sreal("s09_nonmd"):
        methods = cfg.validation.redocking.methods
        for m in methods:
            if m in _TOOLS:                      # vina|gnina (diffdock = python+env)
                tools.add(m)
        if "vina" in methods:
            tools.add("obabel")
    if sreal("s09_nonmd") and cfg.validation.stability.method == "foldx":
        tools.add("foldx")
    if sreal("s10_md") and cfg.validation.md.enabled:
        deps.add("openmm")
    return tools, deps


def collect(config_path: Optional[str] = None) -> Report:
    r = Report()
    r.add("evoliez", OK, f"v{__version__}")

    import sys

    r.add("python", OK, sys.version.split()[0])

    # Load the run config up front so a missing tool/dep this run will ACTUALLY
    # invoke is reported as a BLOCK (a real gate), not a mere MISSING (P0 #6).
    cfg, cfg_err = _load_cfg(config_path)
    req_tools, req_deps = _required_under_real(cfg)

    for mod, why in _PY_DEPS.items():
        if _has_module(mod):
            st = OK
        else:
            st = BLOCK if mod in req_deps else MISSING
        r.add(f"py:{mod}", st, why)

    if _has_module("torch"):
        try:
            import torch

            r.add(
                "cuda",
                OK if torch.cuda.is_available() else WARN,
                f"torch {torch.__version__}, "
                f"cuda={torch.cuda.is_available()}, "
                f"n_gpu={torch.cuda.device_count() if torch.cuda.is_available() else 0}",
            )
        except Exception as exc:
            r.add("cuda", WARN, str(exc))

    for tool, (stage, _) in _TOOLS.items():
        present = shutil.which(tool) is not None
        if present:
            st = OK
        else:
            st = BLOCK if tool in req_tools else MISSING
        r.add(f"tool:{tool}", st, stage)

    # GPU inventory (shared server)
    try:
        from evoliez.utils.gpu import query_gpus

        gpus = query_gpus()
        if gpus:
            for g in gpus:
                st = OK if g.mem_free_mib >= 16000 else WARN
                r.add(
                    f"gpu:{g.index}",
                    st,
                    f"{g.name} {g.mem_free_mib}MiB free, {g.util_pct}% util",
                )
        else:
            r.add("gpu", WARN, "no nvidia-smi (laptop / CPU-only)")
    except Exception as exc:
        r.add("gpu", WARN, str(exc))

    # storage: warn if cwd on a near-full FS; check /mnt targets
    for label, p in [("cwd", Path.cwd()),
                     ("/mnt/data2", Path("/mnt/data2")),
                     ("/mnt/data", Path("/mnt/data"))]:
        if p.exists():
            free_gb = shutil.disk_usage(p).free / 1e9
            st = OK if free_gb >= 30 else (WARN if free_gb >= 5 else BLOCK)
            writable = os.access(p, os.W_OK)
            r.add(
                f"disk:{label}",
                st if writable else WARN,
                f"{free_gb:.0f} GB free, writable={writable}",
            )

    for ev in _ENV_VARS:
        val = os.environ.get(ev)
        r.add(f"env:{ev}", OK if val else MISSING, val or "(unset)")

    if config_path:
        _check_config(r, cfg, cfg_err, config_path)
    return r


def _check_config(r: Report, cfg, cfg_err, config_path: str) -> None:
    from evoliez.config import Backend

    if cfg_err is not None:
        r.add("config", BLOCK, f"failed to load: {cfg_err}")
        return
    r.add("config", OK, f"{config_path} (backend={cfg.backend.value})")

    out = Path(cfg.project.output_dir)
    on_safe = str(out).startswith(("/mnt/data2", "/mnt/data"))
    if cfg.backend is Backend.real and not on_safe:
        r.add("config:output_dir", WARN,
              f"{out} not on /mnt/data2 - root may be full")
    else:
        r.add("config:output_dir", OK, str(out))

    if cfg.backend is Backend.real and not cfg.msa.remote_server:
        db = cfg.homologs.database
        if not db:
            r.add("config:homolog_db", BLOCK,
                  "backend=real needs homologs.database or msa.remote_server")
        elif not Path(db).parent.exists():
            r.add("config:homolog_db", WARN, f"{db} (parent missing)")
        else:
            r.add("config:homolog_db", OK, str(db))

    if cfg.gnn.enabled and not _has_module("torch"):
        r.add("config:gnn", WARN,
              "gnn.enabled but torch missing -> heuristic fallback")

    # A real run must NOT proceed on the bundled illustrative placeholder or
    # with catalytic/fixed tokens that disagree with the target sequence -
    # both silently corrupt the science (active-site protection + scoring).
    if cfg.backend is Backend.real:
        from evoliez.stages.s01_input_preprocess import _residue_token_wt

        ic = cfg.input
        seq = (ic.target_sequence or "").strip().upper()
        hdr = ""
        if not seq and ic.target_fasta and Path(ic.target_fasta).exists():
            ls = Path(ic.target_fasta).read_text().splitlines()
            hdr = next((x for x in ls if x.startswith(">")), "")
            seq = "".join(x.strip() for x in ls
                          if x and not x.startswith(">")).upper()
        if not seq:
            r.add("config:target", BLOCK,
                  "backend=real but no target sequence resolved")
        elif any(w in hdr.lower() for w in ("illustrative", "example",
                                            "placeholder")):
            r.add("config:target", BLOCK,
                  f"target is the bundled placeholder ({hdr.strip()}); set "
                  "input.target_fasta to the real target + verified "
                  "catalytic/fixed numbering before a real run")
        else:
            bad = []
            for kind in ("catalytic_residues", "fixed_residues",
                         "known_binding_site"):
                for t in (getattr(ic, kind, None) or []):
                    wt, pos = _residue_token_wt(t)
                    if (wt and pos and 1 <= pos <= len(seq)
                            and seq[pos - 1] not in (wt, "X")):
                        bad.append(f"{t}->{seq[pos - 1]}{pos}")
            if bad:
                r.add("config:catalytic", BLOCK,
                      "residue tokens disagree with target sequence: "
                      + ", ".join(bad))
            else:
                r.add("config:target", OK,
                      f"{len(seq)} aa, residue tokens consistent")
