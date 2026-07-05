"""Protein-ligand complex prediction (spec section 9).

real: Boltz / Boltz-2 (https://github.com/jwohlwend/boltz).
mock: deterministic synthetic complex (structure + pocketed ligand + scores).

Outputs a diffusion-sample **ensemble** with rich confidence/affinity metrics.
Per the data policy these are FEATURES / sample-weights / weak-labels /
filters - never supervised labels (see ml/labels.py, docs/ML_DATA_POLICY.md).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import yaml

from evoliez.adapters.base import (
    fail_unless_mock_allowed,
    place_ligand_in_pocket,
    synthetic_structure,
    write_min_pdb,
)
from evoliez.config import Backend, ComplexPredictionConfig
from evoliez.logging_utils import get_logger
from evoliez.types import BoltzSample, Complex, Ligand, LigandAtom
from evoliez.utils.gpu import apply_gpu_selection
from evoliez.utils.seeds import derive_seed
from evoliez.utils.subprocess_utils import require, run

log = get_logger("evoliez.boltz")

# Confidence/affinity keys carried on Complex.metrics (Boltz output, §1/§5).
METRIC_KEYS = [
    "confidence_score", "ptm", "iptm", "ligand_iptm", "complex_plddt",
    "complex_iplddt", "complex_pde", "complex_ipde",
    "affinity_pred_value", "affinity_probability_binary",
    "affinity_pred_value1", "affinity_pred_value2", "ensemble_disagreement",
]


def predict_complex(
    label: str,
    sequence: str,
    ligand: Ligand,
    cfg: ComplexPredictionConfig,
    outdir: Path,
    *,
    backend: Backend,
    dry_run: bool = False,
    msa_path: Optional[Path] = None,
    seed: int = 1234,
    extra_ligands: Optional[List[Ligand]] = None,
    gpu_device: Optional[str] = None,
    metal_ccd: Optional[str] = None,
) -> Complex:
    outdir.mkdir(parents=True, exist_ok=True)
    if backend is Backend.real:
        return _predict_real(
            label, sequence, ligand, cfg, outdir, dry_run=dry_run,
            msa_path=msa_path, seed=seed, extra_ligands=extra_ligands,
            gpu_device=gpu_device, metal_ccd=metal_ccd,
        )
    return _predict_mock(label, sequence, ligand, cfg, outdir,
                         extra_ligands=extra_ligands)


# --------------------------------------------------------------------------- #
# Metric synthesis (mock) / aggregation
# --------------------------------------------------------------------------- #
def _synth_metrics(seed: int, sample_idx: int, n: int) -> dict:
    """Deterministic, monotone-ish confidence: sample 0 best, later worse."""
    h = derive_seed(seed, "m", str(sample_idx))
    decay = sample_idx / max(1, n)
    base = 0.85 - 0.30 * decay - (h % 12) / 100.0
    base = max(0.30, min(0.95, base))
    iptm = max(0.2, base - 0.05 - (h % 8) / 100.0)
    lig_iptm = max(0.15, iptm - 0.04 - (h % 7) / 100.0)
    plddt = max(0.4, base + 0.03 - (h % 6) / 100.0)
    iplddt = max(0.35, plddt - 0.05 - (h % 5) / 100.0)
    pde = round(1.0 + 4.0 * (1.0 - base) + (h % 10) / 10.0, 3)  # Å, lower better
    ipde = round(pde + 0.5 + (h % 8) / 10.0, 3)
    aff = round(-1.0 - 4.0 * base - (h % 50) / 100.0, 3)  # log10(IC50)-like
    return {
        "confidence_score": round(0.8 * plddt + 0.2 * iptm, 4),
        "ptm": round(base, 4),
        "iptm": round(iptm, 4),
        "ligand_iptm": round(lig_iptm, 4),
        "complex_plddt": round(plddt, 4),
        "complex_iplddt": round(iplddt, 4),
        "complex_pde": pde,
        "complex_ipde": ipde,
        "affinity_pred_value": aff,
        "affinity_probability_binary": round(min(0.99, max(0.01, base)), 4),
        "affinity_pred_value1": round(aff - (h % 20) / 100.0, 3),
        "affinity_pred_value2": round(aff + (h % 20) / 100.0, 3),
    }


def _finalize(cx: Complex) -> Complex:
    if cx.samples:
        best = cx.samples[0]
        cx.metrics = dict(best.metrics)
        v1 = cx.metrics.get("affinity_pred_value1")
        v2 = cx.metrics.get("affinity_pred_value2")
        if v1 is not None and v2 is not None:
            cx.metrics["ensemble_disagreement"] = round(abs(v1 - v2), 4)
        cx.confidence = cx.metrics.get("confidence_score", cx.confidence)
        cx.affinity_score = cx.metrics.get("affinity_pred_value", cx.affinity_score)
    return cx


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #
def _predict_mock(
    label: str, sequence: str, ligand: Ligand, cfg: ComplexPredictionConfig,
    outdir: Path, extra_ligands: Optional[List[Ligand]] = None,
) -> Complex:
    seed = derive_seed(0xB01D, label, sequence[:32])
    struct = synthetic_structure(sequence, seed=seed, method="boltz-mock")
    n = max(1, cfg.diffusion_samples)

    # One shared consensus pocket pose; samples vary modestly around it so
    # recurring (residue, ligand-atom) contacts emerge -> meaningful
    # ensemble contact frequency (the priority-1 feature).
    base = place_ligand_in_pocket(struct, ligand, seed=seed)
    samples: List[BoltzSample] = []
    order = []
    for i in range(n):
        s = derive_seed(seed, "s", str(i))
        drift = 0.15 + 0.9 * (i / max(1, n))  # small wobble, grows with idx
        moved = []
        for k, a in enumerate(base):
            o = ((derive_seed(s, str(k)) % 200) / 100.0 - 1.0) * drift
            na = LigandAtom(**vars(a))
            na.coord = (a.coord[0] + o, a.coord[1] - o * 0.4, a.coord[2] + o * 0.3)
            moved.append(na)
        metrics = _synth_metrics(seed, i, n)
        rp = [round(metrics["complex_plddt"], 3)] * len(struct.residues)
        samples.append(BoltzSample(idx=i, ligand_atoms=moved, metrics=metrics,
                                   residue_plddt=rp))
        order.append((metrics["confidence_score"], i))

    order.sort(reverse=True)
    samples = [samples[i] for _, i in order]
    for j, smp in enumerate(samples):
        smp.idx = j

    best = samples[0]
    # v2 multi-ligand: place each co-modelled EXTRA ligand in the pocket too, so the mock
    # complex carries the FULL functional state (cofactor/substrate/metal), not just the
    # primary design ligand — s06/s09 then see every functional partner.
    extra_poses: Dict[str, List[LigandAtom]] = {}
    for el in extra_ligands or []:
        try:
            extra_poses[el.id] = place_ligand_in_pocket(
                struct, el, seed=derive_seed(seed, "extra", el.id))
        except Exception:  # noqa: BLE001 — a placement failure must not break the mock
            pass
    pdb = outdir / f"{label}_complex.pdb"
    write_min_pdb(pdb, struct, best.ligand_atoms)
    struct.pdb_path = str(pdb)
    cx = Complex(
        structure=struct,
        ligand=Ligand(
            id=ligand.id, smiles=ligand.smiles, atoms=best.ligand_atoms,
            formal_charge=ligand.formal_charge,
            n_rotatable_bonds=ligand.n_rotatable_bonds, source=ligand.source,
            charges_mol2=ligand.charges_mol2, allow_am1bcc=ligand.allow_am1bcc,
        ),
        method="boltz-mock",
        path=str(pdb),
        samples=samples,
        extra_ligand_atoms=extra_poses,
    )
    return _finalize(cx)


# --------------------------------------------------------------------------- #
# Real
# --------------------------------------------------------------------------- #
def _to_a3m(src: Path, dst: Path) -> Optional[Path]:
    """Boltz-2 rejects FASTA MSAs ('only a3m or csv'); s03 writes aligned
    FASTA. A block alignment (equal-length, '-' gaps, no a3m insertion
    semantics) is valid a3m once uppercased, so rewrite it with an .a3m
    extension. Returns None if the source is missing/empty."""
    try:
        text = src.read_text()
    except OSError:
        return None
    recs: List[tuple[str, str]] = []
    hdr, buf = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if hdr is not None:
                recs.append((hdr, "".join(buf)))
            hdr, buf = line.rstrip(), []
        elif line.strip():
            buf.append(line.strip())
    if hdr is not None:
        recs.append((hdr, "".join(buf)))
    recs = [(h, s.upper().replace(".", "-").replace(" ", "-"))
            for h, s in recs if s.strip()]
    if not recs:
        return None
    # a3m REQUIRES the query (first record) ungapped: its residues DEFINE the
    # match columns that every MSA feature is indexed by. s03 emits a BLOCK
    # alignment whose query row itself carries '-' gaps (insertion columns from
    # divergent homologs), so naively relabelling it .a3m hands Boltz a gapped
    # query — the MSA columns then no longer map to query residues and the fold
    # silently degrades (patchy low pLDDT). Restrict every row to the query's
    # non-gap columns so the query is ungapped (len == target) and each homolog
    # stays aligned to it (insertions relative to the query are dropped, which is
    # valid a3m and does not affect per-query-residue features).
    q_seq = recs[0][1]
    match_cols = [i for i, c in enumerate(q_seq) if c != "-"]
    if match_cols and len(match_cols) != len(q_seq):
        recs = [(h, "".join(s[i] if i < len(s) else "-" for i in match_cols))
                for h, s in recs]
    dst.parent.mkdir(parents=True, exist_ok=True)   # defensive: target dir may not exist
    dst.write_text("".join(f"{h}\n{s}\n" for h, s in recs))
    return dst


def _build_spec(
    label: str,
    sequence: str,
    ligand: Ligand,
    cfg: ComplexPredictionConfig,
    outdir: Path,
    msa_path: Optional[Path],
    extra_ligands: Optional[List[Ligand]] = None,
    metal_ccd: Optional[str] = None,
):
    """Build a single rep's Boltz input spec (protein A + primary ligand B +
    co-modelled extra ligands C+ + the MSA/affinity wiring). Returns
    (spec_dict, resolved_msa_path) — the resolved MSA path may differ from the
    input (FASTA→a3m rewrite, or None if the file was unusable) and the caller
    uses it to keep the `--use_msa_server` guard correct. Shared verbatim by the
    per-rep (`_predict_real`) and the GPU-batched (`predict_batch`) paths so both
    emit the IDENTICAL spec.
    """
    spec = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},
            {"ligand": {"id": "B", "smiles": ligand.smiles}},
        ],
    }
    # Co-modelled cofactors/substrates as their OWN ligand entities (chains C,
    # D, …) so Boltz sees the complete active site. The affinity binder stays
    # "B" (the design-target ligand); the extras are structural context.
    _chains = "CDEFGHIJKLMNOPQRSTUVWXYZ"
    for cid, el in zip(_chains, extra_ligands or []):
        spec["sequences"].append({"ligand": {"id": cid, "smiles": el.smiles}})
    # ROADMAP_V5 V5-2 — co-fold the physiological metal ion (e.g. Mg2+) as a CCD ion so the
    # anionic substrate + cofactor are pre-organized around it during diffusion rather than
    # repelling to a non-productive pose. Boltz accepts `ccd` XOR `smiles` per ligand entity
    # (schema.py:1028-1034). The ion is DROPPED from the parsed ligand set (metal-ion elements
    # are skipped in the structure parsers) — MD re-places it deterministically as an Amber
    # ion; it never reaches OpenFF. `MG` etc. are standard PDB CCD codes in Boltz's dictionary.
    _n_extra = len(extra_ligands or [])
    if metal_ccd and _n_extra < len(_chains):
        spec["sequences"].append({"ligand": {"id": _chains[_n_extra], "ccd": metal_ccd}})
    if cfg.predict_affinity:
        spec["properties"] = [{"affinity": {"binder": "B"}}]
    if msa_path is not None:
        mp = Path(msa_path)
        if mp.suffix.lower() in (".a3m", ".csv"):
            a3m: Optional[Path] = mp
        else:
            a3m = _to_a3m(mp, outdir / f"{label}_msa.a3m")
        if a3m is not None:
            spec["sequences"][0]["protein"]["msa"] = str(a3m)
            msa_path = a3m  # keep the --use_msa_server guard correct
        else:
            log.warning("MSA %s unusable; falling back to MSA server", mp)
            msa_path = None
    # No MSA file AND no MSA server -> explicit single-sequence mode. Boltz needs
    # the `msa: empty` sentinel; omitting `msa` entirely makes it error out.
    if msa_path is None and not cfg.use_msa_server:
        spec["sequences"][0]["protein"]["msa"] = "empty"
    return spec, msa_path


def _build_predict_cmd(
    in_path: Path,
    out_dir: Path,
    cfg: ComplexPredictionConfig,
    seed: int,
    *,
    any_msa_server: bool,
) -> list:
    """Build the `boltz predict <in_path> --out_dir <out_dir>` command.
    `in_path` is one YAML (per-rep path) or a directory of YAMLs (batch path) —
    Boltz handles both. `any_msa_server` adds `--use_msa_server` (true only when
    at least one input needs the remote server). Shared so the per-rep and the
    batched paths carry the SAME perf flags + `--no_kernels` logic."""
    cmd = [
        "boltz", "predict", str(in_path), "--out_dir", str(out_dir),
        "--diffusion_samples", str(max(1, cfg.diffusion_samples)),
        "--seed", str(seed),
        # Boltz defaults to mmCIF; force PDB so the structure parser works
        # (a CIF backstop parser also exists below).
        "--output_format", "pdb",
    ]
    if any_msa_server:
        cmd.append("--use_msa_server")
    if not getattr(cfg, "use_kernels", False):
        # Boltz-2 hard-fails (ModuleNotFoundError cuequivariance_torch) if the
        # optimized kernels' optional dep is missing; --no_kernels uses the
        # pure-torch path. Stock `pip install boltz` has no cuequivariance.
        cmd.append("--no_kernels")
    # SPEED-ONLY Boltz flags from the environment (do NOT affect the output, so
    # they are NOT in the config/run-fingerprint): how many diffusion samples to
    # run in parallel on the GPU (memory-bound), dataloader workers, and the
    # preprocessing thread count (Boltz default 1). The launch sets these to
    # match the box (e.g. MAX_PARALLEL_SAMPLES=10, NUM_WORKERS=2, threads=4) so
    # multiple Boltz processes don't oversubscribe the CPU.
    for env_key, flag in (
        ("EVOLIEZ_BOLTZ_MAX_PARALLEL_SAMPLES", "--max_parallel_samples"),
        ("EVOLIEZ_BOLTZ_NUM_WORKERS", "--num_workers"),
        ("EVOLIEZ_BOLTZ_PREPROCESSING_THREADS", "--preprocessing-threads"),
    ):
        val = os.environ.get(env_key)
        if val and val.strip().isdigit():
            cmd += [flag, val.strip()]
    return cmd


def _predict_real(
    label: str,
    sequence: str,
    ligand: Ligand,
    cfg: ComplexPredictionConfig,
    outdir: Path,
    *,
    dry_run: bool,
    msa_path: Optional[Path],
    seed: int = 1234,
    extra_ligands: Optional[List[Ligand]] = None,
    gpu_device: Optional[str] = None,
    metal_ccd: Optional[str] = None,
) -> Complex:
    # dry-run previews the FULL command set (like Vina) without the tool
    # installed, writes the exact Boltz input YAML so the contract can be
    # eyeballed, and returns the same structured mock contract as a real run.
    if dry_run:
        log.info("[dry-run] boltz predict (diffusion_samples=%d) for %s",
                 cfg.diffusion_samples, label)
    else:
        require("boltz")
        # gpu_device pins THIS subprocess (parallel ensemble across GPUs); the
        # global apply_gpu_selection would race + put every worker on one GPU.
        if gpu_device is None:
            apply_gpu_selection()
    spec, msa_path = _build_spec(
        label, sequence, ligand, cfg, outdir, msa_path, extra_ligands,
        metal_ccd=metal_ccd,
    )

    yml = outdir / f"{label}_boltz_input.yaml"
    yml.write_text(yaml.safe_dump(spec, sort_keys=False))

    # Per-label deterministic seed so the diffusion ensemble (and every feature
    # derived from it) is REPRODUCIBLE across reruns of the same input. Without
    # --seed, Boltz draws fresh samples every run, so a crash-resume mid-stage
    # mixes two RNG draws and provenance can't reconstruct the result. Distinct
    # per label so WT / each homolog / each mutant don't share an identical draw.
    boltz_seed = derive_seed(seed, "boltz", label) % (2 ** 31 - 1)
    cmd = _build_predict_cmd(
        yml, outdir, cfg, boltz_seed,
        any_msa_server=cfg.use_msa_server and msa_path is None,
    )
    boltz_env = (
        {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_device)}
        if gpu_device is not None else None
    )
    # Resume: if a COMPLETE Boltz output already exists in this outdir (>= the
    # requested diffusion samples), reuse it instead of recomputing. This is
    # scoped to `outdir`, so a caller that batches many predictions (the s06b
    # ensemble) MUST give each its OWN outdir, otherwise this — and the result
    # discovery below — would pick up a different prediction's files.
    want = max(1, cfg.diffusion_samples)
    done = [p for r in sorted(outdir.glob("boltz_results_*"))
            for p in r.rglob("*.pdb") if not p.name.endswith("_complex.pdb")]
    if dry_run or len(done) < want:
        run(cmd, dry_run=dry_run, timeout=None, env=boltz_env)
    else:
        log.info("reusing %d existing Boltz model(s) for %s (skip recompute)",
                 len(done), label)

    if dry_run:
        return _predict_mock(label, sequence, ligand, cfg, outdir,
                         extra_ligands=extra_ligands)

    # Boltz writes predictions ONLY under boltz_results_*/ (predictions/...).
    # Scope discovery there and exclude our own mock fallback (*_complex.pdb)
    # so a stale/mock file is never mis-read as a real Boltz structure
    # (previously a leftover wt_complex.pdb was stamped method=boltz2).
    roots = sorted(outdir.glob("boltz_results_*"))
    found: List[Path] = []
    for root in roots:
        found += sorted(root.rglob("*.cif")) + sorted(root.rglob("*.pdb"))
    found = [p for p in found if not p.name.endswith("_complex.pdb")]
    if not found:
        fail_unless_mock_allowed(
            f"Boltz produced no prediction for {label} "
            "(no boltz_results_*/.../*.cif|pdb)")
        log.warning(
            "Boltz produced no prediction for %s "
            "(no boltz_results_*/.../*.cif|pdb); mock fallback", label,
        )
        return _predict_mock(label, sequence, ligand, cfg, outdir,
                         extra_ligands=extra_ligands)

    return _assemble_real_complex(
        outdir, found, sequence, ligand, cfg.primary_method
    )


def _assign_residue_plddt(residues, token_plddt) -> None:
    """Map Boltz per-TOKEN pLDDT (protein residues first, then ligand atoms)
    onto Residue.plddt. Only the leading protein-token slice is used (ligand
    tokens dropped); values are normalized to the 0-100 scale the mock path
    uses so Residue.plddt is backend-consistent. No-op on a missing array.

    Without this the real path left Residue.plddt at its 0.0 dataclass default
    and every per-residue confidence feature (confidence windows, pocket /
    catalytic pLDDT, GNN node feature, ML export, PDB b-factor) read zero."""
    if not residues or not token_plddt:
        return
    for r, v in zip(residues, token_plddt[:len(residues)]):
        fv = float(v)
        r.plddt = fv * 100.0 if fv <= 1.5 else fv


def _assemble_real_complex(
    outdir, found, sequence, ligand, primary_method, *, structure_only: bool = False
) -> Complex:
    """Parse a finished Boltz run (shared backbone + per-sample ligand poses +
    confidence + per-residue pLDDT) into a finalized Complex. Split out of
    _predict_real so the real-output assembly is testable without the binary.

    ``structure_only`` (default False — every existing caller is byte-identical):
    parse ONLY the representative model (``found[0]``) for ``cx.structure`` +
    ``cx.ligand`` (+ ``structure.pdb_path``) and SKIP the full diffusion-sample
    ensemble (``_parse_real_samples``) and the per-sample ``_load_plddt`` —
    leaving ``cx.samples == []``. The s06b DOCKING phase only consumes the
    receptor + reference ligand, never the samples, so this drops the per-stem
    re-read of every model PDB + confidence JSON + plddt npz for that path."""
    structure_file = found[0]
    cx = _parse_real_structure(structure_file, sequence, ligand)
    cx.method = primary_method
    cx.path = str(structure_file)
    if structure_only:
        # Structure-only: keep cx.structure + cx.ligand + structure.pdb_path,
        # leave cx.samples = [] and skip the per-sample pLDDT load. _finalize is
        # a no-op without samples, so cx.metrics/confidence stay at defaults.
        return _finalize(cx)
    cx.samples = _parse_real_samples(outdir, cx.ligand.atoms, found)
    if not cx.samples:  # at least one sample from aggregate scores
        m = _parse_one_confidence(outdir) or {}
        cx.samples = [BoltzSample(idx=0, ligand_atoms=cx.ligand.atoms, metrics=m)]
    # Wire real per-residue pLDDT onto the structure we kept (found[0] is model
    # index 0, aligned with plddt index 0).
    _assign_residue_plddt(cx.structure.residues, _load_plddt(outdir, 0))
    return _finalize(cx)


# --------------------------------------------------------------------------- #
# GPU-batched ensemble (s06b): ONE `boltz predict <dir>` per GPU, then a
# per-stem-SCOPED parse of each rep's prediction. One model load per GPU
# instead of one per rep. Quality-neutral: same model, same diffusion_samples,
# same per-rep spec (protein+ligand+MSA) — only the launch topology changes.
# --------------------------------------------------------------------------- #
def parse_prediction_dir(
    stem_dir: Path, sequence: str, ligand: Ligand, primary_method: str,
    *, structure_only: bool = False,
) -> Optional[Complex]:
    """Parse ONE rep's Boltz prediction, SCOPED to a single
    ``predictions/<stem>/`` directory, into a finalized Complex.

    CRITICAL: a GPU-batched run writes EVERY rep in the chunk under one shared
    ``boltz_results_<chunkdir>/predictions/`` tree, one ``<stem>/`` subdir per
    rep. ``_parse_real_samples`` / ``_load_plddt`` discover samples by
    ``rglob('confidence*model_*.json')`` / ``rglob('plddt*.npz')``, so they MUST
    be rooted at the ONE stem's subdir — never the shared parent. If they were
    pointed at the parent they would mix every rep's confidence/plddt/pdb files
    into a single prediction (the exact unscoped-glob cross-contamination defect
    fixed once before). Here ``stem_dir`` IS that single subdir, so reusing
    ``_assemble_real_complex(stem_dir, <stem_dir's own pdbs>, …)`` is scoped by
    construction. Returns None if the subdir holds no structure file.

    ``structure_only`` (default False keeps every caller byte-identical): parse
    ONLY the representative model into ``cx.structure`` + ``cx.ligand`` and skip
    the full diffusion-sample ensemble (``cx.samples == []``). The s06b DOCKING
    phase needs only the receptor + reference ligand, so this avoids re-reading
    every model PDB / confidence JSON / plddt npz for that path."""
    found = sorted(stem_dir.glob("*.cif")) + sorted(stem_dir.glob("*.pdb"))
    found = [p for p in found if not p.name.endswith("_complex.pdb")]
    if not found:
        return None
    return _assemble_real_complex(
        stem_dir, found, sequence, ligand, primary_method,
        structure_only=structure_only,
    )


def write_batch_input(
    in_dir: Path,
    label: str,
    sequence: str,
    ligand: Ligand,
    cfg: ComplexPredictionConfig,
    *,
    msa_path: Optional[Path] = None,
    extra_ligands: Optional[List[Ligand]] = None,
    metal_ccd: Optional[str] = None,
) -> bool:
    """Write ONE rep's ``<label>_boltz_input.yaml`` into the shared chunk
    ``in_dir`` using the SAME spec as the per-rep path (`_build_spec`). MSA a3m
    rewrites land in ``in_dir`` too. Returns True if this rep needs the remote
    MSA server (the caller ORs these to decide ``--use_msa_server`` for the
    whole chunk). The yaml stem is ``<label>_boltz_input``; Boltz emits its
    prediction under ``predictions/<label>_boltz_input/``."""
    in_dir.mkdir(parents=True, exist_ok=True)
    spec, resolved_msa = _build_spec(
        label, sequence, ligand, cfg, in_dir, msa_path, extra_ligands,
        metal_ccd=metal_ccd,
    )
    (in_dir / f"{label}_boltz_input.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False)
    )
    return cfg.use_msa_server and resolved_msa is None


def predict_batch(
    in_dir: Path,
    out_dir: Path,
    cfg: ComplexPredictionConfig,
    *,
    seed: int,
    gpu_device: Optional[str] = None,
    any_msa_server: bool = False,
    dry_run: bool = False,
) -> Path:
    """Run ONE ``boltz predict <in_dir> --out_dir <out_dir>`` over a whole chunk
    of rep YAMLs = ONE model load for the whole chunk (vs one per rep). Pinned to
    ``gpu_device`` via CUDA_VISIBLE_DEVICES. ``in_dir`` and ``out_dir`` MUST be
    distinct dirs — Boltz rescans ``in_dir`` and errors if ``out_dir`` is nested
    inside it. Reuses the per-rep cmd builder so the chunk carries the identical
    EVOLIEZ_BOLTZ_* perf flags + ``--no_kernels`` logic.

    SEED: the per-rep ``derive_seed(seed,'boltz',label)`` can't apply to a single
    batched process (one process, one ``--seed``). We use a deterministic
    per-CHUNK seed ``derive_seed(seed,'boltz_batch',gpu)``. This changes WHICH
    diffusion samples are drawn versus the per-rep path, but it is quality-neutral
    (same model, same sample count, same spec) and reproducible (fixed function of
    the run seed + GPU), so a crash-resume reproduces the same chunk.

    Returns the ``boltz_results_<in_dirname>`` directory that holds the chunk's
    ``predictions/<stem>/`` subdirs (so the caller can scoped-parse each rep).
    Resume is PER-STEM: every input stem with >= diffusion_samples models is
    skipped and only the INCOMPLETE stems are re-folded (one missing stem no
    longer re-folds the whole chunk). The already-complete stems' outputs stay in
    place, so the returned results dir still contains every stem. If all stems are
    already complete the Boltz subprocess is skipped entirely."""
    if not dry_run:
        require("boltz")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = out_dir / f"boltz_results_{in_dir.name}"

    batch_seed = derive_seed(seed, "boltz_batch", str(gpu_device)) % (2 ** 31 - 1)
    boltz_env = (
        {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_device)}
        if gpu_device is not None else None
    )

    # Per-stem chunk-skip on resume: a stem is DONE iff its predictions/<stem>/
    # already holds >= diffusion_samples model PDBs. Only the INCOMPLETE stems
    # get re-folded — the all-or-nothing check (one missing stem -> re-fold the
    # whole chunk) is replaced by filtering the chunk's input down to just the
    # incomplete stems. Already-complete stems' outputs stay in place so the
    # downstream per-stem parse_prediction_dir still finds every stem.
    want = max(1, cfg.diffusion_samples)
    stems = sorted(p.stem for p in in_dir.glob("*.yaml"))
    preds_root = results_dir / "predictions"

    def _stem_done(stem: str) -> bool:
        sub = preds_root / stem
        if not sub.is_dir():
            return False
        models = [p for p in sub.glob("*.pdb")
                  if not p.name.endswith("_complex.pdb")]
        return len(models) >= want

    incomplete = [] if dry_run else [s for s in stems if not _stem_done(s)]

    # Dry-run previews the FULL chunk command. A live run with every stem already
    # complete skips the subprocess entirely (mirrors the per-rep resume-skip).
    if not dry_run and stems and not incomplete:
        log.info("reusing complete batched Boltz chunk %s (%d stems, skip "
                 "recompute)", results_dir.name, len(stems))
        return results_dir

    # Full launch when: dry-run, no per-stem layout yet (every stem missing), or
    # every stem is incomplete. Run Boltz over the original in_dir unchanged so
    # the output lands directly in boltz_results_<in_dir.name>/.
    if dry_run or len(incomplete) == len(stems):
        cmd = _build_predict_cmd(
            in_dir, out_dir, cfg, batch_seed, any_msa_server=any_msa_server
        )
        run(cmd, dry_run=dry_run, timeout=None, env=boltz_env)
        return results_dir

    # PARTIAL resume: re-fold ONLY the incomplete stems. Build a filtered input
    # dir holding just those stems' YAMLs (+ any MSA file they reference that
    # lives in in_dir), run Boltz over it into a SEPARATE results tree, then move
    # each freshly produced predictions/<stem>/ back into the canonical
    # results_dir so downstream parses every stem. results_dir name is unchanged.
    import shutil

    log.info("partial-resume batched Boltz chunk %s: re-folding %d of %d stem(s) "
             "(%s)", results_dir.name, len(incomplete), len(stems),
             ", ".join(incomplete))
    filt_in = out_dir / f"{in_dir.name}_resume"
    if filt_in.exists():
        shutil.rmtree(filt_in)
    filt_in.mkdir(parents=True, exist_ok=True)
    for stem in incomplete:
        yml = in_dir / f"{stem}.yaml"
        shutil.copy2(yml, filt_in / yml.name)
        # Copy any MSA file the spec references that lives inside in_dir so the
        # (possibly relative) path still resolves from the filtered dir; absolute
        # external MSAs keep working untouched.
        try:
            spec = yaml.safe_load(yml.read_text()) or {}
            for seq_entry in spec.get("sequences", []):
                mp = (seq_entry.get("protein", {}) or {}).get("msa")
                if not mp or mp == "empty":
                    continue
                msa_src = Path(mp)
                cand = msa_src if msa_src.is_absolute() else (in_dir / mp)
                if cand.is_file() and cand.parent == in_dir:
                    shutil.copy2(cand, filt_in / cand.name)
        except Exception:
            pass  # MSA copy is best-effort; absolute paths still resolve

    filt_out = out_dir / f"_resume_out_{in_dir.name}"
    filt_out.mkdir(parents=True, exist_ok=True)
    cmd = _build_predict_cmd(
        filt_in, filt_out, cfg, batch_seed, any_msa_server=any_msa_server
    )
    run(cmd, dry_run=False, timeout=None, env=boltz_env)

    # Merge the freshly folded stems back into the canonical predictions tree.
    fresh_preds = filt_out / f"boltz_results_{filt_in.name}" / "predictions"
    preds_root.mkdir(parents=True, exist_ok=True)
    for stem in incomplete:
        src = fresh_preds / stem
        if not src.is_dir():
            continue
        dst = preds_root / stem
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
    shutil.rmtree(filt_in, ignore_errors=True)
    shutil.rmtree(filt_out, ignore_errors=True)
    return results_dir


def _parse_real_samples(outdir: Path, lig_atoms, structure_files=None) -> List[BoltzSample]:
    from evoliez.features.ligand import relabel_to_canonical

    # Per-model structure files, sorted to the SAME model order as the
    # confidence/plddt files so position i refers to the same diffusion sample.
    structure_files = list(structure_files or [])
    canonical = list(lig_atoms)
    samples: List[BoltzSample] = []
    conf_files = sorted(outdir.rglob("confidence*model_*.json")) or sorted(
        outdir.rglob("confidence*.json")
    )
    aff = {}
    for jf in outdir.rglob("affinity*.json"):
        try:
            aff = json.loads(jf.read_text())
        except Exception:
            aff = {}
    for i, jf in enumerate(conf_files):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        m = {k: float(d[k]) for k in d if isinstance(d.get(k), (int, float))}
        for ak in ("affinity_pred_value", "affinity_probability_binary",
                   "affinity_pred_value1", "affinity_pred_value2"):
            if ak in aff:
                m[ak] = float(aff[ak])
        m.setdefault(
            "confidence_score",
            0.8 * m.get("complex_plddt", 0.0) + 0.2 * m.get("iptm", 0.0),
        )
        rp = _load_plddt(outdir, i)
        # DISTINCT ligand pose per diffusion sample: parse THIS model's own
        # structure file and adopt its HETATM coords onto the canonical atom
        # ids. Without this every sample reused one pose, so the priority-#1
        # ensemble contact-frequency feature was degenerate (freq always 0/1).
        pose = canonical
        if i < len(structure_files):
            _res, model_lig, _extra = _parse_structure_atoms(structure_files[i])
            if model_lig:
                if canonical:
                    relabeled, _ok = relabel_to_canonical(model_lig, canonical)
                    pose = relabeled or model_lig
                else:
                    pose = model_lig
        samples.append(BoltzSample(idx=i, ligand_atoms=pose, metrics=m,
                                   residue_plddt=rp))
    samples.sort(key=lambda s: -s.metrics.get("confidence_score", 0.0))
    for j, s in enumerate(samples):
        s.idx = j
    return samples


def _parse_one_confidence(outdir: Path) -> Optional[dict]:
    for jf in outdir.rglob("confidence*.json"):
        try:
            d = json.loads(jf.read_text())
            return {k: float(v) for k, v in d.items()
                    if isinstance(v, (int, float))}
        except Exception:
            pass
    return None


def _load_plddt(outdir: Path, idx: int) -> List[float]:
    try:
        import numpy as np

        npzs = sorted(outdir.rglob("plddt*.npz"))
        if idx < len(npzs):
            arr = np.load(npzs[idx])
            key = arr.files[0]
            return [float(x) for x in np.asarray(arr[key]).ravel()]
    except Exception:
        pass
    return []


def _parse_cif_atoms(path: Path, fold_only_metal: "str | None" = None):
    """Minimal mmCIF _atom_site loop parser (Boltz default output format).
    Returns (residues[CA], ligand_atoms[HETATM]). ``fold_only_metal`` (a CCD code)
    drops ONLY the metal residue EvoLiEZ co-folded — never a target's own metal."""
    from evoliez.types import LigandAtom, Residue

    drop = _fold_only_metal_resname(fold_only_metal)
    lines = path.read_text().splitlines()
    cols: list[str] = []
    residues: list[Residue] = []
    lig: list[LigandAtom] = []
    extra: dict = {}                       # v2 multi-ligand: non-primary ligand chains
    primary_chain: Optional[str] = None   # first ligand chain = primary (chain B)
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            hdr = []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                hdr.append(lines[j].strip())
                j += 1
            if hdr:
                cols = [h.split(".", 1)[1] for h in hdr]
                idx = {c: k for k, c in enumerate(cols)}
                need = ("group_PDB", "type_symbol", "label_atom_id",
                        "Cartn_x", "Cartn_y", "Cartn_z")
                if all(c in idx for c in need):
                    seqk = ("label_seq_id" if "label_seq_id" in idx
                            else "auth_seq_id")
                    while j < len(lines):
                        s = lines[j].strip()
                        if not s or s.startswith(("#", "loop_", "_")):
                            break
                        p = s.split()
                        if len(p) >= len(cols):
                            grp = p[idx["group_PDB"]]
                            x, y, z = (float(p[idx["Cartn_x"]]),
                                       float(p[idx["Cartn_y"]]),
                                       float(p[idx["Cartn_z"]]))
                            if grp == "ATOM" and p[idx["label_atom_id"]] == "CA":
                                try:
                                    ri = int(p[idx.get(seqk, -1)])
                                except (ValueError, KeyError):
                                    ri = len(residues) + 1
                                residues.append(
                                    Residue(index=ri, aa="X", ca=(x, y, z),
                                            sidechain_centroid=(x, y, z))
                                )
                            elif grp == "HETATM":
                                # primary ligand = first ligand chain -> lig;
                                # co-modelled extra ligands (other chains) -> extra
                                # (v2 multi-ligand: full functional state for s06).
                                # Drop ONLY the fold-only metal WE injected (by CCD residue
                                # name) -> never a target's own heme/metalloenzyme metal.
                                rn = (p[idx["label_comp_id"]].upper()
                                      if "label_comp_id" in idx else "")
                                el = p[idx["type_symbol"]]
                                # drop ONLY a true monatomic ion (resname==element==injected CCD)
                                # -> a polyatomic ligand sharing the code keeps every atom
                                if not (drop and rn == drop
                                        and (el or "").strip().upper() == drop):
                                    ck = ("label_asym_id" if "label_asym_id" in idx
                                          else "auth_asym_id" if "auth_asym_id" in idx
                                          else None)
                                    ch = p[idx[ck]] if ck else None
                                    if primary_chain is None:
                                        primary_chain = ch
                                    if ch is None or ch == primary_chain:
                                        lig.append(LigandAtom(
                                            id=f"{el}{len(lig)}", element=el or "C",
                                            coord=(x, y, z)))
                                    else:
                                        _e = extra.setdefault(ch, [])
                                        _e.append(LigandAtom(
                                            id=f"{el}{len(_e)}", element=el or "C",
                                            coord=(x, y, z)))
                        j += 1
                i = j
                continue
        i += 1
    return residues, lig, extra


# CCD codes of monatomic metal ions EvoLiEZ may co-fold as a fold-only prior (Mg2+ bridging the
# CAR substrate/cofactor anions). Used ONLY to VALIDATE a caller-supplied `fold_only_metal`: the
# parser drops a metal residue ONLY when the caller states which ion WE injected into THIS Boltz
# spec. A target's OWN metal — heme Fe, a metalloenzyme catalytic Zn, a metal-centered organic
# ligand — is NEVER dropped (fold_only_metal is None for those), so this stays a general-purpose
# enzyme parser. The co-folded ion is re-placed at MD as an Amber ion, so dropping it here keeps
# s09/s10 seeing the same organic ligand set as before the co-fold — only substrate coords change.
_KNOWN_METAL_CCDS = {"MG", "MN", "ZN", "CA", "FE", "NI", "CO", "CU", "K", "NA"}


def _fold_only_metal_resname(fold_only_metal: "str | None") -> "str | None":
    """Normalize+validate the caller's injected metal CCD to a resname to drop, or None.
    Only a RECOGNIZED metal CCD is honoured, so a stray value can never drop a real ligand."""
    if not fold_only_metal:
        return None
    code = str(fold_only_metal).strip().upper()
    return code if code in _KNOWN_METAL_CCDS else None


def _parse_pdb_atoms(path: Path, fold_only_metal: "str | None" = None):
    """Minimal PDB parser: (residues[CA], ligand_atoms[HETATM]). Mirrors
    _parse_cif_atoms so the structure backbone and the per-sample ligand poses
    share one extraction path. ``fold_only_metal`` (a CCD code, e.g. 'MG') drops
    ONLY the metal residue EvoLiEZ co-folded into this spec — never a target's own metal."""
    from evoliez.types import LigandAtom, Residue

    drop = _fold_only_metal_resname(fold_only_metal)
    residues: list[Residue] = []
    lig: list[LigandAtom] = []
    extra: dict = {}                       # v2 multi-ligand: non-primary ligand chains
    primary_chain: Optional[str] = None
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            idx = int(line[22:26])
            x, y, z = (float(line[30:38]), float(line[38:46]),
                       float(line[46:54]))
            residues.append(
                Residue(index=idx, aa="X", ca=(x, y, z),
                        sidechain_centroid=(x, y, z))
            )
        elif line.startswith("HETATM"):
            # The PRIMARY (design-target) ligand = the FIRST ligand chain (chain B); keep it
            # in `lig` (its contacts/docking-reference/canonical-relabel must NOT be polluted
            # by other ligands). Co-modelled cofactors/substrates/metals are separate chains
            # (C, D, …) -> collected into `extra` (v2 multi-ligand: the FULL functional state
            # for the s06 design mask). The PDB chain id is column 22 (0-based index 21).
            el = line[76:78].strip() or line[12:14].strip()
            if drop and line[17:20].strip().upper() == drop and el.upper() == drop:
                continue      # fold-only MONATOMIC metal WE injected (resname==element==CCD)
            chain = line[21] if len(line) > 21 else " "
            if primary_chain is None:
                primary_chain = chain
            x, y, z = (float(line[30:38]), float(line[38:46]),
                       float(line[46:54]))
            if chain == primary_chain:
                lig.append(LigandAtom(id=f"{el}{len(lig)}", element=el or "C",
                                      coord=(x, y, z)))
            else:
                _e = extra.setdefault(chain, [])
                _e.append(LigandAtom(id=f"{el}{len(_e)}", element=el or "C",
                                     coord=(x, y, z)))
    return residues, lig, extra


def _parse_structure_atoms(path: Path, fold_only_metal: "str | None" = None):
    """(residues[CA], ligand_atoms[HETATM]) from a Boltz .cif or .pdb model."""
    if path.suffix in (".cif", ".mmcif"):
        return _parse_cif_atoms(path, fold_only_metal)
    return _parse_pdb_atoms(path, fold_only_metal)


def _parse_real_structure(pdb: Path, sequence: str, ligand: Ligand,
                          fold_only_metal: "str | None" = None) -> Complex:
    from evoliez.types import ProteinStructure

    residues, lig_atoms, extra_poses = _parse_structure_atoms(pdb, fold_only_metal)
    for i, r in enumerate(residues):
        if i < len(sequence):
            r.aa = sequence[i]
    if not lig_atoms:
        lig_atoms = ligand.atoms
    # atom-index lock: keep canonical ids/chemistry, adopt Boltz coordinates
    from evoliez.features.ligand import relabel_to_canonical

    lig_atoms, locked = relabel_to_canonical(lig_atoms, ligand.atoms)
    if not locked and ligand.atoms:
        # relabel_to_canonical already logged WHY (count mismatch or
        # chemistry-graph atom-order not verified); record the consequence and
        # tag the ligand source so id-keyed features can treat it as unreliable.
        log.warning(
            "Boltz ligand atom ids NOT verified for %s (canonical=%d, "
            "tool=%d); downstream id-keyed features may be inconsistent",
            getattr(ligand, "id", "?"), len(ligand.atoms), len(lig_atoms),
        )
    struct = ProteinStructure(
        sequence=sequence, residues=residues, method="boltz", pdb_path=str(pdb)
    )
    return Complex(structure=struct, ligand=Ligand(
        id=ligand.id, smiles=ligand.smiles, atoms=lig_atoms,
        formal_charge=ligand.formal_charge,
        source=ligand.source if locked else ligand.source + "|reindexed",
        charges_mol2=ligand.charges_mol2, allow_am1bcc=ligand.allow_am1bcc,
    ), extra_ligand_atoms=extra_poses)
