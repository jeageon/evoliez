"""Stage 06b - family interaction-geometry model (spec 9.2 + 13.3).

For clustered representative homologs: predict the Boltz **diffusion-sample
ensemble**, extract per-ligand-atom interaction-distance fingerprints, use
ensemble **contact frequency** + Boltz pose reliability (as a SAMPLE WEIGHT,
never a label), statistically select family-consensus poses (augmented
training data), and train a self-supervised consensus/outlier classifier.
Exports the pose- and edge-level ML datasets.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from evoliez.adapters.boltz import predict_complex
from evoliez.context import RunContext
from evoliez.features.boltz_features import ensemble_contacts, pose_consensus
from evoliez.features.interaction_descriptor import (
    complex_fingerprint,
    describe,
    fingerprint_dim,
)
from evoliez.ml.datasets import edge_rows, pose_rows
from evoliez.ml.interaction_model import InteractionModel
from evoliez.ml.pose_selection import PoseRecord, select_poses
from evoliez.stages.base import Stage


def _resolve_n_reps(sizes: List[int], n_total: int, cfg):
    """How many representatives to build the ensemble from. Either the explicit
    ``representative_homologs`` int (-1 = every cluster) or, for ``"auto"``, the
    number of LARGEST subfamily clusters needed to cover ``representative_coverage``
    of the homolog pool, clamped to [representative_min, representative_max] — so
    a deeper / more diverse MSA auto-scales to more representatives. ``sizes`` is
    the per-cluster member count, sorted descending. Returns (n, human note)."""
    spec = cfg.representative_homologs
    n_clusters = len(sizes)
    if spec != "auto":
        n = int(spec)
        if n < 0:
            return n_clusters, f"all {n_clusters} clusters (representative_homologs=-1)"
        return n, f"fixed representative_homologs={n}"
    target = cfg.representative_coverage * n_total
    cum = n = 0
    for s in sizes:  # sizes already sorted desc
        cum += s
        n += 1
        if cum >= target:
            break
    lo, hi = cfg.representative_min, cfg.representative_max
    n_clamped = max(lo, min(hi, n))
    note = (
        f"auto: {n}/{n_clusters} largest clusters cover "
        f"{cfg.representative_coverage:.0%} of {n_total} homologs"
        + (f" -> clamped to {n_clamped} [{lo},{hi}]" if n_clamped != n else "")
    )
    return n_clamped, note


def _pick_representatives(homologs, cfg):
    """Representative homologs for the Boltz family ensemble.

    One representative (highest identity) per subfamily cluster, taking the
    LARGEST clusters first (best-supported subfamilies — not cluster-id order),
    topped up by identity-diverse extras to reach the target count. The count is
    an explicit ``representative_homologs`` int or, when ``"auto"``, derived from
    the MSA cluster structure (see :func:`_resolve_n_reps`). Returns (reps, note)."""
    by_cluster: Dict[int, list] = {}
    for h in homologs:
        by_cluster.setdefault(h.cluster_id, []).append(h)
    # rank clusters by size (desc); tie-break on the best identity inside each
    clusters = sorted(
        by_cluster.values(),
        key=lambda hs: (len(hs), max(h.identity for h in hs)),
        reverse=True,
    )
    n, note = _resolve_n_reps([len(hs) for hs in clusters], len(homologs), cfg)
    reps = [max(hs, key=lambda h: h.identity) for hs in clusters[:n]]
    if len(reps) < n:  # fewer clusters than requested -> top up by identity
        chosen = {id(r) for r in reps}
        extra = sorted(
            (h for h in homologs if id(h) not in chosen),
            key=lambda h: h.identity, reverse=True,
        )
        reps.extend(extra[: n - len(reps)])
    return reps[:n], note


def _prepare_rep_msas(ctx, cfg, reps, cp_cfg):
    """Resolve the per-representative MSA source (cfg.rep_msa). Returns
    (cp_cfg, {sequence: a3m_path | None}).

    "server" -> remote ColabFold (unchanged). "local"/"single" disable the MSA
    server: "local" batch-builds each rep's a3m from one mmseqs GPU search vs the
    UniRef30 DB; "single" runs Boltz single-sequence."""
    mode = cfg.rep_msa
    if mode == "server":
        return cp_cfg, {}
    cp_cfg = cp_cfg.model_copy(update={"use_msa_server": False})
    if mode == "single":
        return cp_cfg, {}
    # local: one batched mmseqs search over all reps -> per-rep a3m (cached)
    from evoliez.adapters.local_msa import ensure_local_msas

    h = ctx.config.homologs
    search_db = h.databases.get("mmseqs2") or h.database
    if not search_db:
        raise ValueError(
            "interaction_model.rep_msa='local' needs a UniRef mmseqs DB in "
            "homologs.database or homologs.databases['mmseqs2']"
        )
    cache = (Path(h.cache_dir) / "rep_local_msa" if h.cache_dir
             else ctx.paths.structures / "representatives" / "local_msa")
    msas = ensure_local_msas(
        [r.sequence for r in reps], cache,
        mmseqs_bin=h.mmseqs_bin or "mmseqs", search_db=search_db,
        gpu=h.mmseqs_gpu, max_seqs=cfg.rep_msa_max_seqs,
        threads=h.search_threads, dry_run=ctx.dry_run,
    )
    return cp_cfg, msas


def _pose_records_from_complex(i, cx, identity, cutoff, k_nearest):
    """Per-sample interaction fingerprints -> (rep_index, [PoseRecord], pose_rows)
    for ONE representative's predicted Complex. The SINGLE source of pose-record
    semantics: group_id=f'hom_{i:03d}', one PoseRecord per diffusion sample with
    its complex_fingerprint, identity_to_target, and pred_score (Boltz confidence
    as a SAMPLE WEIGHT, never a label). Shared by the per-rep path (`_rep_worker`)
    and the GPU-batched Phase-2 parse so both emit IDENTICAL-shape records."""
    gid = f"hom_{i:03d}"
    recs = []
    for s in (cx.samples or []):
        fp = complex_fingerprint(
            cx.structure, s.ligand_atoms, cutoff=cutoff, k_nearest=k_nearest
        )
        recs.append(PoseRecord(
            group_id=gid, fingerprint=fp, msa_membership=1.0,
            identity_to_target=identity,
            # Boltz pose reliability -> SAMPLE WEIGHT (not a label)
            pred_score=round(s.metrics.get("confidence_score", cx.confidence), 4),
        ))
    return i, recs, pose_rows(gid, cx)


def _rep_stem_dir(outdir, i: int) -> Path:
    """The single ``predictions/<stem>/`` subdir a per-rep ``predict_complex``
    writes its Boltz model + confidence + plddt into, for representative ``i``.
    Re-derived (not retained at fold time) so the per-rep docking step can
    scoped-parse the rep structure later via ``parse_prediction_dir``. Mirrors
    the layout ``predict_complex`` produces under ``hom_{i:03d}/``."""
    label = f"hom_{i:03d}"
    return (Path(outdir) / label / f"boltz_results_{label}_boltz_input"
            / "predictions" / f"{label}_boltz_input")


def _rep_worker(payload, gpu):
    """Build ONE representative's pose records: Boltz diffusion ensemble ->
    per-sample interaction-distance fingerprints. Module-level + fully picklable
    so it runs in a ProcessPool worker (GIL-free), with the Boltz subprocess
    pinned to ``gpu``. Returns (rep_index, [PoseRecord], pose_rows, stem_dir) —
    the stem_dir is the rep's ``predictions/<stem>/`` subdir so the per-rep
    docking augmentation can re-parse this rep's structure."""
    (i, seq, identity, ligand, cp_cfg, backend, outdir, seed, dry_run,
     msa_path, cutoff, k_nearest, extra_ligands) = payload
    # CRITICAL: each representative gets its OWN output subdir. Boltz writes
    # boltz_results_* and predict_complex discovers them by globbing the outdir;
    # a SHARED outdir made every rep glob `found[0]` = hom_000 and mix all reps'
    # confidence files into one prediction (silent cross-contamination + an
    # O(N^2) re-parse of the whole tree per rep). Per-rep scoping fixes both.
    rep_outdir = Path(outdir) / f"hom_{i:03d}"
    rep_outdir.mkdir(parents=True, exist_ok=True)
    cx = predict_complex(
        f"hom_{i:03d}", seq, ligand, cp_cfg, rep_outdir,
        backend=backend, dry_run=dry_run, seed=seed,
        msa_path=msa_path, gpu_device=gpu, extra_ligands=extra_ligands,
    )
    i, recs, prows = _pose_records_from_complex(i, cx, identity, cutoff, k_nearest)
    return i, recs, prows, _rep_stem_dir(outdir, i)


def _run_chunk(gpu, payloads):
    """Run a round-robin chunk of representatives on ONE pinned GPU (one worker
    process per GPU). Serial within the chunk; chunks run in parallel."""
    return [_rep_worker(p, gpu) for p in payloads]


# --------------------------------------------------------------------------- #
# GPU-batched ensemble (quality-neutral speedup). Per-rep predict_complex reloads
# the Boltz model ONCE PER REP (150 reps -> 150 loads). Batched: one
# `boltz predict <chunk_dir>` per GPU = one model load per GPU. Two phases:
#   1. GPU, batched   — split reps round-robin across GPUs; ONE Boltz process
#                       per GPU over that GPU's chunk of rep YAMLs.
#   2. CPU, parallel  — per-stem-SCOPED parse + fingerprinting of each rep,
#                       fanned across processes (the GIL-bound part).
# --------------------------------------------------------------------------- #
def _run_batch_chunk(payload):
    """Phase 1 on ONE GPU: write every rep YAML in this chunk into a shared chunk
    IN_DIR, then ONE `boltz predict <in_dir>` pinned to the GPU. The whole chunk =
    ONE model load. Returns (gpu, results_dir, [(rep_index, label, seq, identity)])
    so Phase 2 can scoped-parse each rep from results_dir/predictions/<stem>/."""
    from evoliez.adapters.boltz import predict_batch, write_batch_input

    (gpu, reps_meta, ligand, cp_cfg, base_outdir, seed, extra_ligands,
     rep_msas) = payload
    # IN_DIR and OUT_DIR MUST differ: Boltz rescans IN_DIR and errors if OUT_DIR
    # is nested inside it. Per-GPU names keep concurrent chunks isolated.
    chunk_in = Path(base_outdir) / f"_batch_in_gpu{gpu}"
    chunk_out = Path(base_outdir) / f"_batch_out_gpu{gpu}"
    any_msa_server = False
    for (i, seq, _identity) in reps_meta:
        label = f"hom_{i:03d}"
        any_msa_server |= write_batch_input(
            chunk_in, label, seq, ligand, cp_cfg,
            msa_path=rep_msas.get(seq.strip()), extra_ligands=extra_ligands,
        )
    results_dir = predict_batch(
        chunk_in, chunk_out, cp_cfg, seed=seed, gpu_device=gpu,
        any_msa_server=any_msa_server,
    )
    return gpu, str(results_dir), [(i, f"hom_{i:03d}", seq, identity)
                                   for (i, seq, identity) in reps_meta]


def _parse_rep_worker(payload):
    """Phase 2 (CPU, GIL-bound, one process per rep): scoped-parse ONE rep from
    its OWN predictions/<stem>/ subdir, then fingerprint. Each rep is parsed ONLY
    from its stem subdir (parse_prediction_dir scopes the confidence/plddt globs
    there), so a shared chunk output dir never cross-contaminates reps. Returns
    (rep_index, [PoseRecord], pose_rows, stem_dir) — same shape as _rep_worker
    (the stem_dir lets the per-rep docking augmentation re-parse this rep)."""
    from evoliez.adapters.boltz import parse_prediction_dir

    (i, label, seq, identity, ligand, cp_cfg, results_dir, cutoff,
     k_nearest) = payload
    stem_dir = Path(results_dir) / "predictions" / f"{label}_boltz_input"
    cx = parse_prediction_dir(stem_dir, seq, ligand, cp_cfg.primary_method)
    if cx is None:
        # No prediction for this stem (Boltz skipped/failed it): emit an empty
        # rep rather than crashing the whole batch — mirrors a degenerate run.
        return i, [], [], stem_dir
    i, recs, prows = _pose_records_from_complex(i, cx, identity, cutoff, k_nearest)
    return i, recs, prows, stem_dir


def _run_batched_ensemble(reps, payloads, gpu_list, ligand, cp_cfg, outdir,
                          seed, extra_ligands, rep_msas, log):
    """GPU-batched ensemble: Phase 1 (one batched Boltz process per GPU) then
    Phase 2 (parallel per-rep scoped parse + fingerprint). Returns the same
    ``out`` list the per-rep fallback builds: out[i] = (recs, prows, stem_dir),
    in rep index order. ``payloads`` is reused only for its (i, seq, identity)
    tuples so round-robin assignment matches the fallback path."""
    import multiprocessing as mp
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # FORK on Linux: the batch call-path (Boltz subprocess + numpy parsing) is
    # torch/xgboost-FREE, so children never touch CUDA in-process, and fork
    # avoids the spawn-time libomp double-load segfault on the server. macOS has
    # no clean fork -> spawn (local tests only).
    mpctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")

    # Round-robin reps across GPUs (same partition as the per-rep fallback).
    # payload layout: (i, seq, identity, ligand, cp_cfg, backend, outdir, seed,
    # dry_run, msa_path, cutoff, k_nearest, extra_ligands).
    cutoff, k_nearest = payloads[0][10], payloads[0][11]
    reps_meta_all = [(p[0], p[1], p[2]) for p in payloads]  # (i, seq, identity)
    chunks = [
        (g, reps_meta_all[gi::len(gpu_list)], ligand, cp_cfg, str(outdir),
         seed, extra_ligands, rep_msas)
        for gi, g in enumerate(gpu_list)
    ]

    # --- Phase 1: one batched Boltz process per GPU (concurrent) ---
    log.info("batched ensemble Phase 1: %d reps over %d GPUs %s -> %d model "
             "load(s) total", len(reps), len(gpu_list), gpu_list, len(gpu_list))
    parse_jobs = []  # (i, label, seq, identity, results_dir)
    with ProcessPoolExecutor(max_workers=len(gpu_list), mp_context=mpctx) as ex:
        futs = [ex.submit(_run_batch_chunk, ch) for ch in chunks]
        for fut in as_completed(futs):
            _gpu, results_dir, reps_in_chunk = fut.result()
            for (i, label, seq, identity) in reps_in_chunk:
                parse_jobs.append((i, label, seq, identity, results_dir))

    # --- Phase 2: parallel per-rep scoped parse + fingerprint (GIL-bound) ---
    log.info("batched ensemble Phase 2: parsing %d reps (per-stem scoped, "
             "parallel)", len(parse_jobs))
    out: List = [None] * len(reps)
    parse_payloads = [
        (i, label, seq, identity, ligand, cp_cfg, results_dir, cutoff, k_nearest)
        for (i, label, seq, identity, results_dir) in parse_jobs
    ]
    n_workers = min(len(parse_payloads), (os.cpu_count() or 2))
    with ProcessPoolExecutor(max_workers=max(1, n_workers),
                             mp_context=mpctx) as ex:
        for i, recs, prows, stem_dir in ex.map(_parse_rep_worker, parse_payloads):
            out[i] = (recs, prows, stem_dir)
    # Any rep whose chunk produced no parse job (shouldn't happen) -> empty.
    for j in range(len(out)):
        if out[j] is None:
            out[j] = ([], [], None)
    return out


def _boltz_consensus_fp(records):
    """Median interaction fingerprint over the BOLTZ records (role == "") — the
    family-geometry teacher pattern docking poses are scored against. Matches the
    consensus select_poses computes from the same Boltz-only rows. Empty -> None."""
    import numpy as np
    fps = [r.fingerprint for r in records if not getattr(r, "role", "")]
    if not fps:
        return None
    return np.median(np.vstack(fps), axis=0)


def _dock_target_worker(payload, gpu):
    """Dock the design ligand into ONE target's Boltz structure (the WT or one
    representative) with every requested engine, then classify each pose against
    the FAMILY Boltz consensus. Module-level + fully picklable so it runs in a
    ProcessPool worker.

    The two references handed to ``classify_docking_poses`` are deliberately
    different in origin:
      * overlap reference  = the FAMILY ``consensus_fp`` (interaction
        fingerprints are coordinate-frame-independent) — the SAME for every
        target.
      * RMSD + catalytic ref = this TARGET's OWN Boltz ligand atoms
        (``ref_atoms``) — DIFFERENT per target. Each rep is its own coordinate
        frame, so a cross-frame RMSD against the family's atoms is meaningless.

    The docking adapters (gnina/diffdock) are torch-FREE, so pre-setting
    ``CUDA_VISIBLE_DEVICES`` here pins the docking subprocess and the worker is
    fork-safe (no in-process CUDA init). Each engine is wrapped so one failing
    engine skips that engine, not the whole target. Returns
    (target_key, [kept PoseRecord], [PoseClassification diagnostics])."""
    import numpy as np

    from evoliez.adapters.boltz import parse_prediction_dir
    from evoliez.ml.multi_engine import DockingPoseInput, classify_docking_poses

    (target_key, stem_dir, seq, ligand, consensus_fp, methods, dock_cfg,
     docking_root, backend, cutoff, k_nearest, key_positions, primary_only,
     cfg, primary_method) = payload

    if gpu is not None:
        # apply_gpu_selection() respects a pre-set CVD -> this pins the (torch-
        # free) docking subprocess to `gpu`; fork-safe (no in-process CUDA).
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    consensus_fp = np.asarray(consensus_fp, dtype=float)
    cx = parse_prediction_dir(Path(stem_dir), seq, ligand, primary_method)
    if cx is None or not cx.ligand.atoms:
        # No structure for this target (Boltz skipped/failed it, or no ligand):
        # nothing to dock against -> contribute no augmenting rows.
        return target_key, [], []
    # RMSD + catalytic reference = this target's OWN Boltz ligand pose.
    ref_atoms = list(cx.ligand.atoms)
    # Cofactor/substrate context: keep the NON-primary het chains (the co-modelled
    # extras, e.g. formate) as FIXED receptor context so the design ligand is
    # docked into the real catalytic environment, mirroring s05. het[0] is the
    # primary (design) ligand; het[1:] are the extras. gnina honours this;
    # diffdock ignores context_chains. (Single-ligand reps -> het[1:] empty -> None.)
    from evoliez.adapters.base import het_chains_in_pdb
    _pdb = getattr(cx.structure, "pdb_path", None)
    _het = het_chains_in_pdb(_pdb) if _pdb else []
    context_chains = _het[1:] or None

    dock_inputs = []
    for method in methods:
        try:
            if method == "gnina":
                from evoliez.adapters import gnina
                pose = gnina.redock(
                    f"{target_key}_gnina", cx.structure, ref_atoms, dock_cfg,
                    Path(docking_root) / f"me_{target_key}_gnina",
                    instability=0.05, backend=backend, dry_run=False,
                    context_chains=context_chains,  # cofactor/substrate as context
                )
            elif method == "diffdock":
                from evoliez.adapters import diffdock
                pose = diffdock.redock(
                    f"{target_key}_diffdock", cx.structure, ref_atoms, dock_cfg,
                    Path(docking_root) / f"me_{target_key}_diffdock",
                    instability=0.05, smiles=ligand.smiles, backend=backend,
                    dry_run=False, context_chains=None,
                )
            else:  # unknown engine in cfg -> skip rather than crash the target
                continue
        except Exception:  # one engine failing must not sink the rest
            continue
        if pose is None or not pose.ligand_atoms:
            continue
        dock_inputs.append(DockingPoseInput(
            source=method, structure=cx.structure,
            ligand_atoms=pose.ligand_atoms, score=float(pose.score),
            score_is_better_low=(method != "diffdock"),  # diffdock conf: higher better
            primary_only=primary_only, candidate_id=target_key,
        ))

    if not dock_inputs:
        return target_key, [], []
    # overlap reference = FAMILY consensus_fp (frame-independent, same for all);
    # RMSD/catalytic reference = THIS target's own ligand atoms (per-frame).
    kept = classify_docking_poses(
        consensus_fp, ref_atoms, dock_inputs,
        cfg=cfg, key_positions=key_positions,
    )
    return target_key, list(kept), list(getattr(kept, "diagnostics", []))


def _run_per_rep_docking(targets, gpu_list, log):
    """Fan the per-target docking workers across the pinned GPUs (round-robin),
    one process per GPU, mirroring the ensemble fan-out. Each ``targets`` entry
    is a ``(_dock_target_worker payload)`` tuple. With <=1 GPU declared, runs a
    serial loop (no pool). Returns (all_kept_PoseRecords, all_diagnostics)."""
    if not targets:
        return [], []

    all_kept: List = []
    all_diags: List = []
    if len(gpu_list) <= 1:
        gpu = gpu_list[0] if gpu_list else None
        for payload in targets:
            tkey, kept, diags = _dock_target_worker(payload, gpu)
            all_kept.extend(kept)
            all_diags.extend(diags)
            log.info("multi_engine[%s]: %d kept, %d classified",
                     tkey, len(kept), len(diags))
        return all_kept, all_diags

    import multiprocessing as mp
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # FORK on Linux (the docking workers + adapters are torch/xgboost-free, so a
    # forked child never touches CUDA in-process and avoids the spawn-time libomp
    # double-load segfault); spawn on macOS (local tests only). Mirrors the
    # ensemble fan-out topology.
    mpctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
    log.info("multi_engine per-rep docking: %d targets across %d GPUs %s",
             len(targets), len(gpu_list), gpu_list)
    with ProcessPoolExecutor(max_workers=len(gpu_list), mp_context=mpctx) as ex:
        futs = {
            ex.submit(_dock_target_worker, payload, gpu_list[ti % len(gpu_list)]):
                payload[0]
            for ti, payload in enumerate(targets)
        }
        for fut in as_completed(futs):
            tkey, kept, diags = fut.result()
            all_kept.extend(kept)
            all_diags.extend(diags)
            log.info("multi_engine[%s]: %d kept, %d classified",
                     tkey, len(kept), len(diags))
    return all_kept, all_diags


def _augment_with_docking(ctx, cfg, wt, records, rep_stem_dirs, reps, gpu_list,
                          log):
    """Multi-engine augmentation, PER REPRESENTATIVE: dock the design ligand into
    the WT complex AND each representative's Boltz structure with every
    ``cfg.multi_engine_methods`` engine, classify each pose against the FAMILY
    Boltz consensus, and EXTEND ``records`` with the weighted PoseRecords.

    Matches the per-rep Boltz ensemble scale so hard-negatives are harvested
    across the whole family, not just the WT. Wrapped by the caller in
    try/except so a missing tool NEVER crashes s06b. Returns the per-pose
    diagnostics (may be empty) for logging/meta."""
    consensus_fp = _boltz_consensus_fp(records)
    if consensus_fp is None:
        log.warning("multi_engine: no Boltz consensus (empty ensemble); skip")
        return []

    methods = list(cfg.multi_engine_methods or [])
    if not methods:
        log.warning("multi_engine on but multi_engine_methods empty; skip")
        return []

    dock_cfg = ctx.config.validation.redocking
    backend = ctx.config.backend_for(InteractionModelStage.name)
    primary_method = ctx.config.complex_prediction.primary_method
    docking_root = str(ctx.paths.docking)
    key_positions = ctx.get("catalytic_positions", []) or []
    # The design ligand is docked ALONE here; if the system co-models extra
    # ligands (cofactor + substrate) those are NOT in this docking pose, so each
    # is a primary-ligand-only pose and is tagged accordingly (must never
    # override the full cofactor+substrate geometry).
    primary_only = bool(ctx.get("extra_ligands", []) or [])
    ligand = ctx.require("ligand")

    # WT stem dir: the predictions/<stem>/ subdir holding the s04 WT Boltz model,
    # derived from the kept WT structure's pdb_path; fall back to a glob.
    wt_pdb = getattr(wt.structure, "pdb_path", None)
    wt_stem = Path(wt_pdb).parent if wt_pdb else None
    if wt_stem is None or not (wt_stem.exists()):
        from evoliez.io.interaction_model_report import find_wt_complex_pdb
        found = find_wt_complex_pdb(ctx.paths.complexes)
        wt_stem = Path(found).parent if found else None

    # Target list = WT + every representative whose structure we retained. Each
    # rep is parsed in its OWN coordinate frame (its own ligand atoms supply the
    # RMSD/catalytic reference); the family consensus_fp is the shared overlap
    # reference for every target.
    targets = []
    if wt_stem is not None:
        targets.append((
            "wt", str(wt_stem), wt.structure.sequence, ligand, consensus_fp,
            methods, dock_cfg, docking_root, backend, cfg.contact_cutoff,
            cfg.k_nearest_residues, key_positions, primary_only, cfg,
            primary_method,
        ))
    else:
        log.warning("multi_engine: WT complex stem dir not found; skipping WT "
                    "docking target")
    # Per-rep docking scope cap (cfg.multi_engine_max_reps; 0 = all reps). Reps are
    # identity-sorted, so an evenly-spaced subsample spans the whole family rather
    # than only the top cluster. The WT is always docked (added above).
    rep_indices = list(range(len(reps)))
    max_reps = getattr(cfg, "multi_engine_max_reps", 0) or 0
    if max_reps and len(rep_indices) > max_reps:
        step = len(rep_indices) / max_reps
        rep_indices = [rep_indices[int(j * step)] for j in range(max_reps)]
        log.info("multi_engine: capping per-rep docking to %d of %d reps "
                 "(identity-stratified sample)", max_reps, len(reps))
    for i in rep_indices:
        rep = reps[i]
        stem = rep_stem_dirs[i] if i < len(rep_stem_dirs) else None
        if stem is None:
            continue
        targets.append((
            f"rep_{i:03d}", str(stem), rep.sequence, ligand, consensus_fp,
            methods, dock_cfg, docking_root, backend, cfg.contact_cutoff,
            cfg.k_nearest_residues, key_positions, primary_only, cfg,
            primary_method,
        ))

    if not targets:
        log.warning("multi_engine: no docking targets (WT + reps all missing "
                    "structures); records unchanged")
        return []

    all_kept, diags = _run_per_rep_docking(targets, gpu_list, log)
    records.extend(all_kept)
    # Durable audit / reproducibility artifact: EVERY classified docking pose
    # (including excluded ones), with its target, engine, role, geometry and
    # weight. s06b_artifacts.json snapshots pose_table BEFORE augmentation, so
    # this is the only durable record of the docking augmentation (paper methods).
    try:
        import json as _json
        audit = ctx.paths.interaction_graphs / "multi_engine_docking.json"
        audit.write_text(_json.dumps({
            "methods": methods, "n_targets": len(targets),
            "n_kept": len(all_kept), "n_classified": len(diags),
            "rows": [{
                "target": d.candidate_id, "source": d.source, "role": d.role,
                "rmsd_to_own_pose": d.rmsd_to_consensus,
                "fp_overlap_family": d.fp_overlap, "clash": d.clash,
                "key_contacts_ok": d.key_contacts_ok, "score": d.score,
                "sample_weight": d.sample_weight, "primary_only": d.primary_only,
            } for d in diags],
        }, indent=1, default=str))
        log.info("multi_engine docking audit -> %s (%d rows)", audit, len(diags))
    except Exception as exc:  # audit is secondary — never fail the stage
        log.warning("multi_engine docking audit write failed: %s", exc)

    by_role: Dict[str, int] = {}
    for d in diags:
        by_role[d.role] = by_role.get(d.role, 0) + 1
    log.info(
        "multi_engine: %d target(s) over %s -> %d classified pose(s), kept %d "
        "(weak_positive=%d, hard_negative=%d, strong_negative=%d, excluded=%d)"
        "%s",
        len(targets), methods, len(diags), len(all_kept),
        by_role.get("weak_positive", 0), by_role.get("hard_negative", 0),
        by_role.get("strong_negative", 0), by_role.get("excluded", 0),
        " [primary-ligand-only, tagged]" if primary_only else "",
    )
    return diags


class InteractionModelStage(Stage):
    name = "s06b_interaction"

    def run(self, ctx: RunContext) -> None:
        cfg = ctx.config.interaction_model
        if not cfg.enabled:
            self.log.info("interaction model disabled; skipping")
            ctx.put("interaction_model", None)
            return

        homologs = ctx.require("homologs")
        ligand = ctx.require("ligand")
        wt = ctx.require("wt_complex")
        backend = ctx.config.backend_for(self.name)

        # Boltz diffusion-sample ensemble. The representative COUNT is resolved
        # from the MSA cluster structure (cfg.representative_homologs may be
        # "auto"); poses_per_homolog sets the diffusion samples per homolog.
        n_samp = cfg.poses_per_homolog
        reps, rep_note = _pick_representatives(homologs, cfg)
        if ctx.dry_run:
            # dry-run only previews commands; real Boltz never runs, so do NOT
            # grind the synthetic mock over the full (e.g. 80x15) ensemble - cap
            # it so dry-run stays a fast command preview.
            reps, n_samp = reps[:2], min(n_samp, 2)
            self.log.info("[dry-run] capping s06b ensemble to %d reps x %d "
                          "samples (command-preview only)", len(reps), n_samp)
        cp_cfg = ctx.config.complex_prediction.model_copy(
            update={"diffusion_samples": n_samp}
        )
        # MSA source only matters for real Boltz; the mock ignores it (and we must
        # not invoke mmseqs under a mock/test run).
        rep_msas: Dict[str, object] = {}
        if backend.value == "real":
            cp_cfg, rep_msas = _prepare_rep_msas(ctx, cfg, reps, cp_cfg)
        self.log.info(
            "representatives=%d (of %d homologs) [%s], Boltz samples/homolog=%d, "
            "rep_msa=%s",
            len(reps), len(homologs), rep_note, n_samp, cfg.rep_msa,
        )

        # Ensemble fan-out across the pinned GPUs (CUDA_VISIBLE_DEVICES, e.g.
        # "0,2,3"). Each representative's Boltz prediction + per-sample
        # fingerprinting runs in a SEPARATE PROCESS, not a thread: the per-rep
        # CPU work (Boltz-output parsing + fingerprinting every diffusion sample)
        # is GIL-bound, so a ThreadPool serialised it onto a single core and
        # starved the other GPUs (1 GPU busy, the rest idle). Processes side-step
        # the GIL so the GPUs run genuinely in parallel. 'spawn' (not fork) keeps
        # it safe even if an upstream stage already initialised CUDA in-process.
        # A single device / unset -> sequential (classic behaviour).
        outdir = ctx.paths.structures / "representatives"
        gpu_list = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
        # Co-fold the SAME extra ligands as the WT (s04) -- cofactor + substrate --
        # so the family ensemble captures the cofactor/substrate geometry, not a
        # primary-ligand-only pocket. The chain-scoping parser then keeps only the
        # PRIMARY ligand in the fingerprint; the extras are structural context.
        extra_ligands = ctx.get("extra_ligands", []) or None
        payloads = [
            (i, h.sequence, float(h.identity), ligand, cp_cfg, backend, outdir,
             ctx.config.seed, ctx.dry_run, rep_msas.get(h.sequence.strip()),
             cfg.contact_cutoff, cfg.k_nearest_residues, extra_ligands)
            for i, h in enumerate(reps)
        ]
        out: List = [None] * len(reps)
        if len(gpu_list) > 1 and not ctx.dry_run and backend.value == "real":
            # GPU-BATCHED: ONE `boltz predict <dir>` per GPU = one model load per
            # GPU instead of one per rep (~150 reloads -> 3), then a parallel
            # per-stem-SCOPED parse. Quality-neutral. Real Boltz only — the mock
            # predictor has no batch CLI, so mock multi-GPU uses the per-rep pool.
            out = _run_batched_ensemble(
                reps, payloads, gpu_list, ligand, cp_cfg, outdir,
                ctx.config.seed, extra_ligands, rep_msas, self.log)
        elif len(gpu_list) > 1 and not ctx.dry_run:
            import multiprocessing as mp
            import sys
            from concurrent.futures import ProcessPoolExecutor, as_completed
            # per-rep process pool (mock multi-GPU): round-robin reps across GPUs.
            # FORK on Linux (torch/xgboost-free child, no libomp segfault); spawn
            # on macOS (local tests only).
            chunks = [(g, payloads[gi::len(gpu_list)]) for gi, g in enumerate(gpu_list)]
            mpctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
            self.log.info("ensemble fan-out: %d reps across %d GPUs %s "
                          "(per-rep process pool)", len(reps), len(gpu_list), gpu_list)
            with ProcessPoolExecutor(max_workers=len(gpu_list),
                                     mp_context=mpctx) as ex:
                futs = [ex.submit(_run_chunk, g, ch) for g, ch in chunks]
                for fut in as_completed(futs):
                    for i, recs, prows, stem_dir in fut.result():
                        out[i] = (recs, prows, stem_dir)
        else:
            gpu = gpu_list[0] if gpu_list else None
            for p in payloads:
                i, recs, prows, stem_dir = _rep_worker(p, gpu)
                out[i] = (recs, prows, stem_dir)

        records: List[PoseRecord] = []
        pose_table: List[dict] = []
        rep_stem_dirs: List = [None] * len(reps)
        n_poses_total = 0
        for idx, (recs, prows, stem_dir) in enumerate(out):  # rep-index order
            records.extend(recs)
            pose_table.extend(prows)
            rep_stem_dirs[idx] = stem_dir
            n_poses_total += len(recs)

        # Ensemble contact frequency (priority #1) + confidence-weighted edges,
        # from the WT Boltz ensemble -> edge-level dataset.
        econ = ensemble_contacts(
            wt.structure, wt.samples,
            cutoff=cfg.contact_cutoff,
            ligand_iptm=float(wt.metrics.get("ligand_iptm", 1.0)),
        )
        edge_ds = edge_rows(econ)
        ctx.put("ensemble_contacts", econ)
        ctx.put("pose_dataset", pose_table)
        ctx.put("edge_dataset", edge_ds)
        # Persist ALL bus artifacts (not just the model) so --resume restores a
        # complete state: previously load() rebuilt only interaction_model, so a
        # resumed s08/s11 silently read empty ensemble_contacts/pose/edge
        # datasets and exported truncated ML data.
        self._save_artifacts(ctx, econ, pose_table, edge_ds)
        self.log.info(
            "WT pose consensus: %s | %d ensemble contacts (freq>=0.5: %d)",
            pose_consensus(wt.samples), len(econ),
            sum(1 for e in econ if e.contact_frequency >= 0.5),
        )

        # MULTI-ENGINE pose-consensus augmentation (gnina / diffdock). The Boltz
        # consensus stays the positive teacher; docking poses are classified
        # relative to it (weak-positive / hard-negative / excluded) and EXTEND
        # `records` with per-source weighted rows. Gated on cfg.multi_engine
        # (default False -> records + behaviour byte-identical to before) and
        # fully wrapped: a missing/failed docking tool logs a warning and skips,
        # never crashing s06b.
        multi_engine_diags: List = []
        if cfg.multi_engine:
            try:
                multi_engine_diags = _augment_with_docking(
                    ctx, cfg, wt, records, rep_stem_dirs, reps, gpu_list,
                    self.log)
            except Exception as exc:
                self.log.warning(
                    "multi_engine augmentation failed (%s); continuing with "
                    "Boltz-only records", exc)

        sel_kw = dict(
            select_z=cfg.pose_select_mad_z,
            outlier_z=cfg.pose_outlier_mad_z,
            min_decoys_per_group=cfg.min_decoys_per_homolog,
            seed=ctx.config.seed,
            keep_alternative_band=cfg.keep_alternative_band,
            alternative_weight=cfg.alternative_weight,
            hard_decoys_per_group=cfg.hard_decoys_per_homolog,
        )
        sel = select_poses(records, **sel_kw)

        # subfamily-holdout validation: train without one homolog group,
        # check the model still ranks its held-out consensus poses above
        # decoys (guards against memorising / consensus circularity).
        holdout_auroc = self._subfamily_holdout(records, sel_kw, cfg)
        if holdout_auroc is not None:
            self.log.info("subfamily-holdout AUROC = %.3f", holdout_auroc)

        model = InteractionModel(
            cutoff=cfg.contact_cutoff, k_nearest=cfg.k_nearest_residues
        )
        model.fit(sel)
        model.save(ctx.paths.interaction_graphs / "interaction_model.json")
        ctx.put("interaction_model", model)

        meta = {
            "representatives": len(reps),
            "representative_selection": rep_note,
            "poses_total": n_poses_total,
            "train_rows": int(sel.X.shape[0]),
            "n_consensus": sel.n_positive,
            "n_alternative": sel.n_alternative,
            "n_outlier": sel.n_outlier,
            "n_decoy": sel.n_decoy,
            "n_hard_decoy": sel.n_hard_decoy,
            "subfamily_holdout_auroc": holdout_auroc,
            "model_kind": model.kind,
            "fp_dim": fingerprint_dim(cfg.k_nearest_residues),
        }
        if cfg.multi_engine:
            # Per-rep docking now classifies poses across the WHOLE family (WT +
            # ~150 reps x methods), so the per-pose `roles` list can be large:
            # keep a full role HISTOGRAM (cheap, complete) and cap the verbose
            # per-pose dump so the meta JSON stays bounded.
            ROLE_CAP = 60
            role_hist: Dict[str, int] = {}
            for d in multi_engine_diags:
                role_hist[d.role] = role_hist.get(d.role, 0) + 1
            meta["multi_engine"] = {
                "methods": list(cfg.multi_engine_methods or []),
                "n_weak_positive": sel.n_weak_positive,
                "n_hard_negative": sel.n_hard_negative,
                "poses_classified": len(multi_engine_diags),
                "role_histogram": role_hist,
                "roles_truncated": len(multi_engine_diags) > ROLE_CAP,
                "roles": [
                    {"source": d.source, "role": d.role,
                     "rmsd_to_consensus": d.rmsd_to_consensus,
                     "fp_overlap": d.fp_overlap, "clash": d.clash,
                     "key_contacts_ok": d.key_contacts_ok,
                     "score": d.score, "sample_weight": d.sample_weight,
                     "primary_only": d.primary_only}
                    for d in multi_engine_diags[:ROLE_CAP]
                ],
            }
        ctx.persist_meta("interaction_model", meta)
        self.log.info(
            "trained %s on %d rows (consensus=%d, alt=%d, outlier=%d, "
            "decoy=%d, hard=%d%s)",
            model.kind, sel.X.shape[0], sel.n_positive, sel.n_alternative,
            sel.n_outlier, sel.n_decoy, sel.n_hard_decoy,
            (f", multi_engine weak_pos={sel.n_weak_positive} "
             f"hard_neg={sel.n_hard_negative}")
            if (sel.n_weak_positive or sel.n_hard_negative) else "",
        )
        if sel.consensus.size:
            self.log.info(
                "family consensus interaction: %s",
                describe(sel.consensus, cfg.k_nearest_residues),
            )

        # s06b family interaction-geometry report (representative-selection
        # funnel + ensemble pose reliability + ensemble contact-frequency
        # fingerprint + self-supervised composition + subfamily-holdout AUROC +
        # an interactive 3D view of the conserved pocket). Built from the
        # in-context econ / pose_table / model / meta + the WT complex PDB.
        # Wrapped so a report failure NEVER fails the stage (mirrors s03_msa).
        try:
            import dataclasses
            from datetime import datetime

            from evoliez.io.interaction_model_report import (
                compute_interaction_stats, find_wt_complex_pdb,
                write_interaction_report)

            artifacts = {
                "ensemble_contacts": [dataclasses.asdict(e) for e in econ],
                "pose_dataset": pose_table,
                "edge_dataset": edge_ds,
            }
            model_dict = {
                "kind": model.kind, "fp_dim": model.fp_dim,
                "consensus": (model.consensus.tolist()
                              if model.consensus is not None else []),
                "thr": model.thr, "scale": model.scale,
                "cutoff": model.cutoff, "k_nearest": model.k_nearest,
                "n_bins": model.n_bins,
            }
            wt_pdb = (getattr(wt.structure, "pdb_path", None)
                      or find_wt_complex_pdb(ctx.paths.complexes))
            out = ctx.paths.reports / "interaction_model_report.html"
            write_interaction_report(
                out,
                target_id=ctx.config.input.target_id,
                stats=compute_interaction_stats(
                    meta, artifacts, model_dict, wt_pdb),
                generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                conditions=[
                    ("representatives", rep_note),
                    ("Boltz samples / representative", str(n_samp)),
                    ("interaction cutoff",
                     f"{cfg.contact_cutoff} Å, k-nearest "
                     f"{cfg.k_nearest_residues}"),
                    ("pose selection",
                     f"MAD-z: consensus ≤ {cfg.pose_select_mad_z}, "
                     f"outlier > {cfg.pose_outlier_mad_z}"),
                    ("model", model.kind),
                    ("validation", "subfamily / phylogenetic holdout (AUROC)"),
                    ("backend", backend.value),
                ],
            )
            self.log.info("interaction-model report: %s", out)
        except Exception as exc:  # report is secondary — never fail the stage
            self.log.warning("interaction-model report failed: %s", exc)

    def _subfamily_holdout(self, records, sel_kw, cfg):
        """Hold out one homolog group; train on the rest; AUROC of the
        held-out consensus poses vs that group's decoys."""
        if not cfg.subfamily_holdout:
            return None
        # Holdout integrity under multi-engine augmentation: pick the held
        # subfamily from BOLTZ rows only (role==""), evaluate on its Boltz poses,
        # and drop from TRAIN both the held rep's Boltz poses AND any docking rows
        # derived from it ("dock_<engine>_rep_<idx>_*") — otherwise the model
        # trains on the held structure's own docking poses and the AUROC leaks.
        # WT docking rows ("dock_<engine>_wt_*") have no held subfamily -> kept.
        boltz = [r for r in records if not getattr(r, "role", "")]
        groups = sorted({r.group_id for r in boltz})
        if len(groups) < 3:
            return None
        held = groups[-1]                                  # e.g. "hom_149"
        held_dock_tag = f"_rep_{held.split('_')[-1]}_"     # "_rep_149_"
        train_recs = [r for r in records
                      if r.group_id != held and held_dock_tag not in r.group_id]
        held_recs = [r for r in boltz if r.group_id == held]  # Boltz-only holdout
        if not train_recs or not held_recs:
            return None
        sel = select_poses(train_recs, **sel_kw)
        m = InteractionModel(
            cutoff=cfg.contact_cutoff, k_nearest=cfg.k_nearest_residues
        ).fit(sel)
        held_sel = select_poses(held_recs, **{**sel_kw, "seed": sel_kw["seed"] + 1})
        if held_sel.X.shape[0] == 0 or len(set(held_sel.y.tolist())) < 2:
            return None
        from evoliez.ml.benchmark import _auroc

        scores = [m.score_vector(x) for x in held_sel.X]
        return _auroc(scores, [int(v) for v in held_sel.y.tolist()])

    def _artifacts_path(self, ctx: RunContext):
        return ctx.paths.interaction_graphs / "s06b_artifacts.json"

    def _save_artifacts(self, ctx, econ, pose_table, edge_ds) -> None:
        import dataclasses
        import json
        self._artifacts_path(ctx).write_text(json.dumps({
            "ensemble_contacts": [dataclasses.asdict(e) for e in econ],
            "pose_dataset": pose_table,
            "edge_dataset": edge_ds,
        }))

    def load(self, ctx: RunContext) -> bool:
        import json

        from evoliez.features.boltz_features import EnsembleContact

        model_p = ctx.paths.interaction_graphs / "interaction_model.json"
        art_p = self._artifacts_path(ctx)
        # Only skip the (expensive) re-run if the FULL checkpoint is present.
        # A partial checkpoint must re-run, never hand downstream empty data.
        if not model_p.exists() or not art_p.exists():
            return False
        try:
            data = json.loads(art_p.read_text())
            econ = [EnsembleContact(**d) for d in data["ensemble_contacts"]]
            ctx.put("ensemble_contacts", econ)
            ctx.put("pose_dataset", data["pose_dataset"])
            ctx.put("edge_dataset", data["edge_dataset"])
            ctx.put("interaction_model", InteractionModel.load(model_p))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return False
        return True
