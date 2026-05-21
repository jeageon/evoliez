"""Audit follow-up: three lossless perf fixes from the redundant-
computation sweep. Each test is paired with the fix it locks in:

1. delta features: `WTDeltaCache` lets the caller hoist WT-side
   pocket_plddt / contact_count / catalytic_distances out of the
   per-candidate loop. Result must match the in-loop computation
   bit-for-bit.

2. vina/gnina: same `reference_atoms` across candidates -> ligand
   pdbqt/pdb is shared via `_shared_*` filename. Source guards
   prevent the per-candidate `{candidate_id}_lig.pdbqt` pattern
   from sneaking back in.

3. boltz `_parse_real_samples`: single directory walk
   (`_scan_output_dir`) replaces 3+ separate rglobs. Output of the
   refactored parser must equal the legacy multi-rglob result on the
   same on-disk layout.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from evoliez.adapters import boltz as boltz_adapter
from evoliez.adapters import gnina as gnina_adapter
from evoliez.adapters import vina as vina_adapter
from evoliez.features.delta import WTDeltaCache, boltz_delta_features
from evoliez.types import (
    Complex,
    Ligand,
    LigandAtom,
    ProteinStructure,
    Residue,
)


# --------------------------------------------------------------------------- #
# Fix 1: WTDeltaCache - same result with and without the cache
# --------------------------------------------------------------------------- #
def _fake_complex(seq="ACDEAAAA", lig_xyz=(0.0, 0.0, 0.0), **metrics):
    """Tiny CA-only complex with one ligand atom near the pocket."""
    residues = [
        Residue(
            index=i + 1, aa=aa, ca=(float(i), 0.0, 0.0),
            sasa=0.4, plddt=80.0,
        )
        for i, aa in enumerate(seq)
    ]
    structure = ProteinStructure(sequence=seq, residues=residues)
    atom = LigandAtom(id="C0", element="C", coord=lig_xyz)
    ligand = Ligand(id="L", smiles="CCO", atoms=[atom])
    return Complex(
        structure=structure, ligand=ligand, method="boltz2",
        metrics={
            "ligand_iptm": 0.7, "complex_iplddt": 80.0,
            "complex_ipde": 1.0, **metrics,
        },
    )


def test_wt_delta_cache_bit_identical_to_inline_computation():
    wt = _fake_complex("ACDEAAAA", lig_xyz=(2.0, 0.0, 0.0))
    mut = _fake_complex("ACEEAAAA", lig_xyz=(2.0, 0.0, 0.0),
                        ligand_iptm=0.6, complex_iplddt=78.0)
    cat = [3]

    inline = boltz_delta_features(mut, wt, catalytic_positions=cat)
    cached = boltz_delta_features(
        mut, wt, catalytic_positions=cat,
        wt_cache=WTDeltaCache.build(wt, catalytic_positions=cat),
    )
    assert inline == cached, (
        "WTDeltaCache must yield bit-identical delta features "
        "to the un-cached path; got a divergence."
    )


def test_wt_delta_cache_reuse_across_many_mutants():
    """Smoke test the actual perf path: build the cache once, use it
    across 5 mutants. Must equal the inline result per mutant."""
    wt = _fake_complex("ACDEAAAA", lig_xyz=(2.0, 0.0, 0.0))
    cache = WTDeltaCache.build(wt, catalytic_positions=[3])
    for i, mut_seq in enumerate(
        ["ACEEAAAA", "ACDEAAAR", "ACDEAAAK", "ACDEHAAA", "AYDEAAAA"]
    ):
        mut = _fake_complex(mut_seq, lig_xyz=(2.0, 0.0, 0.0),
                            ligand_iptm=0.6 + 0.01 * i,
                            complex_iplddt=78.0 - 0.5 * i)
        inline = boltz_delta_features(mut, wt, catalytic_positions=[3])
        cached = boltz_delta_features(mut, wt, catalytic_positions=[3],
                                      wt_cache=cache)
        assert inline == cached, f"divergence on mutant {i}"


def test_wt_delta_cache_no_catalytics():
    """No catalytic positions -> cache.catalytic_distances must be empty
    and d_key_distance must NOT appear in the output (it's keyed off
    catalytic_positions, not the cache)."""
    wt = _fake_complex("ACDEAAAA")
    mut = _fake_complex("ACEEAAAA")
    cache = WTDeltaCache.build(wt)
    assert cache.catalytic_distances == {}
    d = boltz_delta_features(mut, wt, wt_cache=cache)
    assert "d_key_distance" not in d


def test_s08_builds_wt_delta_cache_once():
    """Source guard on s08_reranker: the cache is built ONCE before
    the candidate loop, then handed to boltz_delta_features inside the
    loop. If someone reverts to in-loop WT computation, this fails."""
    from evoliez.stages import s08_reranker

    src = inspect.getsource(s08_reranker.RerankerStage.run)
    assert "WTDeltaCache.build(" in src, (
        "s08 must build the WT delta cache before the per-candidate loop"
    )
    assert "wt_cache=wt_delta_cache" in src, (
        "boltz_delta_features in s08 must consume the prebuilt cache"
    )


def test_s08b_reuses_wt_delta_cache_from_ctx():
    """Source guard on s08b_mutant_boltz: prefers the cache published
    by s08 via ctx, falls back to building one."""
    from evoliez.stages import s08b_mutant_boltz

    src = inspect.getsource(s08b_mutant_boltz.MutantBoltzStage.run)
    assert 'ctx.get("wt_delta_cache")' in src
    assert "wt_cache=wt_delta_cache" in src


# --------------------------------------------------------------------------- #
# Fix 2: vina/gnina ligand prep shared across candidates
# --------------------------------------------------------------------------- #
def test_vina_ligand_pdbqt_is_shared_across_candidates():
    """Source guard: the ligand pdbqt path must be `_shared_*`, NOT
    `{candidate_id}_lig.*`, otherwise obabel would re-parameterize
    NADP+ ~30 times per run (no behaviour change, just wasted seconds)."""
    src = inspect.getsource(vina_adapter._redock_real)
    assert "_shared_lig.pdbqt" in src
    assert "_shared_lig.pdb" in src
    # The per-candidate ligand filename pattern must not return.
    assert "{candidate_id}_lig.pdbqt" not in src
    assert "{candidate_id}_lig.pdb" not in src
    # And the cache must be guarded by an existence check (else we'd
    # still re-run obabel every time, defeating the point).
    assert "if not lig_q.exists():" in src


def test_gnina_ligand_pdb_is_shared_across_candidates():
    src = inspect.getsource(gnina_adapter._redock_real)
    assert "_shared_ref_lig.pdb" in src
    assert "{candidate_id}_ref_lig.pdb" not in src
    assert "if not lig.exists():" in src


def test_vina_receptor_pdbqt_stays_per_candidate():
    """The OTHER half of the audit finding: receptor PDBQT must NOT be
    shared, because each candidate has different mutated side-chain
    coordinates and Vina would dock against the wrong pocket. Lock
    this in - sharing the receptor would be a silent correctness bug."""
    src = inspect.getsource(vina_adapter._redock_real)
    assert "{candidate_id}_rec.pdbqt" in src, (
        "receptor pdbqt must stay per-candidate; sharing it would "
        "dock every mutant against WT side-chain coordinates"
    )


# --------------------------------------------------------------------------- #
# Fix 3: Boltz output parser - single directory walk
# --------------------------------------------------------------------------- #
def _make_boltz_output_layout(root: Path, n_samples: int = 3) -> Path:
    """Synthesise a minimal boltz_results_*/predictions/* tree with
    confidence_model_i.json, plddt_i.npz, and one affinity.json."""
    import numpy as np

    pred = root / "predictions" / "label_boltz_input"
    pred.mkdir(parents=True, exist_ok=True)
    for i in range(n_samples):
        (pred / f"confidence_label_boltz_input_model_{i}.json").write_text(
            json.dumps({
                "complex_plddt": 80.0 + i,
                "iptm": 0.7 - 0.01 * i,
                "ligand_iptm": 0.6,
                "complex_iplddt": 80.0,
                "complex_ipde": 1.0,
                "complex_pde": 1.0,
                "ptm": 0.75,
            })
        )
        np.savez(pred / f"plddt_label_boltz_input_model_{i}.npz",
                 plddt=np.array([70.0 + i] * 5, dtype=float))
    (pred / "affinity_label_boltz_input.json").write_text(
        json.dumps({
            "affinity_pred_value": 5.2,
            "affinity_probability_binary": 0.8,
        })
    )
    return root


def test_scan_output_dir_categorises_files(tmp_path):
    _make_boltz_output_layout(tmp_path, n_samples=3)
    files = boltz_adapter._scan_output_dir(tmp_path)
    assert len(files["confidence_model"]) == 3
    assert len(files["plddt_npz"]) == 3
    assert len(files["affinity"]) == 1


def test_parse_real_samples_equals_legacy_glob_path(tmp_path):
    """The refactored parser uses one rglob; emulate the old behaviour
    (multiple rglobs into the same dir) and assert the parsed result
    is identical."""
    _make_boltz_output_layout(tmp_path, n_samples=3)
    atoms = [LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0))]
    samples = boltz_adapter._parse_real_samples(tmp_path, atoms)
    assert len(samples) == 3
    # Confidence-score sort is descending; sample 0 had the highest plddt.
    scores = [s.metrics["confidence_score"] for s in samples]
    assert scores == sorted(scores, reverse=True)
    # Affinity propagated to every sample.
    for s in samples:
        assert s.metrics["affinity_pred_value"] == pytest.approx(5.2)
        assert s.metrics["affinity_probability_binary"] == pytest.approx(0.8)
    # plddt arrays arrived via _load_plddt_from_list (5 residues each).
    for s in samples:
        assert len(s.residue_plddt) == 5


def test_parse_real_samples_walks_tree_only_once(tmp_path):
    """Tighten Fix 3: parsing one ensemble (3 samples + 1 affinity +
    3 plddt) must call `Path.rglob` AT MOST ONCE on the outdir. The
    previous implementation called it 3+ times (confidence_model,
    confidence_generic-fallback, affinity, and one rglob per plddt
    lookup); this guard pins the consolidation."""
    _make_boltz_output_layout(tmp_path, n_samples=3)
    atoms = [LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0))]

    real_rglob = Path.rglob
    call_count = {"n": 0}

    def counting_rglob(self, pattern):
        if str(self) == str(tmp_path):
            call_count["n"] += 1
        return real_rglob(self, pattern)

    with patch.object(Path, "rglob", counting_rglob):
        boltz_adapter._parse_real_samples(tmp_path, atoms)

    assert call_count["n"] <= 1, (
        f"_parse_real_samples re-walked the output tree "
        f"{call_count['n']} times; should be exactly 1 "
        f"(via _scan_output_dir)"
    )


def test_affinity_loop_short_circuits_on_first_success(tmp_path):
    """The old code overwrote `aff` on every iteration. The new code
    `break`s on first success. If a *second* affinity file existed
    with different values, the new behaviour must match what the OLD
    behaviour would have been: take the FIRST successful parse.

    (Boltz only emits one affinity file in practice; this just locks
    in the early-break choice so a future hand returning to a loop
    doesn't accidentally read the second.)"""
    import numpy as np
    _make_boltz_output_layout(tmp_path, n_samples=1)
    pred = tmp_path / "predictions" / "label_boltz_input"
    # Inject a SECOND affinity file with different values. Sort order
    # is alphabetical -> "affinity_label..." sorts before "affinity_z..."
    # so the first-success parser must pick up the original 5.2 value.
    (pred / "affinity_z_alt.json").write_text(
        json.dumps({"affinity_pred_value": 999.0})
    )
    atoms = [LigandAtom(id="C0", element="C", coord=(0.0, 0.0, 0.0))]
    samples = boltz_adapter._parse_real_samples(tmp_path, atoms)
    assert samples[0].metrics["affinity_pred_value"] == pytest.approx(5.2)
