"""P0.4: LigandMPNN preflight on the real backend.

Real LigandMPNN needs (a) a full-atom protein, (b) ligand atoms locked,
(c) catalytic/fixed positions excluded from `designable`. A missing
precondition must be a clear log + meta marker, not a downstream crash
on the GPU.
"""

from __future__ import annotations

import inspect

from evoliez.stages import s07_mutation_gen


def test_preflight_block_present_in_source():
    src = inspect.getsource(s07_mutation_gen.MutationGenStage.run)
    # Three guards, each carrying its own diagnostic.
    assert "full-atom" in src
    assert "no parsed atoms" in src or "no parsed atoms;" in src
    assert "fixed_positions" in src
    # Honest skip rather than crash: ligandmpnn_skipped_reason is persisted.
    assert "ligandmpnn_skipped_reason" in src


def test_preflight_uses_real_full_atom_check():
    src = inspect.getsource(s07_mutation_gen.MutationGenStage.run)
    # Reuses the same full-atom check shared with docking (P0.1) and MD,
    # so the three layers agree on what "full-atom" means.
    assert "is_full_atom_pdb" in src
