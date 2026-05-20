"""Server-confirmed: s08b's previous position BEFORE s09 caused a set
mismatch between "candidates that got real per-mutant Boltz" (rerank
top-N) and "candidates that reach s10_md" (post-s09 top-K). s09 reorders
candidates by `ml_score + redocking_consistency - 0.2*ddg_fold`, so the
two top-N sets diverge when s09 promotes proxy candidates over the
rerank-top ones.

Fix: ALL_STAGES order changed to `... rerank, s09_nonmd, s08b_mutant_
boltz, s10_md, s11_final`. s08b now operates on `md_candidates` (set by
s09), guaranteeing every md candidate has a real mutant structure.

These tests lock the ordering + s08b's input source in source so a
refactor can't silently regress the invariant.
"""

from __future__ import annotations

import inspect

from evoliez.stages import ALL_STAGES
from evoliez.stages.s08b_mutant_boltz import MutantBoltzStage
from evoliez.stages.s09_nonmd_validation import NonMDValidationStage
from evoliez.stages.s10_md import MDStage


def test_all_stages_order_puts_s08b_after_s09_before_s10():
    names = [s.__name__ for s in ALL_STAGES]
    i_s09 = names.index("NonMDValidationStage")
    i_s08b = names.index("MutantBoltzStage")
    i_s10 = names.index("MDStage")
    assert i_s09 < i_s08b < i_s10, (
        f"stage order regressed: NonMDValidationStage={i_s09}, "
        f"MutantBoltzStage={i_s08b}, MDStage={i_s10}. The fix requires "
        "s09 < s08b < s10 so s08b runs on md_candidates."
    )


def test_s08b_reads_from_md_candidates_first():
    src = inspect.getsource(MutantBoltzStage.run)
    # The new flow prefers md_candidates (s09 output) over redock_candidates
    # so real Boltz lands on what s10 will actually use.
    assert 'md_candidates' in src
    # Pre-fix code read redock_candidates directly. The new code falls back
    # to it ONLY if md_candidates isn't set (e.g., MD disabled).
    assert 'ctx.get("md_candidates")' in src
    # The result writes BACK to md_candidates so s10 sees the updated
    # mut_complexes mapping (not redock_candidates, which would clobber
    # s09's full output that s11 still needs).
    assert 'ctx.put("md_candidates"' in src
    assert 'ctx.put("redock_candidates"' not in src
