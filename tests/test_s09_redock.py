"""Audit P1 #8 + #2-redock — s09 redocking fixes.

#8       redock_with() must use the CALLING stage's backend, not always
         s05_docking's, so `--stage-backend s09_nonmd=real|mock` is honoured.
#2-redock s09 redocks the real per-mutant Boltz structure from s08b (top-N)
         instead of the WT-coords proxy; the rest fall back to the proxy.
"""

from pathlib import Path

from evoliez.config import Backend, load_config
from evoliez.context import RunContext
from evoliez.pipeline import Pipeline
from evoliez.stages.s05_docking import redock_with
from evoliez.types import LigandAtom, ProteinStructure, Residue
from evoliez.utils.seeds import seed_everything

ROOT = Path(__file__).resolve().parents[1]


class _Ctx:
    """Minimal stand-in: redock_with only reads .config + .dry_run."""
    def __init__(self, cfg):
        self.config = cfg
        self.dry_run = False


def _struct():
    return ProteinStructure(sequence="AG",
                            residues=[Residue(index=1, aa="A", ca=(0.0, 0.0, 0.0))])


def test_redock_with_uses_calling_stage_backend(tmp_path):
    # global real, but s09 overridden to mock. The s09 redock must take the
    # MOCK path (deterministic pose, no tools). Before the fix it hardcoded
    # backend_for("s05_docking")=real and would hit the real tool path.
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {"project.output_dir": str(tmp_path / "run"),
         "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
         "backend": "real", "backends": {"s09_nonmd": "mock"}},
    )
    assert cfg.backend_for("s09_nonmd") is Backend.mock
    assert cfg.backend_for("s05_docking") is Backend.real
    ref = [LigandAtom(id="C1", element="C", coord=(0.0, 0.0, 0.0))]
    pose = redock_with("vina", _Ctx(cfg), "c", _struct(), ref,
                       cfg.validation.redocking, tmp_path / "rd", 0.2, "CCO",
                       stage_name="s09_nonmd")
    assert pose is not None and pose.score is not None     # mock path, no raise


def test_s09_redocks_mutant_boltz_structure(tmp_path):
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {"project.output_dir": str(tmp_path / "run"),
         "input.target_fasta": str(ROOT / "examples" / "fdh" / "target.fasta"),
         "mutation_generation": {"methods": ["chemistry_rules"],
                                 "max_candidates": 20},
         "reranking": {"top_for_redocking": 6, "top_for_md": 3,
                       "mutant_boltz_enabled": True, "mutant_boltz_top_n": 4},
         "gnn": {"build_dataset": False}},
    )
    seed_everything(cfg.seed)
    ctx = RunContext(cfg, allow_small_disk=True).setup()
    Pipeline().run(ctx, to_stage="s09_nonmd")

    assert len(ctx.get("mutant_complexes", {}) or {}) == 4   # s08b top-N
    cands = ctx.get("md_candidates", []) or []
    srcs = [c.details.get("redock_structure_source") for c in cands]
    # every candidate is tagged, the s08b top-N used the real mutant structure,
    # and the rest fell back to the WT proxy.
    assert all(s in ("mutant_boltz", "wt_proxy") for s in srcs)
    assert srcs.count("mutant_boltz") == 4
    assert "wt_proxy" in srcs
