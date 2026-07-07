"""Tool-validation metrics: pose RMSD + PLIF recovery (expert plan)."""

from evoliez.adapters.plip import IFPContact
from evoliez.config import Backend, load_config
from evoliez.ml.pose_validity import plif_recovery, pose_rmsd
from evoliez.types import LigandAtom
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pose_rmsd():
    ref = [LigandAtom(id="C0", element="C", coord=(0, 0, 0)),
           LigandAtom(id="O1", element="O", coord=(3, 0, 0))]
    assert pose_rmsd(ref, ref) == 0.0
    shifted = [LigandAtom(id="C0", element="C", coord=(2, 0, 0)),
               LigandAtom(id="O1", element="O", coord=(5, 0, 0))]
    assert pose_rmsd(shifted, ref) == 2.0          # uniform 2 Å shift


def test_plif_recovery():
    ref = [IFPContact(10, "O1", "hbond", 3.0),
           IFPContact(20, "P1", "salt_bridge", 3.5),
           IFPContact(30, "C2", "hydrophobic", 4.0)]
    pred = [IFPContact(10, "O1", "hbond", 3.1),       # match
            IFPContact(20, "P1", "salt_bridge", 3.4),  # match
            IFPContact(99, "C9", "hydrophobic", 4.2)]  # miss
    assert plif_recovery(pred, ref) == round(2 / 3, 4)
    assert plif_recovery([], ref) == 0.0


def test_stage_backend_override_semantics():
    # the CLI --stage-backend s04_complex=real maps to backends={...}
    cfg = load_config(
        ROOT / "configs" / "example_fdh_nadp.yaml",
        {"backend": "mock", "backends": {"s04_complex": "real"}},
    )
    assert cfg.backend_for("s04_complex") is Backend.real
    assert cfg.backend_for("s10_md") is Backend.mock
