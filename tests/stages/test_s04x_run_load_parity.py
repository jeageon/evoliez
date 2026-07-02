from pathlib import Path

from evoliez.stages.s04x_reference_ensemble import CTX_KEYS, ReferenceEnsembleStage


class FakePaths:
    def __init__(self, root: Path):
        self.root = root


class FakeCtx:
    def __init__(self, root: Path):
        self.paths = FakePaths(root)
        self.artifacts = {}

    def put(self, key, value):
        self.artifacts[key] = value

    def get(self, key, default=None):
        return self.artifacts.get(key, default)


def test_s04x_run_load_parity(tmp_path):
    ctx = FakeCtx(tmp_path)
    stage = ReferenceEnsembleStage()
    stage.run(ctx)
    first = {key: ctx.get(key) for key in CTX_KEYS}
    assert all(first[key] is not None for key in CTX_KEYS)

    ctx2 = FakeCtx(tmp_path)
    assert stage.load(ctx2) is True
    second = {key: ctx2.get(key) for key in CTX_KEYS}
    assert second == first
