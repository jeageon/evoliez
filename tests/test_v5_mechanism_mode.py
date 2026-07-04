"""ROADMAP_V5 step 3/4 — mechanism-mode policy + geometry source-of-truth resolver."""
from types import SimpleNamespace

import pytest

from evoliez.mechanism.mode import (
    LEGACY_BANNER, SRC_LEGACY, SRC_LEGACY_FROM_MECHANISM, SRC_MECHANISM_PENDING,
    SRC_NONE, STRICT_ENV, enforce_mechanism_policy, mechanism_mode,
    resolve_geometry_source,
)


def _cfg(mechanism=None):
    return SimpleNamespace(mechanism=mechanism)


# --- mode + strict enforcement --------------------------------------------------------

def test_absent_mechanism_is_legacy_mode():
    assert mechanism_mode(_cfg(None)) == "legacy"
    assert mechanism_mode(_cfg(object())) == "mechanism_spec"


def test_legacy_returns_banner_and_does_not_raise(monkeypatch):
    monkeypatch.delenv(STRICT_ENV, raising=False)
    mode, banner = enforce_mechanism_policy(_cfg(None))
    assert mode == "legacy"
    assert banner == LEGACY_BANNER


def test_strict_env_requires_mechanism(monkeypatch):
    monkeypatch.setenv(STRICT_ENV, "1")
    with pytest.raises(RuntimeError):
        enforce_mechanism_policy(_cfg(None))
    # a declared mechanism satisfies strict mode
    mode, banner = enforce_mechanism_policy(_cfg(object()))
    assert mode == "mechanism_spec" and banner is None


# --- geometry source of truth ---------------------------------------------------------

def test_geometry_source_priority():
    # mechanism + terms + legacy path running -> the legacy O->P ran, mechanism-declared
    assert resolve_geometry_source(_cfg(object()), ["term"], True) == SRC_LEGACY_FROM_MECHANISM
    # mechanism + terms but legacy NOT enabled -> declared, no consumer running yet
    assert resolve_geometry_source(_cfg(object()), ["term"], False) == SRC_MECHANISM_PENDING
    # no mechanism, legacy reactive_geometry enabled -> plain legacy
    assert resolve_geometry_source(_cfg(None), None, True) == SRC_LEGACY
    # nothing -> none
    assert resolve_geometry_source(_cfg(None), None, False) == SRC_NONE
    # mechanism declared but NO geometry_terms yet -> not counted as mechanism -> legacy/none
    assert resolve_geometry_source(_cfg(object()), None, True) == SRC_LEGACY
    assert resolve_geometry_source(_cfg(object()), [], False) == SRC_NONE
