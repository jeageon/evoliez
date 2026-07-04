"""Mechanism-mode policy (ROADMAP_V5 step 3/4): legacy vs. mechanism-spec, strict
enforcement, and the reaction-geometry source-of-truth resolver.

Design notes:
  * Strictness is an ENV var (``EVOLIEZ_STRICT_MECHANISM``), NOT a Config field. A new
    top-level Config field would change ``config_sha1`` (context.py computes it from
    ``config.model_dump(mode="json")`` incl. defaults) and purge every existing run on
    resume. The env var is purge-safe and mirrors ``EVOLIEZ_STRICT_CLAIMS``.
  * Absent mechanism = LEGACY mode: never hard-fails a run that predates mechanism specs;
    reports carry a banner and reaction-geometry claims stay capped.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

LEGACY_BANNER = (
    "MechanismSpec not declared: running the legacy geometry path. "
    "Reaction-geometry claims are capped at uncalibrated / hypothesis-grade."
)
STRICT_ENV = "EVOLIEZ_STRICT_MECHANISM"

# geometry-source labels (single source of truth for reports/provenance)
SRC_MECHANISM_PENDING = "mechanism_spec_pending"
SRC_LEGACY = "legacy_reactive_geometry"
SRC_NONE = "none"


def mechanism_mode(config) -> str:
    """'mechanism_spec' if a MechanismSpec is declared on the config, else 'legacy'."""
    return "mechanism_spec" if getattr(config, "mechanism", None) is not None else "legacy"


def _strict_enabled() -> bool:
    return os.environ.get(STRICT_ENV, "").lower() in ("1", "true", "yes", "on")


def enforce_mechanism_policy(config) -> Tuple[str, Optional[str]]:
    """Return ``(mode, banner)``. In legacy mode: raise if ``EVOLIEZ_STRICT_MECHANISM`` is
    set, otherwise return the legacy banner (never hard-fails legacy runs). In mechanism mode:
    ``(mode, None)``."""
    mode = mechanism_mode(config)
    if mode == "legacy":
        if _strict_enabled():
            raise RuntimeError(
                f"{STRICT_ENV}=1 but no MechanismSpec is declared. Add a `mechanism:` block "
                "(reaction.class + reaction_state), or unset the env var to run legacy.")
        return mode, LEGACY_BANNER
    return mode, None


def resolve_geometry_source(config, geometry_terms, nac_enabled: bool) -> str:
    """Which reaction-geometry source s10 uses, in priority order (ROADMAP_V5 §0.5.3 #5):
    ``mechanism.geometry_terms`` > legacy ``reactive_geometry`` > none.

    The mechanism branch is a consumer HOOK: the trajectory NAC engine still consumes the
    legacy ``ReactiveSpec``, so until the adenylation template + ReactiveSpec/GeometryTerm
    bridge land (V5-3/V5-6) a declared mechanism is reported as ``mechanism_spec_pending`` —
    honest about intent vs. what actually ran (never claims a consumer that is not wired)."""
    if getattr(config, "mechanism", None) is not None and geometry_terms:
        return SRC_MECHANISM_PENDING
    if nac_enabled:
        return SRC_LEGACY
    return SRC_NONE
