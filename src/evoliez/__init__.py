"""EvoLigand-Enzyme Engineer.

Ligand-aware, enzyme-family-specific protein engineering pipeline with
molecular-dynamics validation. See docs/ARCHITECTURE.md for the module map.
"""

# --- shared-server thread-pool clamp ----------------------------------------
# MUST run before numpy / OpenBLAS / MKL / numexpr / xgboost are imported
# anywhere in the process (this package's __init__ is the first thing the
# `evoliez` CLI imports, and it pulls in nothing heavy, so this is the
# earliest safe point).
#
# The lab box is a SHARED 48-core server (SERVER_RUNBOOK constraint #2).
# Left unbounded, BLAS + numexpr + xgboost each spawn 16-48 threads *inside*
# the per-pose / per-candidate loops of s06b/s08; on a loaded box this
# oversubscribes the CPU and thrashes (observed: s06b 355 s, s08 206 s of
# pure context-switching at load avg ~14, for work that is milliseconds of
# actual computation). Bounding the pools removes the thrash AND keeps us a
# good neighbour on the shared machine. Users can raise it explicitly with
# EVOLIEZ_NUM_THREADS, or by pre-setting any individual *_NUM_THREADS var.
import os as _os


def _limit_thread_pools() -> None:
    try:
        want = int(_os.environ.get("EVOLIEZ_NUM_THREADS", "") or 0)
    except ValueError:
        want = 0
    if want <= 0:
        want = min(4, _os.cpu_count() or 4)
    for _var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        _os.environ.setdefault(_var, str(want))  # respect explicit overrides
    _os.environ.setdefault("EVOLIEZ_NUM_THREADS", str(want))


_limit_thread_pools()

__version__ = "0.1.0"

__all__ = ["__version__"]
