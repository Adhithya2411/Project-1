"""Load OpenMP-linking optional dependencies in a safe order.

The problem
-----------
Two of AHRAG's optional extras each bundle their own OpenMP runtime:
``xgboost`` (the learned router) and ``torch``, pulled in by
``sentence-transformers`` (the neural embedding backend). On macOS, if torch's
``libomp`` is initialised **first** and xgboost's is loaded afterwards, the
process aborts — not a Python exception that can be caught, an immediate
abort with no traceback.

The failure is order-dependent and reproducible:

    import sentence_transformers; import xgboost   -> abort
    import xgboost; import sentence_transformers   -> fine

It is easy to hit by accident and very hard to diagnose from the symptom,
because the crash surfaces wherever the *second* library happens to be
imported. In AHRAG that is inside ``LearnedRouter.__init__``, several layers
below a script that only asked to compare some baselines: the engine builds its
index (loading torch), then the router loads its model (loading xgboost), and
the process dies with its stdout still buffered, so even the progress output is
lost. ``KMP_DUPLICATE_LIB_OK`` does not help.

The fix
-------
Import xgboost first, at package import time, before anything can reach torch.
It stays genuinely optional: if it is not installed, or if importing it fails
for any reason, that is recorded and ignored — the learned router already
degrades to the rule-based policy in that case.

This is a workaround for an environment interaction, not a design choice. If
the two libraries ever stop shipping separate OpenMP runtimes, deleting this
module and its import in ``ahrag/__init__.py`` is safe.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Whether xgboost was successfully pre-loaded. Reported by the health
#: endpoint so an operator can tell a genuinely-absent model apart from a
#: library that failed to load.
xgboost_preloaded = False


def preload() -> bool:
    """Import xgboost ahead of any OpenMP-linking library. Never raises."""
    global xgboost_preloaded
    try:
        import xgboost  # noqa: F401
    except ImportError:
        # Expected whenever the learned-router extra is not installed.
        return False
    except Exception as exc:  # pragma: no cover - broken install
        logger.warning(
            "xgboost is installed but failed to import (%s); the learned "
            "router will fall back to the rule-based policy.",
            exc,
        )
        return False
    xgboost_preloaded = True
    return True


preload()


__all__ = ["preload", "xgboost_preloaded"]
