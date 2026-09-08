"""AHRAG — Adaptive Hybrid Retrieval-Augmented Generation.

A local, offline-capable research prototype of a *governance-aware* adaptive
router for enterprise RAG. The organising claim the code is built to
demonstrate (and that the evaluation harness is built to test) is:

    Route selection is constrained by authorised source scope and optimised for
    evidence sufficiency, freshness, source authority, estimated latency, and
    estimated cost -- not merely query complexity.

Nothing in this package should be read as a claim of novelty, patentability, or
state-of-the-art performance. See ``RESEARCH_LIMITATIONS.md`` and
``INVENTION_DISCLOSURE.md`` at the repository root.
"""

# Must run before anything can import torch. ``xgboost`` and ``torch`` each
# bundle an OpenMP runtime, and on macOS loading them in the wrong order aborts
# the process outright. See ahrag/_openmp.py for the full explanation.
from . import _openmp as _openmp  # noqa: E402,F401  (import-order sensitive)

__version__ = "0.3.0"

__all__ = ["__version__"]
