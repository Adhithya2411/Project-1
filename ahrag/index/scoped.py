"""Scope-Pure Index Specialisation (SPIS): retrieval statistics per ACL class.

Design position
---------------
``index/lattice.py`` argues that the corpus is a lattice of subcorpora and that
fitting retrieval statistics globally both misweights ranking and opens an
information-flow channel. This module fixes both by fitting a separate index per
ACL equivalence class.

For class *C*, the specialised bundle is built from *C*'s chunks and nothing
else. Consequently:

* BM25 IDF and average document length describe the pool actually searched.
* The TF-IDF vocabulary and LSA basis are fitted on that pool.
* Optionally, the confidence calibration constant is estimated from that
  pool too, so the router's ``min_probe_for_answering`` threshold could mean
  the same thing for a contractor reading 35 chunks as for an engineer reading
  54. That part is off by default: the threshold was tuned against a fixed
  constant, so rescaling confidence without re-tuning it jointly moves routes
  for reasons unrelated to retrieval quality. See ``_calibrate``.

And the property that motivates the design: **every parameter of C's index is a
function of C's chunks alone.** Mutating, adding, or deleting any chunk outside
*C* cannot change any score, any ranking, or any route decision for a principal
in *C*. That is non-interference at the index level, and
``tests/test_noninterference.py`` asserts it by construction rather than
assuming it.

Cost and its bounds
-------------------
Specialisation is per *class*, not per *user*: the class count is bounded by the
distinct role-set combinations on documents, not by headcount. Bundles are built
lazily on first use and cached under an LRU cap, so an adversarially wide lattice
degrades to the global index rather than exhausting memory. A class that cannot
support a dense fit — too few chunks for a rank-2 SVD — keeps its specialised
sparse index and borrows the global dense one; that borrowing is recorded in
:attr:`ScopedIndex.dense_is_pure` rather than quietly assumed away, because it
is exactly the case where the purity claim does not hold.

This is opt-in. ``Settings.index_specialisation`` defaults to ``False`` so the
unspecialised system remains the reference point for the ablation.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..config import Settings
from ..models import AuthorisedScope, Chunk
from .embeddings import tokenize
from .lattice import ACLClass, EMPTY_SIGNATURE, ScopeLattice
from .store import IndexBundle

logger = logging.getLogger(__name__)

# Saturation constant used when a class is too small to calibrate from its own
# content. Matches the historic hard-coded value in ``retrieval/pipeline.py``,
# so an uncalibrated class behaves exactly as the unspecialised system did.
DEFAULT_SATURATION = 6.0

# Pseudo-queries drawn per class when estimating the saturation constant, and
# how many leading tokens of a chunk to use as one. Both are deliberately small:
# the estimate only needs to fix a scale, not to be precise.
_CALIBRATION_SAMPLES = 24
_CALIBRATION_QUERY_TOKENS = 12


@dataclass
class ScopedIndex:
    """A retrieval view specialised to one ACL class.

    Exposes the same four members ``retrieval/pipeline.py`` uses on
    ``IndexBundle`` — ``sparse``, ``vectors``, ``encode_query``, ``get`` — so it
    substitutes for one without touching any call site.
    """

    signature: str
    bundle: IndexBundle
    chunk_count: int
    saturation: float
    dense_is_pure: bool
    specialised: bool
    global_bundle: IndexBundle

    @property
    def sparse(self):  # noqa: ANN201 - mirrors IndexBundle.sparse
        """The BM25 index for this class."""
        return self.bundle.sparse

    @property
    def vectors(self):  # noqa: ANN201 - mirrors IndexBundle.vectors
        """The vector store for this class, or the global one if borrowed."""
        return self.bundle.vectors if self.dense_is_pure else self.global_bundle.vectors

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a query in this class's latent space.

        Raises:
            RuntimeError: If the underlying index has not been built.
        """
        source = self.bundle if self.dense_is_pure else self.global_bundle
        return source.encode_query(query)

    def get(self, chunk_id: str) -> Chunk | None:
        """Resolve a chunk by ID.

        Falls back to the global snapshot. This is a metadata lookup for a
        chunk that a ranking function already selected, not a ranking function
        itself, so it carries no information about unauthorised material.
        """
        return self.bundle.get(chunk_id) or self.global_bundle.get(chunk_id)


class ScopeSpecialisedIndex:
    """Builds and caches one specialised index per ACL equivalence class."""

    def __init__(
        self,
        settings: Settings,
        global_bundle: IndexBundle,
        lattice: ScopeLattice | None = None,
    ) -> None:
        """Wire the manager to its settings and the global fallback bundle.

        Args:
            settings: Supplies ``index_specialisation``,
                ``specialisation_lambda``, and ``specialisation_max_classes``.
            global_bundle: The unspecialised index, used as the fallback and as
                the shrinkage reference.
            lattice: Optional pre-built lattice. One is derived from the global
                bundle's snapshot when omitted.
        """
        self.settings = settings
        self.global_bundle = global_bundle
        self.lattice = lattice or ScopeLattice(global_bundle.chunks)
        self._cache: OrderedDict[str, ScopedIndex] = OrderedDict()
        self._global_idf: dict[str, float] | None = None
        self._builds = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._dense_fallbacks = 0

    # -- lifecycle ----------------------------------------------------------

    def invalidate(self) -> None:
        """Drop every cached bundle. Called after re-ingestion."""
        self._cache.clear()
        self._global_idf = None
        self.lattice.rebuild(self.global_bundle.chunks)

    @property
    def enabled(self) -> bool:
        """True when specialisation is switched on and the corpus is non-empty."""
        return bool(self.settings.index_specialisation) and bool(
            self.global_bundle.chunks
        )

    # -- hot path -----------------------------------------------------------

    def for_scope(self, scope: AuthorisedScope) -> ScopedIndex:
        """Return the specialised index for ``scope``.

        Falls back to a non-specialised view — the global bundle, wrapped so the
        caller's interface does not change — when specialisation is disabled,
        the scope is empty, or the cache cap has been reached.

        Args:
            scope: The authorised scope for this query.

        Returns:
            A :class:`ScopedIndex`. ``specialised`` records which case applied.
        """
        if not self.enabled or not scope.allowed_chunk_ids:
            return self._global_view(EMPTY_SIGNATURE if not scope.allowed_chunk_ids else "global")

        acl_class = self.lattice.classify(scope)
        cached = self._cache.get(acl_class.signature)
        if cached is not None:
            self._hits += 1
            self._cache.move_to_end(acl_class.signature)
            return cached

        self._misses += 1
        cap = max(1, int(self.settings.specialisation_max_classes))
        if len(self._cache) >= cap:
            # Evict the least-recently-used class rather than refusing to
            # specialise: a bounded working set is the common case, and an
            # unbounded lattice should degrade in latency, not in correctness.
            evicted, _ = self._cache.popitem(last=False)
            self._evictions += 1
            logger.info("Evicted specialised index for class %s", evicted[:12])

        built = self._build(acl_class)
        self._cache[acl_class.signature] = built
        return built

    # -- construction -------------------------------------------------------

    def _build(self, acl_class: ACLClass) -> ScopedIndex:
        """Fit a specialised bundle for one class.

        Every input is drawn from ``acl_class.chunk_ids``; that restriction is
        the entire purity argument, so it is enforced here at the single point
        where the subcorpus is selected.
        """
        subset = [c for c in self.global_bundle.chunks if c.chunk_id in acl_class.chunk_ids]
        if not subset:
            return self._global_view(acl_class.signature)

        bundle = IndexBundle(self.settings)
        dense_is_pure = True
        try:
            bundle.build(subset)
        except (ValueError, RuntimeError) as exc:
            # The sparse index is built first inside ``IndexBundle.build`` and
            # survives a dense failure, so keep it and borrow the global dense
            # side. Recorded, not hidden: this class is only sparsely pure.
            logger.warning(
                "Dense fit failed for ACL class %s (%d chunks): %s; "
                "borrowing the global dense index.",
                acl_class.signature[:12],
                len(subset),
                exc,
            )
            dense_is_pure = False
            self._dense_fallbacks += 1

        if bundle.embedder is None or len(bundle.vectors) == 0:
            dense_is_pure = False
            self._dense_fallbacks += 1

        lam = float(self.settings.specialisation_lambda)
        if lam > 0.0:
            bundle.sparse.shrink_idf_towards(self._global_idf_table(), lam)

        saturation = self._calibrate(bundle, subset)
        self._builds += 1
        logger.info(
            "Built specialised index for ACL class %s: %d chunks, "
            "saturation=%.3f, lambda=%.2f, dense_pure=%s",
            acl_class.signature[:12],
            len(subset),
            saturation,
            lam,
            dense_is_pure,
        )
        return ScopedIndex(
            signature=acl_class.signature,
            bundle=bundle,
            chunk_count=len(subset),
            saturation=saturation,
            dense_is_pure=dense_is_pure,
            specialised=True,
            global_bundle=self.global_bundle,
        )

    def _global_view(self, signature: str) -> ScopedIndex:
        """Wrap the global bundle so callers get one uniform interface."""
        return ScopedIndex(
            signature=signature,
            bundle=self.global_bundle,
            chunk_count=len(self.global_bundle.chunks),
            saturation=DEFAULT_SATURATION,
            dense_is_pure=True,
            specialised=False,
            global_bundle=self.global_bundle,
        )

    def _global_idf_table(self) -> dict[str, float]:
        """Cache and return the global IDF table used as the shrinkage target."""
        if self._global_idf is None:
            self._global_idf = self.global_bundle.sparse.idf_snapshot()
        return self._global_idf

    def _calibrate(self, bundle: IndexBundle, subset: Sequence[Chunk]) -> float:
        """Estimate the BM25 saturation constant for one class.

        ``sparse_confidence`` in ``retrieval/pipeline.py`` maps a raw BM25 score
        to ``score / (score + c)``. With a single global ``c`` the same
        confidence value means different things to differently-scoped
        principals, and the router then compares all of them against one hard
        threshold. Estimating ``c`` from the class's own content is what would
        make that threshold principal-invariant.

        Two details matter, both learned the hard way:

        * A chunk queried with its own leading tokens retrieves *itself* almost
          perfectly, so self-matches must be excluded. Including them put the
          estimate around 25 on the seed corpus against a historic constant of
          6.0, which deflated every confidence and moved routes.
        * Even excluding self-matches, chunk-derived queries are denser than
          natural questions, so a low quantile of the match distribution — not
          its median — is the defensible saturation point.

        Returns :data:`DEFAULT_SATURATION` unless
        ``Settings.specialisation_calibrate_confidence`` is set, because
        rescaling confidence without re-tuning ``min_probe_for_answering``
        changes routing for reasons unrelated to retrieval quality.

        Args:
            bundle: The freshly-built specialised bundle.
            subset: The class's chunks.

        Returns:
            A positive saturation constant, or :data:`DEFAULT_SATURATION` when
            calibration is disabled or the class cannot support an estimate.
        """
        if not self.settings.specialisation_calibrate_confidence:
            return DEFAULT_SATURATION
        if not subset or len(bundle.sparse) == 0:
            return DEFAULT_SATURATION

        allowed = [c.chunk_id for c in subset]
        # Evenly spaced rather than randomly sampled, so the constant is
        # reproducible without threading a seed through the index layer.
        step = max(1, len(subset) // _CALIBRATION_SAMPLES)
        scores: list[float] = []
        for chunk in subset[::step][:_CALIBRATION_SAMPLES]:
            tokens = tokenize(chunk.text)[:_CALIBRATION_QUERY_TOKENS]
            if not tokens:
                continue
            hits = bundle.sparse.search(" ".join(tokens), 3, allowed)
            # Best match that is *not* the source chunk.
            other = [raw for cid, raw, _ in hits if cid != chunk.chunk_id]
            if other:
                scores.append(other[0])

        if not scores:
            return DEFAULT_SATURATION
        quantile = min(max(float(self.settings.specialisation_confidence_quantile), 0.0), 1.0)
        estimate = float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))
        # A degenerate estimate would make confidence either always 1 or always 0.
        if not np.isfinite(estimate) or estimate <= 1e-6:
            return DEFAULT_SATURATION
        return round(estimate, 6)

    # -- reporting ----------------------------------------------------------

    def stats(self) -> dict[str, object]:
        """Return cache and purity telemetry for the health endpoint and harness."""
        return {
            "enabled": self.enabled,
            "lambda": float(self.settings.specialisation_lambda),
            "max_classes": int(self.settings.specialisation_max_classes),
            "cached_classes": len(self._cache),
            "builds": self._builds,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "evictions": self._evictions,
            "dense_fallbacks": self._dense_fallbacks,
            "fully_pure": (
                bool(self.settings.index_specialisation)
                and float(self.settings.specialisation_lambda) == 0.0
                and self._dense_fallbacks == 0
            ),
            "class_sizes": {
                sig[:12]: view.chunk_count for sig, view in self._cache.items()
            },
            "saturations": {
                sig[:12]: view.saturation for sig, view in self._cache.items()
            },
        }


__all__ = ["DEFAULT_SATURATION", "ScopeSpecialisedIndex", "ScopedIndex"]
