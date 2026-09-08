"""The retrieval engine: executes one route against the authorised scope.

Every public method takes an :class:`AuthorisedScope` and passes its chunk-ID
allow-list straight through to the index layer. There is no code path in this
module that searches an unrestricted pool.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from ..config import RouterConfig, Settings
from ..governance.acl import AccessControl
from ..index.store import IndexBundle
from ..models import AuthorisedScope, Chunk, Route, RetrievalTrace, ScoredChunk, User
from ..routing.features import ProbeSignals
from .decompose import decompose_query, iteration_weights
from .fusion import rank_positions, reciprocal_rank_fusion
from .rerank import Reranker

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Runs sparse, dense, hybrid, and iterative retrieval under ACL scope."""

    def __init__(
        self,
        index: IndexBundle,
        acl: AccessControl,
        reranker: Reranker,
        config: RouterConfig,
        settings: Settings,
    ) -> None:
        """Wire the engine to its indexes, governance layer, and policy."""
        self.index = index
        self.acl = acl
        self.reranker = reranker
        self.config = config
        self.settings = settings

    # -- probe -------------------------------------------------------------

    def probe(self, query: str, scope: AuthorisedScope) -> ProbeSignals:
        """Run the cheap first-pass retrieval that informs the router.

        Costs one shallow BM25 lookup and one shallow vector lookup, both
        already restricted to ``scope``. The result tells the router whether the
        *authorised* corpus contains anything relevant — the signal that lets it
        distinguish "hard question" from "question this user has no evidence for".

        Returns:
            :class:`ProbeSignals`. On any backend error the probe returns zeroed
            confidence rather than raising, which biases the router toward
            abstention. Failing closed is the correct default here.
        """
        started = time.perf_counter()
        top_k = self.config.retrieval.probe_top_k
        allowed = scope.allowed_chunk_ids
        if not allowed:
            return ProbeSignals(elapsed_ms=0.0)

        try:
            sparse_hits = self.index.sparse.search(query, top_k, allowed)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Sparse probe failed: %s", exc)
            sparse_hits = []

        try:
            query_vector = self.index.encode_query(query)
            dense_hits = self.index.vectors.search(query_vector, top_k, allowed)
        except (ValueError, RuntimeError) as exc:
            logger.warning("Dense probe failed: %s", exc)
            dense_hits = []

        sparse_ids = [cid for cid, _, _ in sparse_hits]
        dense_ids = [cid for cid, _ in dense_hits]

        # Sparse confidence blends the best absolute BM25 score (saturating,
        # so it is comparable across corpora) with the margin over the runner-up.
        # A high top score with no margin means many chunks match equally, which
        # is weak evidence for an exact-match route.
        sparse_confidence = 0.0
        if sparse_hits:
            best_raw = sparse_hits[0][1]
            saturated = best_raw / (best_raw + 6.0)
            margin = 1.0
            if len(sparse_hits) > 1 and sparse_hits[0][1] > 0:
                margin = 1.0 - (sparse_hits[1][1] / sparse_hits[0][1])
            sparse_confidence = min(1.0, saturated * (0.72 + 0.28 * margin))

        dense_confidence = 0.0
        if dense_hits:
            # Cosine similarity is already 0..1-ish for normalised vectors, but
            # can be negative; clamp and rescale so weak matches read as weak.
            best = max(0.0, dense_hits[0][1])
            dense_confidence = min(1.0, best)

        overlap = set(sparse_ids) & set(dense_ids)
        union = set(sparse_ids) | set(dense_ids)
        agreement = len(overlap) / len(union) if union else 0.0

        families = [
            chunk.policy_family
            for chunk in self.acl.resolve_authorised(union, _scope_user(scope))
            if chunk.policy_family
        ]
        conflict_likelihood = _conflict_likelihood(families, self.acl, union, scope)

        return ProbeSignals(
            sparse_confidence=round(sparse_confidence, 4),
            dense_confidence=round(dense_confidence, 4),
            agreement=round(agreement, 4),
            sparse_ids=sparse_ids,
            dense_ids=dense_ids,
            families_seen=sorted(set(families)),
            conflict_likelihood=conflict_likelihood,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    # -- per-route retrieval ------------------------------------------------

    def retrieve(
        self, route: Route, query: str, user: User, scope: AuthorisedScope
    ) -> tuple[list[ScoredChunk], RetrievalTrace]:
        """Execute ``route`` and return reranked candidates plus telemetry.

        Args:
            route: The route chosen by the router.
            query: Normalised query text.
            user: The requesting user, used for the defence-in-depth ACL re-check.
            scope: The authorised scope; its allow-list bounds every index call.

        Returns:
            Reranked, normalised, ACL-verified candidates and a
            :class:`RetrievalTrace`.

        Raises:
            UnauthorisedChunkError: If a chunk outside scope somehow surfaced.
                Unreachable in correct operation; see ``governance/acl.py``.
        """
        trace = RetrievalTrace(
            route=route,
            acl_scoped_pool=len(scope.allowed_chunk_ids),
            acl_withheld=scope.withheld_count,
        )
        if route is Route.R0 or not scope.allowed_chunk_ids:
            return [], trace

        if route is Route.R1:
            candidates = self._sparse_only(query, scope, trace)
        elif route is Route.R2:
            candidates = self._dense_only(query, scope, trace)
        elif route is Route.R3:
            candidates = self._hybrid(query, scope, trace)
        else:
            candidates = self._iterative_hybrid(query, scope, trace)

        # Defence in depth: verify before reranking, which is the first stage
        # where chunk *content* influences ordering.
        self.acl.assert_authorised(candidates, user, stage="pre-rerank")

        started = time.perf_counter()
        reranked = self.reranker.rerank(
            query, candidates, self.config.retrieval.rerank_top_k
        )
        trace.timings_ms["rerank"] = round((time.perf_counter() - started) * 1000, 3)
        trace.reranked_candidates = len(reranked)

        self.acl.assert_authorised(reranked, user, stage="post-rerank")
        # Scores are returned on the reranker's *absolute* 0..1 scale, never
        # max-normalised. Max-normalisation would force the best chunk of every
        # query to 1.0, which would make the evidence-sufficiency relevance
        # floor unfalsifiable — a query with no good answer would look identical
        # to one with a perfect answer.
        return reranked, trace

    # -- route implementations ---------------------------------------------

    def _sparse_only(
        self, query: str, scope: AuthorisedScope, trace: RetrievalTrace
    ) -> list[ScoredChunk]:
        """R1: BM25 over the authorised pool."""
        started = time.perf_counter()
        hits = self.index.sparse.search(
            query, self.config.retrieval.candidate_top_k, scope.allowed_chunk_ids
        )
        trace.timings_ms["sparse"] = round((time.perf_counter() - started) * 1000, 3)
        trace.sparse_candidates = len(hits)
        trace.subqueries = [query]

        out: list[ScoredChunk] = []
        for rank, (chunk_id, _raw, normalised) in enumerate(hits, start=1):
            chunk = self.index.get(chunk_id)
            if chunk is None:
                continue
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=normalised,
                    sparse_rank=rank,
                    retriever="bm25",
                )
            )
        return out

    def _dense_only(
        self, query: str, scope: AuthorisedScope, trace: RetrievalTrace
    ) -> list[ScoredChunk]:
        """R2: vector search over the authorised pool."""
        started = time.perf_counter()
        try:
            query_vector = self.index.encode_query(query)
            hits = self.index.vectors.search(
                query_vector, self.config.retrieval.candidate_top_k, scope.allowed_chunk_ids
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("Dense retrieval failed (%s); returning no candidates.", exc)
            hits = []
        trace.timings_ms["dense"] = round((time.perf_counter() - started) * 1000, 3)
        trace.dense_candidates = len(hits)
        trace.subqueries = [query]

        out: list[ScoredChunk] = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            chunk = self.index.get(chunk_id)
            if chunk is None:
                continue
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=max(0.0, score),
                    dense_rank=rank,
                    retriever="dense",
                )
            )
        return out

    def _hybrid(
        self, query: str, scope: AuthorisedScope, trace: RetrievalTrace
    ) -> list[ScoredChunk]:
        """R3: sparse + dense, fused with RRF."""
        sparse = self._sparse_only(query, scope, trace)
        dense = self._dense_only(query, scope, trace)
        trace.subqueries = [query]

        started = time.perf_counter()
        fused = self._fuse(
            [[s.chunk_id for s in sparse], [d.chunk_id for d in dense]],
            list(sparse) + list(dense),
            query=query,
        )
        trace.timings_ms["fusion"] = round((time.perf_counter() - started) * 1000, 3)
        trace.fused_candidates = len(fused)
        return fused

    def _iterative_hybrid(
        self, query: str, scope: AuthorisedScope, trace: RetrievalTrace
    ) -> list[ScoredChunk]:
        """R4: decompose, run hybrid retrieval per sub-query, fuse across all."""
        params = self.config.retrieval
        subqueries = decompose_query(query, max_subqueries=params.r4_max_iterations)
        trace.subqueries = subqueries
        trace.iterations = len(subqueries)

        rankings: list[list[str]] = []
        all_scored: list[ScoredChunk] = []
        seen_by_id: dict[str, ScoredChunk] = {}

        for index, subquery in enumerate(subqueries):
            started = time.perf_counter()
            sub_trace = RetrievalTrace(route=Route.R4)
            sparse = self._sparse_only(subquery, scope, sub_trace)[: params.r4_subquery_top_k]
            dense = self._dense_only(subquery, scope, sub_trace)[: params.r4_subquery_top_k]
            elapsed = round((time.perf_counter() - started) * 1000, 3)

            for item in list(sparse) + list(dense):
                tagged = item.model_copy(update={"subquery": subquery})
                # Keep the first (highest-priority) sighting of each chunk so
                # the displayed subquery attribution matches the strongest hit.
                seen_by_id.setdefault(tagged.chunk_id, tagged)
                all_scored.append(tagged)

            rankings.append([s.chunk_id for s in sparse])
            rankings.append([d.chunk_id for d in dense])

            trace.per_iteration.append(
                {
                    "iteration": index + 1,
                    "subquery": subquery,
                    "sparse_hits": len(sparse),
                    "dense_hits": len(dense),
                    "elapsed_ms": elapsed,
                }
            )
            trace.sparse_candidates += len(sparse)
            trace.dense_candidates += len(dense)

        started = time.perf_counter()
        # Each iteration contributes two rankings (sparse, dense) at the same
        # weight, decayed by iteration depth.
        per_iteration = iteration_weights(len(subqueries))
        weights = [w for w in per_iteration for _ in range(2)]
        fused = self._fuse(rankings, list(seen_by_id.values()), query=query, weights=weights)
        trace.timings_ms["fusion"] = round((time.perf_counter() - started) * 1000, 3)
        trace.fused_candidates = len(fused)
        return fused

    def _fuse(
        self,
        rankings: Sequence[Sequence[str]],
        sources: Sequence[ScoredChunk],
        query: str,
        weights: Sequence[float] | None = None,
    ) -> list[ScoredChunk]:
        """Apply RRF and rebuild :class:`ScoredChunk` objects with rank provenance."""
        fused = reciprocal_rank_fusion(
            rankings, k=self.config.retrieval.rrf_k, weights=weights
        )
        by_id: dict[str, ScoredChunk] = {}
        for item in sources:
            by_id.setdefault(item.chunk_id, item)

        sparse_positions = rank_positions(
            [s.chunk_id for s in sources if s.retriever == "bm25"]
        )
        dense_positions = rank_positions(
            [s.chunk_id for s in sources if s.retriever == "dense"]
        )

        out: list[ScoredChunk] = []
        for chunk_id, fused_score in fused[: self.config.retrieval.candidate_top_k]:
            base = by_id.get(chunk_id)
            chunk = base.chunk if base else self.index.get(chunk_id)
            if chunk is None:
                continue
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=fused_score,
                    fused_score=fused_score,
                    sparse_rank=sparse_positions.get(chunk_id),
                    dense_rank=dense_positions.get(chunk_id),
                    retriever="hybrid-rrf",
                    subquery=base.subquery if base else None,
                )
            )
        return out


def _scope_user(scope: AuthorisedScope) -> User:
    """Rebuild a minimal user from a scope, for authorisation re-checks."""
    return User(
        user_id=scope.user_id, display_name=scope.user_id, roles=list(scope.roles)
    )


def _conflict_likelihood(
    families: Sequence[str],
    acl: AccessControl,
    candidate_ids: set[str],
    scope: AuthorisedScope,
) -> float:
    """Estimate the chance the probe touched more than one version of a topic.

    Returns 1.0 when two chunks from the *same* policy family but *different*
    parent documents both surfaced — the signature of a live-versus-retired
    policy pair. This runs on authorised chunks only, so a restricted user never
    learns that a conflicting version they cannot read exists.
    """
    chunks: list[Chunk] = acl.resolve_authorised(candidate_ids, _scope_user(scope))
    by_family: dict[str, set[str]] = {}
    for chunk in chunks:
        if chunk.policy_family:
            by_family.setdefault(chunk.policy_family, set()).add(chunk.doc_id)
    if any(len(docs) > 1 for docs in by_family.values()):
        return 1.0
    return 0.0
