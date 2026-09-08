"""Access control: the hard pre-filter that bounds everything downstream.

Design position
---------------
ACL is **not** a term in the routing utility function and **not** a
post-retrieval filter. It is a set-membership constraint computed before the
router runs, which produces an ``AuthorisedScope``; every retriever is then
physically restricted to that set of chunk IDs.

The practical consequence is that an unauthorised chunk cannot influence a
route choice, a rerank ordering, an evidence pack, a generated sentence, a
displayed citation, or a logged evidence body — because it was never a
candidate. Cost-optimal routing therefore cannot trade governance for latency:
the trade is not expressible.

Because "never" is a strong claim, this module also exposes ``assert_authorised``,
a defence-in-depth re-check called at every stage boundary in
``ahrag/pipeline.py``. The re-check should be unreachable; the tests assert that
it *would* fire if the pre-filter were bypassed.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

from ..models import AuthorisedScope, Chunk, ScoredChunk, User

logger = logging.getLogger(__name__)


class UnauthorisedChunkError(PermissionError):
    """Raised when a chunk outside the authorised scope reaches a later stage.

    Reaching this exception indicates a bug in the pipeline, not a user error.
    It is deliberately fatal rather than a filter-and-continue, so that a
    governance defect surfaces in tests instead of silently degrading.
    """


class AccessControl:
    """Computes and enforces per-user authorised scope over the corpus."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        """Bind to the current corpus snapshot.

        Args:
            chunks: All chunks in the corpus. The snapshot is held by
                reference-copy; call :meth:`rebuild` after re-ingestion.
        """
        self._chunks: list[Chunk] = list(chunks)
        self._by_id: dict[str, Chunk] = {c.chunk_id: c for c in self._chunks}

    def rebuild(self, chunks: Sequence[Chunk]) -> None:
        """Replace the corpus snapshot after ingestion."""
        self._chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self._chunks}

    @property
    def total_chunks(self) -> int:
        """Number of chunks in the corpus, authorised or not."""
        return len(self._chunks)

    def scope_for(self, user: User) -> AuthorisedScope:
        """Compute the authorised scope for ``user``.

        This runs *before* the router. The returned scope is the only pool any
        retriever may search.

        Args:
            user: The authenticated (here: selected) demo user.

        Returns:
            An :class:`AuthorisedScope` listing permitted chunk and document
            IDs, plus counts and withheld *document types* for explanation.
            Withheld titles and text are never included: explaining an
            abstention must not itself leak the restricted material.
        """
        roles = user.role_set
        allowed: list[str] = []
        allowed_docs: list[str] = []
        withheld = 0
        withheld_types: set[str] = set()

        seen_docs: set[str] = set()
        for chunk in self._chunks:
            if chunk.readable_by(roles):
                allowed.append(chunk.chunk_id)
                if chunk.doc_id not in seen_docs:
                    seen_docs.add(chunk.doc_id)
                    allowed_docs.append(chunk.doc_id)
            else:
                withheld += 1
                withheld_types.add(chunk.doc_type.value)

        return AuthorisedScope(
            user_id=user.user_id,
            roles=sorted(roles),
            allowed_chunk_ids=allowed,
            allowed_doc_ids=allowed_docs,
            total_chunks=len(self._chunks),
            withheld_count=withheld,
            withheld_doc_types=sorted(withheld_types),
        )

    def filter_chunks(self, chunks: Iterable[Chunk], user: User) -> list[Chunk]:
        """Return only the chunks ``user`` may read."""
        roles = user.role_set
        return [c for c in chunks if c.readable_by(roles)]

    def filter_scored(
        self, scored: Iterable[ScoredChunk], user: User
    ) -> list[ScoredChunk]:
        """Return only the scored chunks ``user`` may read."""
        roles = user.role_set
        return [s for s in scored if s.chunk.readable_by(roles)]

    def assert_authorised(
        self, items: Iterable[ScoredChunk | Chunk], user: User, stage: str
    ) -> None:
        """Re-verify authorisation at a stage boundary.

        Defence in depth. If the pre-filter is correct this never raises; the
        cost is one set-intersection per chunk per stage, which is negligible at
        prototype scale and buys an invariant that is checked rather than
        assumed.

        Args:
            items: Chunks or scored chunks about to enter ``stage``.
            user: The requesting user.
            stage: Stage name, used in the error message and log line.

        Raises:
            UnauthorisedChunkError: If any item is outside the user's scope.
        """
        roles = user.role_set
        for item in items:
            chunk = item.chunk if isinstance(item, ScoredChunk) else item
            if not chunk.readable_by(roles):
                logger.error(
                    "ACL invariant violated at stage=%s user=%s chunk=%s",
                    stage,
                    user.user_id,
                    chunk.chunk_id,
                )
                raise UnauthorisedChunkError(
                    f"Chunk {chunk.chunk_id} reached stage '{stage}' but is not "
                    f"readable by user {user.user_id} (roles={sorted(roles)})."
                )

    def get(self, chunk_id: str) -> Chunk | None:
        """Return a chunk by ID without any authorisation check.

        Internal use only (index construction, evaluation label resolution).
        Callers that serve content to a user must go through
        :meth:`scope_for` or :meth:`filter_chunks`.
        """
        return self._by_id.get(chunk_id)

    def resolve_authorised(self, chunk_ids: Iterable[str], user: User) -> list[Chunk]:
        """Resolve IDs to chunks, dropping any the user may not read."""
        roles = user.role_set
        out: list[Chunk] = []
        for chunk_id in chunk_ids:
            chunk = self._by_id.get(chunk_id)
            if chunk is not None and chunk.readable_by(roles):
                out.append(chunk)
        return out
