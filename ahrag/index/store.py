"""The combined index bundle: sparse + dense over one chunk snapshot."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..config import Settings
from ..models import Chunk
from .embeddings import EmbeddingBackend, build_embedder
from .sparse import SparseIndex
from .vector_store import VectorStore, build_vector_store

logger = logging.getLogger(__name__)


class IndexBundle:
    """Owns the BM25 index, the vector store, and the embedding backend.

    Rebuilt wholesale on ingestion. Holding both families behind one object
    guarantees they always describe the same chunk snapshot — a stale sparse
    index paired with a fresh dense one would silently corrupt RRF fusion.
    """

    def __init__(self, settings: Settings) -> None:
        """Create empty indexes bound to ``settings``."""
        self.settings = settings
        self.sparse = SparseIndex()
        self.vectors: VectorStore = build_vector_store(settings.vector_store)
        self.embedder: EmbeddingBackend | None = None
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        # Lazily created so a specialised bundle (which is itself an
        # IndexBundle) does not recursively construct a specialisation manager.
        self._specialisation: object | None = None

    @property
    def chunks(self) -> list[Chunk]:
        """The indexed chunk snapshot."""
        return list(self._chunks)

    @property
    def embedder_name(self) -> str:
        """Name of the active embedding backend, or ``'none'`` before build."""
        return self.embedder.name if self.embedder is not None else "none"

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Rebuild both indexes over ``chunks``.

        Raises:
            RuntimeError: If the dense index cannot be built. The sparse index
                is built first and kept, so a dense failure degrades the system
                to sparse-only rather than taking it down; the exception still
                propagates so the caller can report it.
        """
        self._chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self._chunks}
        ids = [c.chunk_id for c in self._chunks]
        texts = [c.text for c in self._chunks]

        self.sparse.build(ids, texts)

        if not ids:
            self.vectors.build([], np.zeros((0, 1), dtype=np.float32))
            return

        doc_count = len({c.doc_id for c in self._chunks})
        self.embedder = build_embedder(
            self.settings.embedding_backend,
            self.settings.embedding_model,
            self.settings.embedding_dim,
            corpus_size=max(doc_count, len(ids) // 4),
        )
        try:
            self.embedder.fit(texts)
            vectors = self.embedder.encode(texts)
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "Dense index build failed with %s (%s); retrying with hashing backend.",
                self.embedder.name,
                exc,
            )
            from .embeddings import HashingEmbedder

            self.embedder = HashingEmbedder(dim=self.settings.embedding_dim)
            self.embedder.fit(texts)
            vectors = self.embedder.encode(texts)

        self.vectors.build(ids, vectors)
        # Per-class indexes are derived from this snapshot, so they are stale
        # the moment it changes.
        if self._specialisation is not None:
            self._specialisation.invalidate()
        logger.info(
            "Index built: %d chunks, sparse=BM25, dense=%s/%s",
            len(ids),
            self.embedder.name,
            self.vectors.name,
        )

    # -- scope specialisation ----------------------------------------------

    @property
    def specialisation(self):  # noqa: ANN201 - ScopeSpecialisedIndex
        """The per-ACL-class index manager, created on first access."""
        if self._specialisation is None:
            from .scoped import ScopeSpecialisedIndex

            self._specialisation = ScopeSpecialisedIndex(self.settings, self)
        return self._specialisation

    def for_scope(self, scope):  # noqa: ANN001, ANN201 - AuthorisedScope/ScopedIndex
        """Return the retrieval view specialised to ``scope``.

        The single entry point used by ``retrieval/pipeline.py``. When
        ``Settings.index_specialisation`` is off this returns a view backed by
        this bundle, so the caller's behaviour is bit-identical to reading
        ``self.sparse`` / ``self.vectors`` directly.
        """
        return self.specialisation.for_scope(scope)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query.

        Raises:
            RuntimeError: If the index has not been built.
        """
        if self.embedder is None:
            raise RuntimeError("IndexBundle.encode_query called before build()")
        return self.embedder.encode([query])[0]

    def get(self, chunk_id: str) -> Chunk | None:
        """Return an indexed chunk by ID, without an authorisation check."""
        return self._by_id.get(chunk_id)

    def backend_info(self) -> dict[str, str | int]:
        """Return which backends are active, for the health endpoint and UI."""
        return {
            "sparse": "rank-bm25/BM25Okapi",
            "embedder": self.embedder_name,
            "vector_store": self.vectors.name,
            "chunks": len(self._chunks),
        }
