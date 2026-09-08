"""Vector stores with ACL-scoped search.

Both backends accept an ``allowed_ids`` argument on every search. This is not a
convenience: it is how the "ACL before retrieval" invariant is realised at the
index layer. A search that is not given an allow-list searches nothing, so the
unsafe call is the one that fails, not the one that leaks.
"""

from __future__ import annotations

import logging
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class VectorStore(Protocol):
    """Interface every vector store implements."""

    name: str

    def build(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Replace the index contents with ``ids``/``vectors``."""

    def search(
        self, query_vector: np.ndarray, top_k: int, allowed_ids: Sequence[str]
    ) -> list[tuple[str, float]]:
        """Return ``(id, score)`` pairs restricted to ``allowed_ids``."""

    def __len__(self) -> int:
        """Number of indexed vectors."""


class NumpyVectorStore:
    """Exact cosine search over an in-process matrix.

    At prototype corpus sizes an exact scan is faster than an approximate index
    and removes recall variance from the evaluation, which matters when the
    whole point is to compare routing policies rather than ANN implementations.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self.name = "numpy"
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def build(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Replace the index contents.

        Raises:
            ValueError: If ``ids`` and ``vectors`` lengths disagree.
        """
        if len(ids) != len(vectors):
            raise ValueError(
                f"ids/vectors length mismatch: {len(ids)} vs {len(vectors)}"
            )
        self._ids = list(ids)
        self._index = {cid: i for i, cid in enumerate(self._ids)}
        array = np.asarray(vectors, dtype=np.float32)
        if not self._ids:
            self._matrix = np.zeros((0, 1), dtype=np.float32)
        else:
            self._matrix = array.reshape(len(self._ids), -1)

    def search(
        self, query_vector: np.ndarray, top_k: int, allowed_ids: Sequence[str]
    ) -> list[tuple[str, float]]:
        """Cosine-search the allow-listed subset only."""
        if self._matrix.size == 0 or top_k <= 0:
            return []
        rows = [self._index[cid] for cid in allowed_ids if cid in self._index]
        if not rows:
            return []
        subset = self._matrix[rows]
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = subset @ query
        limit = min(top_k, len(rows))
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[rows[int(i)]], float(scores[int(i)])) for i in top]

    def __len__(self) -> int:
        """Number of indexed vectors."""
        return len(self._ids)


class ChromaVectorStore:
    """Adapter for an optional in-memory ChromaDB collection."""

    def __init__(self, collection_name: str = "ahrag_chunks") -> None:
        """Create an ephemeral Chroma client and collection.

        Raises:
            ImportError: If ``chromadb`` is not installed.
            RuntimeError: If the client cannot be created.
        """
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb is not installed; install requirements-optional.txt or "
                "set AHRAG_VECTOR_STORE=numpy"
            ) from exc
        try:
            self._client = chromadb.EphemeralClient()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"Could not start ChromaDB client: {exc}") from exc
        self._collection_name = collection_name
        self._collection = None
        self._count = 0
        self.name = "chroma"

    def build(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Recreate the collection with the supplied vectors."""
        if len(ids) != len(vectors):
            raise ValueError(
                f"ids/vectors length mismatch: {len(ids)} vs {len(vectors)}"
            )
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - absent collection is the normal case
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
        if len(ids):
            self._collection.add(
                ids=list(ids), embeddings=np.asarray(vectors, dtype=np.float32).tolist()
            )
        self._count = len(ids)

    def search(
        self, query_vector: np.ndarray, top_k: int, allowed_ids: Sequence[str]
    ) -> list[tuple[str, float]]:
        """Search with an explicit ID allow-list passed to Chroma's ``where``."""
        if self._collection is None or top_k <= 0 or not allowed_ids:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=min(top_k, len(allowed_ids)),
            ids=list(allowed_ids),
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        # Chroma returns cosine *distance*; convert back to similarity.
        return [(cid, 1.0 - float(dist)) for cid, dist in zip(ids, distances)]

    def __len__(self) -> int:
        """Number of indexed vectors."""
        return self._count


def build_vector_store(backend: str) -> VectorStore:
    """Construct the configured vector store, degrading gracefully.

    Args:
        backend: ``"auto"``, ``"chroma"``, or ``"numpy"``.

    Returns:
        A ready vector store. ``"auto"`` prefers Chroma and silently falls back
        to numpy; an explicit ``"chroma"`` raises if unavailable.

    Raises:
        ValueError: If ``backend`` is unrecognised.
        ImportError / RuntimeError: If ``"chroma"`` is requested but unusable.
    """
    choice = backend.strip().lower()
    if choice not in {"auto", "chroma", "numpy"}:
        raise ValueError(
            f"Unknown vector store {backend!r}. Expected: auto, chroma, numpy"
        )
    if choice == "numpy":
        return NumpyVectorStore()
    if choice == "chroma":
        return ChromaVectorStore()
    try:
        store = ChromaVectorStore()
        logger.info("Vector store: chroma")
        return store
    except (ImportError, RuntimeError) as exc:
        logger.info("ChromaDB unavailable (%s); using numpy vector store.", exc)
        return NumpyVectorStore()
