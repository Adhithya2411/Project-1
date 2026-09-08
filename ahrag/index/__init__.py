"""Indexing backends: embeddings, vector store, and sparse BM25."""

from .embeddings import EmbeddingBackend, HashingEmbedder, build_embedder
from .sparse import SparseIndex
from .store import IndexBundle
from .vector_store import NumpyVectorStore, VectorStore, build_vector_store

__all__ = [
    "EmbeddingBackend",
    "HashingEmbedder",
    "IndexBundle",
    "NumpyVectorStore",
    "SparseIndex",
    "VectorStore",
    "build_embedder",
    "build_vector_store",
]
