"""Embedding backends for dense retrieval.

Three backends, selected by ``AHRAG_EMBEDDING_BACKEND``:

``sentence-transformers``
    A real neural bi-encoder. Best quality; requires the optional dependency and
    a one-time model download.

``lsa`` (default when sentence-transformers is unavailable)
    Latent Semantic Analysis fitted on the ingested corpus: TF-IDF term-document
    matrix, truncated SVD, query vectors folded in against the learned term
    space. This is a genuine distributional-semantics dense retriever — it
    generalises across paraphrase within the corpus vocabulary — and it runs
    fully offline and deterministically with nothing but numpy.

    It also reproduces the failure mode the routing hypothesis depends on: SVD
    truncation discards the low-variance dimensions that carry rare identifiers,
    so an ``ERR-5041``-style token is smeared toward its semantic neighbours.
    That is the exact weakness the sparse route (R1) exists to cover, and having
    it present offline means the R1-vs-R2 contrast is observable without any
    model download.

``hashing``
    Deterministic hashed character/word n-gram projection. No fitting, so it
    works on a corpus too small for a meaningful SVD. Effectively fuzzy lexical
    matching, not semantics — it is a floor, not a recommendation.

Whichever is active is reported by ``GET /api/health`` and shown in the UI, so
a reviewer is never guessing which backend produced a result.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

# Below this many documents an SVD has too little to learn from; fall back.
_MIN_DOCS_FOR_LSA = 8


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer that keeps hyphenated identifiers intact.

    ``ERR-5041`` stays one token rather than becoming ``err`` + ``5041``. That
    single decision is what makes the sparse route able to match an exact error
    code at all, so it is shared by every backend and by BM25.
    """
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Interface every embedding backend implements."""

    name: str
    dim: int

    def fit(self, texts: Sequence[str]) -> None:
        """Fit any corpus-dependent state. May be a no-op."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2-normalised ``(len(texts), dim)`` float32 matrix."""


class HashingEmbedder:
    """Deterministic hashed n-gram embedding. No training, no downloads."""

    def __init__(self, dim: int = 384, char_ngram: int = 4) -> None:
        """Configure output dimensionality and character n-gram size."""
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.name = "hashing"
        self.dim = dim
        self.char_ngram = char_ngram

    def fit(self, texts: Sequence[str]) -> None:
        """No-op: this backend has no corpus-dependent state."""
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Project each text into the hashed feature space."""
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, weight in self._features(text).items():
                index, sign = self._bucket(feature)
                matrix[row, index] += sign * weight
        return _l2_normalise(matrix)

    def _features(self, text: str) -> dict[str, float]:
        """Word unigrams, bigrams, and character n-grams with sublinear TF."""
        tokens = tokenize(text)
        counts: Counter[str] = Counter()
        counts.update(f"w:{t}" for t in tokens)
        counts.update(f"b:{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        padded = f" {' '.join(tokens)} "
        n = self.char_ngram
        counts.update(
            f"c:{padded[i : i + n]}" for i in range(max(0, len(padded) - n + 1))
        )
        return {feature: 1.0 + math.log(count) for feature, count in counts.items()}

    def _bucket(self, feature: str) -> tuple[int, float]:
        """Map a feature to a bucket index and a signed weight."""
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0


class LSAEmbedder:
    """Latent Semantic Analysis over the ingested corpus.

    Fitting builds a TF-IDF term-document matrix and factorises it with a
    truncated SVD. Terms become vectors in the latent space; any text (chunk or
    query) is embedded by folding its TF-IDF weights against those term vectors.
    """

    # Above this many dense matrix cells (terms x documents) the exact
    # ``np.linalg.svd`` path is abandoned for a randomized truncated SVD that
    # never materialises the matrix. 60M cells is ~240 MB in float32, which is
    # the point where the dense path stops being a reasonable thing to do.
    DENSE_CELL_LIMIT = 60_000_000

    def __init__(self, dim: int = 192, min_df: int = 1) -> None:
        """Configure latent dimensionality and minimum document frequency."""
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.name = "lsa"
        self.dim = dim
        self.min_df = min_df
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._term_vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has successfully run."""
        return self._fitted

    def fit(self, texts: Sequence[str]) -> None:
        """Learn the latent term space from ``texts``.

        Raises:
            ValueError: If ``texts`` is empty or yields no usable vocabulary.
        """
        if not texts:
            raise ValueError("LSAEmbedder.fit requires at least one document")

        doc_tokens = [tokenize(t) for t in texts]
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))
        vocab_terms = sorted(term for term, count in df.items() if count >= self.min_df)
        if not vocab_terms:
            raise ValueError("LSAEmbedder.fit found no vocabulary terms")

        self._vocab = {term: i for i, term in enumerate(vocab_terms)}
        n_docs = len(texts)
        n_terms = len(vocab_terms)
        self._idf = np.array(
            [math.log((1 + n_docs) / (1 + df[term])) + 1.0 for term in vocab_terms],
            dtype=np.float32,
        )

        rank = int(min(self.dim, min(n_terms, n_docs) - 1))
        if rank < 2:
            raise ValueError("Corpus too small to fit an LSA space")

        # The dense term-document matrix is quadratic in corpus size and becomes
        # unusable well before the scale this system is meant to handle: at
        # ~6k documents the vocabulary reaches ~43k terms, so the matrix is
        # 1.2 GB and a full SVD on it does not finish. Below the limit keep the
        # exact path, so small-corpus behaviour is bit-identical to before.
        cells = n_terms * n_docs
        if cells <= self.DENSE_CELL_LIMIT:
            u, s = self._fit_dense(doc_tokens, n_terms, n_docs, rank)
            method = "exact"
        else:
            u, s = self._fit_randomized(doc_tokens, n_terms, n_docs, rank)
            method = "randomized"

        # Term vectors scaled by singular values: dominant semantic axes get
        # more weight, and the truncated tail (where rare identifiers live) is
        # discarded. That truncation is the modelled weakness, not an accident.
        self._term_vectors = (u[:, :rank] * s[:rank]).astype(np.float32)
        self.dim = rank
        self._fitted = True
        logger.info(
            "Fitted LSA space: %d terms, %d docs, rank %d (%s SVD)",
            n_terms,
            n_docs,
            rank,
            method,
        )

    # -- factorisation backends --------------------------------------------

    def _column_weights(self, tokens: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(row_indices, tf_idf_values)`` for one document column.

        Values are L2-normalised within the column, matching the dense path,
        so a long document does not dominate the factorisation.
        """
        counts = Counter(tokens)
        rows: list[int] = []
        values: list[float] = []
        for term, count in counts.items():
            row = self._vocab.get(term)
            if row is not None:
                rows.append(row)
                values.append((1.0 + math.log(count)) * float(self._idf[row]))
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        vector = np.asarray(values, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return np.asarray(rows, dtype=np.int64), vector

    def _fit_dense(
        self,
        doc_tokens: Sequence[Sequence[str]],
        n_terms: int,
        n_docs: int,
        rank: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact truncated SVD via a dense term-document matrix."""
        matrix = np.zeros((n_terms, n_docs), dtype=np.float32)
        for col, tokens in enumerate(doc_tokens):
            rows, values = self._column_weights(tokens)
            if rows.size:
                matrix[rows, col] = values
        try:
            u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - numerical edge
            raise ValueError(f"SVD failed while fitting LSA space: {exc}") from exc
        return u, s

    def _fit_randomized(
        self,
        doc_tokens: Sequence[Sequence[str]],
        n_terms: int,
        n_docs: int,
        rank: int,
        oversampling: int = 10,
        power_iterations: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomized truncated SVD (Halko, Martinsson & Tropp 2011).

        The matrix is never materialised. Only two products are needed —
        ``A @ X`` and ``A.T @ Y`` — and each is accumulated column by column
        from the sparse per-document term weights, so peak memory is
        ``O(n_terms * (rank + oversampling))`` rather than
        ``O(n_terms * n_docs)``.

        Two power iterations are used because TF-IDF spectra decay slowly; with
        none, the leading subspace is noticeably contaminated by the tail.

        Args:
            doc_tokens: Tokenised corpus, one sequence per document.
            n_terms: Vocabulary size.
            n_docs: Document count.
            rank: Target latent dimensionality.
            oversampling: Extra sketch columns, discarded after factorisation.
            power_iterations: Subspace refinement passes.

        Returns:
            ``(u, s)`` truncated to ``rank + oversampling`` columns, matching
            the shape contract of :meth:`_fit_dense`.
        """
        sketch_width = min(rank + oversampling, n_docs)
        # Column weights are needed on every pass; computing them once trades
        # a modest amount of memory for a large amount of repeated tokenisation.
        columns = [self._column_weights(tokens) for tokens in doc_tokens]

        def a_matmul(block: np.ndarray) -> np.ndarray:
            """Return ``A @ block`` for a ``(n_docs, k)`` block."""
            out = np.zeros((n_terms, block.shape[1]), dtype=np.float32)
            for col, (rows, values) in enumerate(columns):
                if rows.size:
                    # Outer product of one sparse column with its row of `block`.
                    np.add.at(out, rows, np.outer(values, block[col]))
            return out

        def at_matmul(block: np.ndarray) -> np.ndarray:
            """Return ``A.T @ block`` for a ``(n_terms, k)`` block."""
            out = np.zeros((n_docs, block.shape[1]), dtype=np.float32)
            for col, (rows, values) in enumerate(columns):
                if rows.size:
                    out[col] = values @ block[rows]
            return out

        rng = np.random.RandomState(1729)
        sketch = rng.normal(size=(n_docs, sketch_width)).astype(np.float32)

        basis, _ = np.linalg.qr(a_matmul(sketch))
        for _ in range(power_iterations):
            basis, _ = np.linalg.qr(a_matmul(at_matmul(basis)))

        # Project onto the captured subspace and factorise the small matrix.
        projected = at_matmul(basis).T  # (sketch_width, n_docs)
        try:
            u_small, s, _ = np.linalg.svd(projected, full_matrices=False)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - numerical edge
            raise ValueError(f"SVD failed while fitting LSA space: {exc}") from exc
        return (basis @ u_small).astype(np.float32), s.astype(np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Fold ``texts`` into the learned latent space.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._fitted:
            raise RuntimeError("LSAEmbedder.encode called before fit()")
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(tokenize(text))
            if not counts:
                continue
            weights = np.zeros(len(self._vocab), dtype=np.float32)
            for term, count in counts.items():
                index = self._vocab.get(term)
                if index is not None:
                    weights[index] = (1.0 + math.log(count)) * self._idf[index]
            norm = float(np.linalg.norm(weights))
            if norm > 0:
                weights /= norm
            matrix[row] = weights @ self._term_vectors
        return _l2_normalise(matrix)


class SentenceTransformerEmbedder:
    """Adapter for an optional ``sentence-transformers`` bi-encoder."""

    def __init__(self, model_name: str) -> None:
        """Load ``model_name``.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
            RuntimeError: If the model cannot be loaded (e.g. offline first run).
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed; "
                "install requirements-optional.txt or set "
                "AHRAG_EMBEDDING_BACKEND=lsa"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # pragma: no cover - network/model errors
            raise RuntimeError(
                f"Could not load embedding model {model_name!r}: {exc}"
            ) from exc
        self.name = f"sentence-transformers:{model_name}"
        # Renamed in sentence-transformers 3.x; the old name still works but
        # emits a FutureWarning on every construction.
        dimension = getattr(self._model, "get_embedding_dimension", None)
        if not callable(dimension):
            dimension = self._model.get_sentence_embedding_dimension
        self.dim = int(dimension())

    def fit(self, texts: Sequence[str]) -> None:
        """No-op: the encoder is pre-trained."""
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode with the neural bi-encoder and L2-normalise."""
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False
        )
        return _l2_normalise(np.asarray(vectors, dtype=np.float32))


def build_embedder(
    backend: str, model_name: str, dim: int, corpus_size: int = 0
) -> EmbeddingBackend:
    """Construct the configured embedding backend, degrading gracefully.

    Args:
        backend: ``"auto"``, ``"sentence-transformers"``, ``"lsa"``, or ``"hashing"``.
        model_name: Model id, used only by the sentence-transformers backend.
        dim: Target dimensionality for the offline backends.
        corpus_size: Number of documents available for fitting. An LSA space
            needs a handful of documents to be meaningful; below that the
            hashing backend is used instead.

    Returns:
        A ready (but not yet fitted) embedding backend.

    Raises:
        ValueError: If ``backend`` is not a recognised name.
        ImportError / RuntimeError: If an explicitly requested backend is
            unavailable. ``"auto"`` never raises for availability reasons.
    """
    choice = backend.strip().lower()
    if choice not in {"auto", "sentence-transformers", "lsa", "hashing"}:
        raise ValueError(
            f"Unknown embedding backend {backend!r}. Expected one of: "
            "auto, sentence-transformers, lsa, hashing"
        )

    if choice == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    if choice == "hashing":
        return HashingEmbedder(dim=dim)
    if choice == "lsa":
        return LSAEmbedder(dim=min(dim, 256))

    # auto
    try:
        embedder = SentenceTransformerEmbedder(model_name)
        logger.info("Embedding backend: %s", embedder.name)
        return embedder
    except (ImportError, RuntimeError) as exc:
        logger.info(
            "sentence-transformers unavailable (%s); using offline embedding backend.",
            exc.__class__.__name__,
        )
    if corpus_size >= _MIN_DOCS_FOR_LSA:
        return LSAEmbedder(dim=min(dim, 256))
    logger.info(
        "Corpus of %d documents is below the LSA minimum of %d; using hashing backend.",
        corpus_size,
        _MIN_DOCS_FOR_LSA,
    )
    return HashingEmbedder(dim=dim)


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows, leaving all-zero rows as zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-9)).astype(np.float32)
