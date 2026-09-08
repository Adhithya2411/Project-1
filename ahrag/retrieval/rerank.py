"""Reranking of the fused candidate pool.

Two backends. A cross-encoder when ``sentence-transformers`` is installed
(joint query/passage scoring, the higher-precision option), and otherwise a
transparent lexical-semantic scorer that a reviewer can read end to end.

The fallback is not a stub. It combines four interpretable components — IDF-
weighted term overlap, exact-phrase and identifier matching, heading match, and
coverage — and its per-component contributions are returned so the UI can show
*why* a chunk was ranked where it was. Precision is lower than a trained
cross-encoder's; explicability is higher. For a prototype whose subject is
auditability, that is the right default.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

from ..index.embeddings import tokenize
from ..models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that
    the to was were what when where which who why will with would you your do
    does i my we our""".split()
)


@runtime_checkable
class Reranker(Protocol):
    """Interface for reranking a candidate pool against a query."""

    name: str

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Return the top ``top_k`` candidates, rescored and reordered."""


class LexicalReranker:
    """Transparent lexical-semantic reranker with per-component explanations."""

    def __init__(self, corpus: Sequence[Chunk] | None = None) -> None:
        """Optionally fit IDF statistics over ``corpus``.

        Without a corpus the reranker still works, treating every term as
        equally informative; with one, rare terms dominate the overlap score,
        which is what makes an exact error code outrank generic policy prose.
        """
        self.name = "lexical"
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0
        if corpus:
            self.fit(corpus)

    def fit(self, corpus: Sequence[Chunk]) -> None:
        """Compute IDF over the chunk corpus."""
        n_docs = len(corpus)
        if n_docs == 0:
            return
        df: Counter[str] = Counter()
        for chunk in corpus:
            df.update(set(tokenize(chunk.text)))
        self._idf = {
            term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()
        }
        self._default_idf = math.log(1 + n_docs) + 1.0

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Rescore ``candidates`` and return the best ``top_k``."""
        if not candidates or top_k <= 0:
            return []
        rescored = [
            item.model_copy(
                update={
                    "rerank_score": self.score(query, item.chunk),
                    "score": self.score(query, item.chunk),
                }
            )
            for item in candidates
        ]
        rescored.sort(key=lambda s: (-s.score, s.chunk_id))
        return rescored[:top_k]

    def score(self, query: str, chunk: Chunk) -> float:
        """Return a ``0..1`` relevance score for one query/chunk pair."""
        components = self.explain(query, chunk)
        return components["score"]

    def explain(self, query: str, chunk: Chunk) -> dict[str, float]:
        """Return the score and its components, for display and debugging."""
        query_tokens = [t for t in tokenize(query) if t not in _STOPWORDS]
        if not query_tokens:
            return {"score": 0.0, "overlap": 0.0, "phrase": 0.0, "heading": 0.0, "coverage": 0.0}

        chunk_tokens = tokenize(chunk.text)
        chunk_counts = Counter(chunk_tokens)
        chunk_len = max(1, len(chunk_tokens))

        # IDF-weighted, length-normalised term overlap.
        matched_weight = 0.0
        total_weight = 0.0
        matched_terms = 0
        for term in set(query_tokens):
            weight = self._idf.get(term, self._default_idf)
            total_weight += weight
            count = chunk_counts.get(term, 0)
            if count:
                matched_terms += 1
                # A single occurrence of a query term is already strong
                # evidence in a ~900-character chunk; repeats add a little
                # more. A steeper saturation constant here would compress every
                # score toward zero and make the sufficiency threshold
                # meaningless.
                saturation = count / (count + 0.35)
                matched_weight += weight * saturation
        overlap = matched_weight / total_weight if total_weight else 0.0

        # Exact phrase / identifier presence: the strongest single signal for
        # the sparse route's target queries.
        lowered_chunk = chunk.text.lower()
        lowered_query = query.lower()
        phrase = 0.0
        bigrams = [
            f"{a} {b}" for a, b in zip(query_tokens, query_tokens[1:]) if len(a) + len(b) > 6
        ]
        if bigrams:
            hits = sum(1 for bg in bigrams if bg in lowered_chunk)
            phrase = min(1.0, hits / len(bigrams))
        identifier_hits = [
            token
            for token in query_tokens
            if _is_distinctive_identifier(token) and token in lowered_chunk
        ]
        if identifier_hits:
            phrase = min(1.0, phrase + 0.6)
        if len(lowered_query) > 12 and lowered_query in lowered_chunk:
            phrase = 1.0

        heading = 0.0
        if chunk.heading:
            heading_tokens = set(tokenize(chunk.heading))
            if heading_tokens:
                heading = len(heading_tokens & set(query_tokens)) / len(set(query_tokens))

        coverage = matched_terms / len(set(query_tokens))

        # Slight penalty for very long chunks: a term match inside a 900-char
        # chunk is weaker evidence than the same match in a tight clause.
        length_penalty = 1.0 / (1.0 + max(0.0, (chunk_len - 220) / 900.0))

        score = (
            0.40 * overlap + 0.18 * phrase + 0.10 * heading + 0.32 * coverage
        ) * length_penalty
        return {
            "score": round(min(1.0, score), 6),
            "overlap": round(overlap, 4),
            "phrase": round(phrase, 4),
            "heading": round(heading, 4),
            "coverage": round(coverage, 4),
        }


class CrossEncoderReranker:
    """Adapter for an optional ``sentence-transformers`` cross-encoder."""

    def __init__(self, model_name: str) -> None:
        """Load the cross-encoder model.

        Raises:
            ImportError: If ``sentence-transformers`` is absent.
            RuntimeError: If the model cannot be loaded.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed; set AHRAG_RERANKER=lexical"
            ) from exc
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:  # pragma: no cover - network/model errors
            raise RuntimeError(
                f"Could not load cross-encoder {model_name!r}: {exc}"
            ) from exc
        self.name = f"cross-encoder:{model_name}"

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Jointly score each query/chunk pair and return the best ``top_k``."""
        if not candidates or top_k <= 0:
            return []
        pairs = [(query, item.chunk.text) for item in candidates]
        raw = self._model.predict(pairs)
        # Cross-encoder logits are unbounded; squash to 0..1 so downstream
        # thresholds (evidence sufficiency) mean the same thing for both backends.
        scores = [1.0 / (1.0 + math.exp(-float(value))) for value in raw]
        rescored = [
            item.model_copy(update={"rerank_score": score, "score": score})
            for item, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda s: (-s.score, s.chunk_id))
        return rescored[:top_k]


def build_reranker(
    backend: str, model_name: str, corpus: Sequence[Chunk] | None = None
) -> Reranker:
    """Construct the configured reranker, degrading gracefully.

    Args:
        backend: ``"auto"``, ``"cross-encoder"``, or ``"lexical"``.
        model_name: Cross-encoder model id.
        corpus: Chunks used to fit IDF for the lexical backend.

    Returns:
        A ready reranker. ``"auto"`` prefers the cross-encoder and falls back
        silently; an explicit ``"cross-encoder"`` raises if unavailable.

    Raises:
        ValueError: If ``backend`` is unrecognised.
    """
    choice = backend.strip().lower()
    if choice not in {"auto", "cross-encoder", "lexical"}:
        raise ValueError(
            f"Unknown reranker {backend!r}. Expected: auto, cross-encoder, lexical"
        )
    if choice == "lexical":
        return LexicalReranker(corpus)
    if choice == "cross-encoder":
        return CrossEncoderReranker(model_name)
    try:
        reranker = CrossEncoderReranker(model_name)
        logger.info("Reranker: %s", reranker.name)
        return reranker
    except (ImportError, RuntimeError) as exc:
        logger.info("Cross-encoder unavailable (%s); using lexical reranker.", exc)
        return LexicalReranker(corpus)


_BARE_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")


def _is_distinctive_identifier(token: str) -> bool:
    """True when a token looks like an enterprise identifier, not just a number.

    ``ERR-5041``, ``POL-HR-014``, and ``SOX-404b`` qualify. Bare numbers and
    bare years do not: matching "2026" between a query and a chunk is a
    coincidence of calendar, not evidence of relevance, and treating it as an
    identifier hit was producing false high-confidence matches on off-topic
    documents.
    """
    if len(token) < 4 or _BARE_YEAR_RE.match(token):
        return False
    has_digit = any(ch.isdigit() for ch in token)
    has_alpha = any(ch.isalpha() for ch in token)
    return has_digit and has_alpha


_WHITESPACE_RE = re.compile(r"\s+")


def normalise_query(query: str, history: Sequence[str] | None = None) -> str:
    """Normalise a query and lightly resolve follow-ups against history.

    A short pronoun-led follow-up ("what about the older one?") is prefixed with
    the previous turn so retrieval has something to match. This is a documented
    heuristic, not coreference resolution: the appended context is visible in
    the R4 subquery trace so a user can see exactly what was searched.
    """
    cleaned = _WHITESPACE_RE.sub(" ", query).strip()
    if not history:
        return cleaned
    tokens = tokenize(cleaned)
    pronoun_led = bool(tokens) and tokens[0] in {
        "it", "that", "this", "they", "those", "these", "he", "she", "and", "what",
    }
    if len(tokens) <= 8 and pronoun_led:
        return f"{history[-1].strip()} {cleaned}".strip()
    return cleaned
