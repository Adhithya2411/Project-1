"""Sparse BM25 index with ACL-scoped search."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import tokenize

logger = logging.getLogger(__name__)


class SparseIndex:
    """BM25 over chunk text, restricted to an allow-list on every query.

    BM25 scores are unbounded and corpus-dependent, so ``search`` returns both
    the raw score and a max-normalised score. The normalised value is what feeds
    the router's ``sparse_confidence`` feature — an absolute BM25 score would
    make the router's thresholds meaningless across corpora.
    """

    def __init__(self) -> None:
        """Create an empty index."""
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        """Build the index over ``texts``.

        Raises:
            ValueError: If ``ids`` and ``texts`` lengths disagree.
        """
        if len(ids) != len(texts):
            raise ValueError(f"ids/texts length mismatch: {len(ids)} vs {len(texts)}")
        self._ids = list(ids)
        self._index = {cid: i for i, cid in enumerate(self._ids)}
        if not ids:
            self._bm25 = None
            return
        corpus = [tokenize(t) for t in texts]
        # rank_bm25 cannot handle a document with no tokens; substitute a
        # placeholder so index positions stay aligned with `ids`.
        corpus = [doc if doc else ["__empty__"] for doc in corpus]
        self._bm25 = BM25Okapi(corpus)
        logger.info("Built BM25 index over %d chunks", len(ids))

    def search(
        self, query: str, top_k: int, allowed_ids: Sequence[str]
    ) -> list[tuple[str, float, float]]:
        """Search the allow-listed subset.

        Args:
            query: Raw query text.
            top_k: Maximum results to return.
            allowed_ids: The ACL-authorised chunk IDs. An empty allow-list
                returns no results — never the full corpus.

        Returns:
            ``(chunk_id, raw_bm25_score, normalised_score)`` triples, best first.
            The normalised score divides by the best score *within the
            authorised subset*, so it answers "how good is this relative to the
            best thing this user is allowed to see".
        """
        if self._bm25 is None or top_k <= 0 or not allowed_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        rows = [self._index[cid] for cid in allowed_ids if cid in self._index]
        if not rows:
            return []

        all_scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        subset = all_scores[rows]
        limit = min(top_k, len(rows))
        top = np.argpartition(-subset, limit - 1)[:limit]
        top = top[np.argsort(-subset[top])]

        best = float(subset.max()) if subset.size else 0.0
        out: list[tuple[str, float, float]] = []
        for position in top:
            raw = float(subset[int(position)])
            if raw <= 0.0:
                continue
            out.append((self._ids[rows[int(position)]], raw, raw / best if best > 0 else 0.0))
        return out

    # -- scope specialisation hooks ----------------------------------------

    def idf_snapshot(self) -> dict[str, float]:
        """Return a copy of the fitted IDF table.

        Exposed so ``index/scoped.py`` can interpolate a class-local IDF
        towards the global one. Returns a copy: a caller mutating the live
        table would silently corrupt every subsequent query.
        """
        if self._bm25 is None:
            return {}
        return dict(self._bm25.idf)

    def shrink_idf_towards(
        self, reference_idf: dict[str, float], lam: float
    ) -> None:
        """Interpolate this index's IDF towards ``reference_idf``.

        The statistical motive is ordinary shrinkage: a small subcorpus gives
        noisy document-frequency estimates, and a broader reference stabilises
        them. The governance consequence is the interesting part. When the
        reference is the *global* IDF table, it is a function of chunks outside
        this class, so every ``lam > 0`` reopens the information-flow channel
        that class-local fitting closes.

        ``lam`` is therefore not a tuning knob like any other: it is an
        explicit information-flow budget for the sparse channel. ``lam=0`` is
        provably non-interfering and ``lam=1`` restores the global IDF values;
        the curve between them is the purity/utility frontier the evaluation
        harness measures. It is a *partial* budget: the dense basis is fitted
        per class independently of ``lam``, so ``lam=1`` does not restore the
        unspecialised system as a whole.

        Args:
            reference_idf: IDF table to shrink towards, typically the global one.
            lam: Interpolation weight in ``[0, 1]``. ``0`` leaves this index
                untouched and pure.

        Raises:
            ValueError: If ``lam`` is outside ``[0, 1]``.
        """
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        if self._bm25 is None or lam == 0.0 or not reference_idf:
            return
        local = self._bm25.idf
        # Terms absent from the local vocabulary are deliberately *not* added.
        # Introducing them would let a term that occurs only in unauthorised
        # documents acquire a score here, which is the leak in its purest form.
        for term, local_value in local.items():
            reference = reference_idf.get(term)
            if reference is not None:
                local[term] = (1.0 - lam) * local_value + lam * reference

    def __len__(self) -> int:
        """Number of indexed chunks."""
        return len(self._ids)
