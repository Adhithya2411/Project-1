"""Reciprocal Rank Fusion for the hybrid routes (R3, R4).

RRF combines rankings rather than scores:

    RRF(d) = Σ_r 1 / (k + rank_r(d))

The rank-based formulation is the point. BM25 scores and cosine similarities
live on incompatible scales, and any attempt to combine them numerically
requires a normalisation that is itself a tunable, corpus-dependent choice. RRF
sidesteps that entirely by discarding magnitudes.

``k`` (default 60) damps the influence of top ranks: a document ranked first by
one retriever does not automatically outrank a document ranked second by both.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked ID lists into one.

    Args:
        rankings: Ranked lists of chunk IDs, best first. Lists may overlap
            partially or not at all.
        k: RRF smoothing constant. Larger values flatten the contribution of
            top ranks.
        weights: Optional per-ranking weights. Defaults to uniform. Supplied by
            R4 so that later decomposition iterations can be down-weighted
            relative to the first.

    Returns:
        ``(chunk_id, fused_score)`` pairs sorted by descending score. Ties break
        on chunk ID so that fusion output is deterministic — which matters
        because the evaluation compares route policies, and non-deterministic
        tie-breaking would add noise to that comparison.

    Raises:
        ValueError: If ``k`` is negative or ``weights`` length mismatches.
    """
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(
            f"weights length {len(weights)} does not match rankings length "
            f"{len(rankings)}"
        )

    effective_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)

    for ranking, weight in zip(rankings, effective_weights):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += weight / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def rank_positions(ranking: Iterable[str]) -> dict[str, int]:
    """Return a ``chunk_id -> 1-based rank`` map, keeping the best rank on repeats."""
    positions: dict[str, int] = {}
    for rank, chunk_id in enumerate(ranking, start=1):
        positions.setdefault(chunk_id, rank)
    return positions
