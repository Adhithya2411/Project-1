"""Query decomposition for the iterative hybrid route (R4).

Rule-based and deterministic. An LLM decomposer would produce better
sub-queries, but it would also make R4's behaviour non-reproducible and its
latency dependent on an external service — both of which would undermine the
latency/cost comparison the evaluation exists to make. The decomposition
strategy is therefore explicit, cheap, and shown to the user in the subquery
trace, so a reviewer can see whether a weak answer came from bad decomposition
or bad retrieval.

Ordering matters: comparison splitting is tried before conjunction splitting,
because "compare A and B" must become two entity-scoped sub-queries rather than
one fragment ending in "compare A".
"""

from __future__ import annotations

import re
from typing import Sequence

_COMPARISON_SPLIT_RE = re.compile(
    r"\s+(?:versus|vs\.?|compared to|compared with|against|and how does)\s+",
    re.IGNORECASE,
)
_CONJUNCTION_SPLIT_RE = re.compile(
    r"\s*(?:;|\?|\band also\b|\band then\b|\bas well as\b)\s*", re.IGNORECASE
)
_DIFFERENCE_RE = re.compile(
    r"\b(?:difference|differences|differ|changed?|compare|comparison|contrast)\b",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)", re.IGNORECASE | re.DOTALL
)

_STOP_PREFIX_RE = re.compile(
    r"^(?:what(?:'s| is| are)?|how|why|when|which|who|tell me|explain|describe|"
    r"can you|could you|please)\s+", re.IGNORECASE
)


def decompose_query(query: str, max_subqueries: int = 3) -> list[str]:
    """Split ``query`` into retrievable sub-queries.

    Args:
        query: The normalised user query.
        max_subqueries: Hard cap on returned sub-queries. R4's iteration limit
            is enforced separately by the retrieval engine; this caps breadth,
            that caps depth.

    Returns:
        Between 1 and ``max_subqueries`` sub-queries. The original query is
        always first, so R4 degenerates gracefully to "hybrid retrieval run
        once" when no decomposition applies, rather than losing the original
        intent.

    Raises:
        ValueError: If ``max_subqueries`` is less than 1.
    """
    if max_subqueries < 1:
        raise ValueError("max_subqueries must be at least 1")

    cleaned = " ".join(query.split())
    if not cleaned:
        return []

    subqueries: list[str] = [cleaned]

    for candidate in _strategies(cleaned):
        normalised = " ".join(candidate.split()).strip(" ,.?")
        if not normalised or len(normalised.split()) < 2:
            continue
        if any(normalised.lower() == existing.lower() for existing in subqueries):
            continue
        subqueries.append(normalised)
        if len(subqueries) >= max_subqueries:
            break

    return subqueries[:max_subqueries]


def _strategies(query: str) -> list[str]:
    """Yield candidate sub-queries from each decomposition strategy, in order."""
    candidates: list[str] = []

    # 1. "between X and Y" -> one sub-query per entity, carrying the topic.
    match = _BETWEEN_RE.search(query)
    if match:
        topic = _topic_of(query[: match.start()])
        for entity in (match.group(1), match.group(2)):
            entity = entity.strip()
            candidates.append(f"{topic} {entity}".strip() if topic else entity)

    # 2. Explicit comparison connectives.
    if not candidates:
        parts = _COMPARISON_SPLIT_RE.split(query)
        if len(parts) > 1:
            topic = _topic_of(parts[0])
            candidates.append(parts[0].strip())
            for part in parts[1:]:
                part = part.strip()
                candidates.append(f"{topic} {part}".strip() if topic else part)

    # 3. Difference questions with no explicit split point: probe each side of
    #    the topic by asking for the current and the previous position.
    if not candidates and _DIFFERENCE_RE.search(query):
        topic = _topic_of(query)
        if topic:
            candidates.append(f"current {topic}")
            candidates.append(f"previous {topic}")

    # 4. Conjunctions and multi-sentence questions.
    if not candidates:
        parts = [p for p in _CONJUNCTION_SPLIT_RE.split(query) if p and p.strip()]
        if len(parts) > 1:
            candidates.extend(part.strip() for part in parts)

    return candidates


def _topic_of(fragment: str) -> str:
    """Strip interrogative scaffolding to leave the topical noun phrase."""
    text = _STOP_PREFIX_RE.sub("", fragment.strip())
    text = re.sub(
        r"\b(?:the|a|an|is|are|was|were|do|does|did|between|of|for|in|on)\b\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    text = _DIFFERENCE_RE.sub("", text)
    text = re.sub(r"\b(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip(" ,.?")


def iteration_weights(count: int, decay: float = 0.75) -> list[float]:
    """Return RRF weights that discount later decomposition iterations.

    The original query is iteration 0 and always carries full weight; each
    sub-query contributes ``decay ** i``. Without this, a query decomposed into
    three sub-queries could let peripheral evidence outvote the direct answer
    purely by appearing in more rankings.

    Raises:
        ValueError: If ``decay`` is outside ``(0, 1]``.
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    return [decay**index for index in range(max(0, count))]


def summarise(subqueries: Sequence[str]) -> str:
    """Render sub-queries as a one-line trace for logs and the UI."""
    return " | ".join(f"{i + 1}. {q}" for i, q in enumerate(subqueries))
