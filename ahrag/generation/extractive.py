"""Deterministic extractive generator — the default, always-offline backend.

It selects the sentences from the evidence pack that best answer the query and
emits them verbatim with ``[chunk_id]`` citations. It never paraphrases and
never composes new claims, which means faithfulness is structural rather than
hoped-for: every sentence in the answer exists, word for word, in a passage the
user was authorised to read.

The trade-off is real and stated plainly: extractive output is blunter and less
fluent than an LLM's, and it cannot synthesise across passages. What it buys is
that the prototype runs with no API key, produces byte-identical output across
runs, and makes evaluation numbers attributable to the *retrieval and routing*
policy rather than to generation variance. When ``ANTHROPIC_API_KEY`` is set,
the adapter takes over and the same citation verifier polices its output.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from ..index.embeddings import tokenize
from ..models import ConflictReport, ScoredChunk
from .base import GeneratedAnswer, estimate_tokens, verify_citations

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9£$])|\n(?=[-*\d#])|\n{2,}")

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that
    the to was were what when where which who why will with would you your do
    does i my we our can could should about""".split()
)

MAX_SENTENCES_PER_CHUNK = 2
MAX_TOTAL_SENTENCES = 6


class ExtractiveGenerator:
    """Selects and cites the most responsive sentences from the evidence pack."""

    def __init__(self, max_sentences: int = MAX_TOTAL_SENTENCES) -> None:
        """Configure how many sentences the answer may contain."""
        self.name = "extractive-deterministic"
        self.max_sentences = max_sentences

    def generate(
        self,
        query: str,
        evidence: Sequence[ScoredChunk],
        conflicts: Sequence[ConflictReport] = (),
        freshness_warnings: Sequence[str] = (),
    ) -> GeneratedAnswer:
        """Build a cited answer from ``evidence``.

        Args:
            query: The user's question.
            evidence: Validated, ACL-filtered evidence, best first.
            conflicts: Conflicts to disclose in the answer body.
            freshness_warnings: Currency warnings to append.

        Returns:
            A :class:`GeneratedAnswer` whose citations have been verified
            against the evidence pack.
        """
        if not evidence:
            return GeneratedAnswer(
                text=(
                    "No authorised evidence was available, so no answer can be "
                    "grounded."
                ),
                generator=self.name,
            )

        query_terms = self._weighted_terms(query, evidence)
        selected = self._select_sentences(query_terms, evidence)

        if not selected:
            # Nothing scored above zero: fall back to the leading sentence of
            # the top chunk rather than emitting an empty answer.
            top = evidence[0]
            first = self._sentences(top.chunk.text)
            selected = [(first[0], top)] if first else []

        lines: list[str] = []
        for sentence, item in selected:
            lines.append(f"- {sentence.strip()} [{item.chunk_id}]")

        body = "\n".join(lines)
        parts = [body]

        if conflicts:
            parts.append("")
            parts.append("**Conflicting sources — disclosed, not resolved:**")
            for conflict in conflicts:
                ids = ", ".join(f"[{cid}]" for cid in conflict.chunk_ids)
                parts.append(f"- {conflict.description} Sources: {ids}")
                if conflict.preference_reason:
                    parts.append(
                        f"  Preferred for a current-state answer: "
                        f"{conflict.preference_reason}"
                    )

        if freshness_warnings:
            parts.append("")
            parts.append("**Freshness notes:**")
            parts.extend(f"- {warning}" for warning in freshness_warnings)

        raw = "\n".join(parts)
        cleaned, citations, stripped = verify_citations(raw, evidence)

        prompt_size = sum(len(item.chunk.text) for item in evidence) + len(query)
        return GeneratedAnswer(
            text=cleaned,
            citations=citations,
            generator=self.name,
            input_tokens=estimate_tokens("x" * prompt_size),
            output_tokens=estimate_tokens(cleaned),
            stripped_citations=stripped,
        )

    # -- selection ---------------------------------------------------------

    @staticmethod
    def _weighted_terms(
        query: str, evidence: Sequence[ScoredChunk]
    ) -> dict[str, float]:
        """Return query terms weighted by IDF over the evidence pack.

        Weighting against the pack rather than the whole corpus is deliberate:
        the useful question here is which query terms *discriminate between the
        passages under consideration*, not which are globally rare.
        """
        terms = [t for t in tokenize(query) if t not in _STOPWORDS]
        if not terms:
            return {}
        n_docs = max(1, len(evidence))
        df: Counter[str] = Counter()
        for item in evidence:
            df.update(set(tokenize(item.chunk.text)))
        weights: dict[str, float] = {}
        for term in set(terms):
            weight = math.log((1 + n_docs) / (1 + df.get(term, 0))) + 1.0
            # Identifier-shaped terms are the highest-value match signal.
            if any(ch.isdigit() for ch in term) or "-" in term:
                weight *= 2.2
            weights[term] = weight
        return weights

    def _select_sentences(
        self, query_terms: dict[str, float], evidence: Sequence[ScoredChunk]
    ) -> list[tuple[str, ScoredChunk]]:
        """Pick the best-matching sentences, capped per chunk and overall."""
        if not query_terms:
            return []

        scored: list[tuple[float, int, int, str, ScoredChunk]] = []
        for chunk_rank, item in enumerate(evidence):
            sentences = self._sentences(item.chunk.text)
            for position, sentence in enumerate(sentences):
                score = self._sentence_score(sentence, query_terms)
                if score <= 0:
                    continue
                # Retrieval rank and sentence position both act as mild priors:
                # better-ranked chunks and earlier sentences win ties.
                adjusted = score * (1.0 - 0.06 * chunk_rank) * (1.0 - 0.015 * position)
                scored.append((adjusted, chunk_rank, position, sentence, item))

        scored.sort(key=lambda row: (-row[0], row[1], row[2]))

        per_chunk: Counter[str] = Counter()
        chosen: list[tuple[float, int, int, str, ScoredChunk]] = []
        for row in scored:
            _, _, _, _, item = row
            if per_chunk[item.chunk_id] >= MAX_SENTENCES_PER_CHUNK:
                continue
            per_chunk[item.chunk_id] += 1
            chosen.append(row)
            if len(chosen) >= self.max_sentences:
                break

        # Present in evidence order, then document order — reads as prose
        # rather than as a ranked list.
        chosen.sort(key=lambda row: (row[1], row[2]))
        return [(sentence, item) for _, _, _, sentence, item in chosen]

    @staticmethod
    def _sentence_score(sentence: str, query_terms: dict[str, float]) -> float:
        """Score one sentence against the weighted query terms."""
        tokens = set(tokenize(sentence))
        if not tokens:
            return 0.0
        matched = sum(weight for term, weight in query_terms.items() if term in tokens)
        total = sum(query_terms.values())
        if total <= 0:
            return 0.0
        coverage = matched / total
        # Prefer informative sentences: very short fragments and very long
        # run-ons both make poor standalone answers.
        length = len(sentence.split())
        length_factor = 1.0 if 6 <= length <= 45 else (0.6 if length < 6 else 0.8)
        return coverage * length_factor

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Split chunk text into sentence-like units, preserving list items."""
        parts = _SENTENCE_RE.split(text)
        out: list[str] = []
        for part in parts:
            cleaned = " ".join(part.split()).strip()
            cleaned = re.sub(r"^[-*#>\s]+", "", cleaned).strip()
            if len(cleaned) >= 12:
                out.append(cleaned)
        return out
