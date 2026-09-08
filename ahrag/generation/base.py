"""Generator interface, citation verification, and backend selection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from ..config import Settings
from ..models import Citation, ConflictReport, ScoredChunk

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9\-_.]*::c\d+)\]")


@dataclass
class GeneratedAnswer:
    """A generated answer with its citations and token accounting."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    generator: str = "extractive"
    input_tokens: int = 0
    output_tokens: int = 0
    stripped_citations: list[str] = field(default_factory=list)


@runtime_checkable
class Generator(Protocol):
    """Interface implemented by every generation backend."""

    name: str

    def generate(
        self,
        query: str,
        evidence: Sequence[ScoredChunk],
        conflicts: Sequence[ConflictReport] = (),
        freshness_warnings: Sequence[str] = (),
    ) -> GeneratedAnswer:
        """Produce a grounded answer from ``evidence`` alone."""


def verify_citations(
    text: str, evidence: Sequence[ScoredChunk]
) -> tuple[str, list[Citation], list[str]]:
    """Strip any citation that does not map to a served evidence chunk.

    This runs on *every* backend's output, including the LLM adapter's. A model
    that invents a plausible-looking chunk id, or cites a real chunk that was
    filtered out of this user's pack, has that citation removed before the
    answer is displayed — so a citation shown to a user is always resolvable to
    a passage that user was authorised to see.

    Args:
        text: Raw generated answer.
        evidence: The evidence pack actually served.

    Returns:
        ``(cleaned_text, citations, stripped_ids)``. ``citations`` is ordered by
        first appearance in the text, so the displayed list matches reading order.
    """
    by_id = {item.chunk_id: item.chunk for item in evidence}
    stripped: list[str] = []
    seen: list[str] = []

    def replace(match: re.Match[str]) -> str:
        chunk_id = match.group(1)
        if chunk_id in by_id:
            if chunk_id not in seen:
                seen.append(chunk_id)
            return match.group(0)
        if chunk_id not in stripped:
            stripped.append(chunk_id)
        return ""

    cleaned = _CITATION_RE.sub(replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)

    citations = [
        Citation(
            chunk_id=chunk_id,
            title=by_id[chunk_id].title,
            version=by_id[chunk_id].version,
            effective_date=by_id[chunk_id].effective_date,
            authority_score=by_id[chunk_id].authority_score,
            acl_status="authorised",
            source_uri=by_id[chunk_id].source_uri,
        )
        for chunk_id in seen
    ]
    if stripped:
        logger.warning(
            "Stripped %d unsupported citation(s) from generated answer: %s",
            len(stripped),
            stripped,
        )
    return cleaned.strip(), citations, stripped


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token).

    Used for the cost column when the extractive generator runs and no real
    usage figures exist. It is an estimate and is labelled as one everywhere it
    is surfaced.
    """
    return max(1, len(text) // 4)


def build_generator(settings: Settings) -> Generator:
    """Return the Anthropic adapter when a key is configured, else extractive.

    The extractive generator is the default and the fallback: if the adapter
    cannot be constructed for any reason, generation degrades to deterministic
    extraction rather than failing the request. The active backend is reported
    on every answer, so a reader always knows which produced the text.
    """
    from .extractive import ExtractiveGenerator

    if not settings.anthropic_api_key:
        logger.info("No ANTHROPIC_API_KEY set; using deterministic extractive generator.")
        return ExtractiveGenerator()

    try:
        from .anthropic_adapter import AnthropicGenerator

        generator = AnthropicGenerator(settings)
        logger.info("Generation backend: %s", generator.name)
        return generator
    except (ImportError, RuntimeError) as exc:
        logger.warning(
            "Anthropic adapter unavailable (%s); falling back to extractive generator.",
            exc,
        )
        return ExtractiveGenerator()
