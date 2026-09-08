"""Optional Anthropic generation adapter.

Active only when ``ANTHROPIC_API_KEY`` is set *and* the ``anthropic`` package is
installed. It changes how the answer is *worded*; it changes nothing about what
evidence is admissible. The ACL pre-filter, the sufficiency gate, and the
citation verifier all run identically either way, and the adapter is never
reached for a query that failed those gates.

Any API failure degrades to the extractive generator rather than surfacing an
error, so enabling an API key can improve fluency but cannot take the prototype
offline.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..config import Settings
from ..models import ConflictReport, ScoredChunk
from .base import GeneratedAnswer, estimate_tokens, verify_citations
from .prompts import EVIDENCE_ONLY_SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class AnthropicGenerator:
    """Generates grounded answers with the Anthropic Messages API."""

    def __init__(self, settings: Settings) -> None:
        """Create the client.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
            RuntimeError: If no API key is configured or the client cannot be
                created.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is not installed; "
                "pip install -r requirements-optional.txt"
            ) from exc

        api_key = settings.anthropic_api_key
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")

        try:
            self._client = anthropic.Anthropic(api_key=api_key)
        except Exception as exc:  # pragma: no cover - client construction
            raise RuntimeError(f"Could not create Anthropic client: {exc}") from exc

        self.settings = settings
        self.model = settings.anthropic_model
        self.name = f"anthropic:{self.model}"

    def generate(
        self,
        query: str,
        evidence: Sequence[ScoredChunk],
        conflicts: Sequence[ConflictReport] = (),
        freshness_warnings: Sequence[str] = (),
    ) -> GeneratedAnswer:
        """Generate a cited answer, falling back to extraction on any failure."""
        if not evidence:
            return GeneratedAnswer(
                text="No authorised evidence was available, so no answer can be grounded.",
                generator=self.name,
            )

        prompt = build_user_prompt(query, evidence, conflicts, freshness_warnings)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.settings.generation_max_tokens,
                system=EVIDENCE_ONLY_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            input_tokens = getattr(response.usage, "input_tokens", estimate_tokens(prompt))
            output_tokens = getattr(response.usage, "output_tokens", estimate_tokens(text))
        except Exception as exc:  # noqa: BLE001 - any API failure must degrade, not fail
            logger.warning(
                "Anthropic generation failed (%s); falling back to extractive generator.",
                exc,
            )
            from .extractive import ExtractiveGenerator

            fallback = ExtractiveGenerator().generate(
                query, evidence, conflicts, freshness_warnings
            )
            fallback.generator = f"{self.name} (failed -> extractive fallback)"
            return fallback

        cleaned, citations, stripped = verify_citations(text, evidence)
        return GeneratedAnswer(
            text=cleaned,
            citations=citations,
            generator=self.name,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            stripped_citations=stripped,
        )
