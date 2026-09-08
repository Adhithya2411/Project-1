"""Freshness and source-authority policy.

Two jobs:

1. Score how "current" a chunk is, so that freshness-sensitive queries prefer
   the live version of a policy over a retired one.
2. Decide the *preferred* source within a policy family, without discarding the
   competing version — AHRAG discloses conflict rather than resolving it
   silently, so the loser of the preference ordering stays in the evidence pack
   and is shown to the user.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

from ..config import EvidenceParams
from ..models import Chunk, ScoredChunk


class FreshnessPolicy:
    """Applies effective-date, supersession, and authority preferences."""

    def __init__(self, params: EvidenceParams, today: date | None = None) -> None:
        """Bind thresholds and (optionally) a fixed clock for reproducibility."""
        self.params = params
        self._today = today or date.today()

    @property
    def today(self) -> date:
        """The reference date used for all freshness comparisons."""
        return self._today

    # -- scoring -----------------------------------------------------------

    def freshness_multiplier(self, chunk: Chunk) -> float:
        """Return a score multiplier in ``(0, 1]`` reflecting currency.

        A superseded document is demoted by
        ``EvidenceParams.superseded_score_multiplier``. A document whose
        effective date is in the future is demoted equally: it is not yet
        authoritative, and treating a not-yet-live policy as current is the same
        class of error as treating a retired one as current.
        """
        multiplier = 1.0
        if self.params.demote_superseded and chunk.is_superseded:
            multiplier *= self.params.superseded_score_multiplier
        if chunk.effective_date > self._today:
            multiplier *= self.params.superseded_score_multiplier
        return multiplier

    def authority_multiplier(self, chunk: Chunk) -> float:
        """Return a mild score multiplier from the 1..5 authority score.

        Deliberately mild (0.9 to 1.1). Authority breaks ties between comparably
        relevant sources; it must not let a highly authoritative but irrelevant
        document outrank a directly responsive one.
        """
        return 0.9 + (chunk.authority_score - 1) * 0.05

    def apply(self, scored: Iterable[ScoredChunk]) -> list[ScoredChunk]:
        """Return copies of ``scored`` with freshness and authority applied."""
        out: list[ScoredChunk] = []
        for item in scored:
            factor = self.freshness_multiplier(item.chunk) * self.authority_multiplier(
                item.chunk
            )
            # Clamped to 1.0 so the adjusted score stays on the same absolute
            # scale the evidence-sufficiency thresholds are expressed in.
            adjusted = min(1.0, item.score * factor)
            out.append(item.model_copy(update={"score": adjusted}))
        return sorted(out, key=lambda s: s.score, reverse=True)

    # -- preference --------------------------------------------------------

    def preferred_in_family(self, chunks: Sequence[Chunk]) -> Chunk | None:
        """Return the chunk whose source should be treated as authoritative.

        Ordering: not-superseded first, then latest effective date that is not in
        the future, then highest authority score. Returns None for an empty
        input.
        """
        if not chunks:
            return None

        def key(chunk: Chunk) -> tuple[int, int, date, int]:
            return (
                0 if chunk.is_superseded else 1,
                0 if chunk.effective_date > self._today else 1,
                chunk.effective_date,
                chunk.authority_score,
            )

        return max(chunks, key=key)

    def is_current(self, chunk: Chunk) -> bool:
        """True when the chunk's source is live: not superseded, already effective."""
        return not chunk.is_superseded and chunk.effective_date <= self._today

    def warnings_for(self, chunks: Sequence[Chunk]) -> list[str]:
        """Return human-readable freshness warnings for an evidence pack.

        Warnings name the document, its version, and its effective date so the
        user can judge the evidence themselves rather than trusting the system's
        preference ordering.
        """
        warnings: list[str] = []
        for chunk in chunks:
            if chunk.is_superseded:
                warnings.append(
                    f"'{chunk.title}' v{chunk.version} (effective "
                    f"{chunk.effective_date.isoformat()}) has been superseded by a "
                    f"newer version ({chunk.superseded_by}). It is shown because it "
                    "is still authorised and relevant, not because it is current."
                )
            elif chunk.effective_date > self._today:
                warnings.append(
                    f"'{chunk.title}' v{chunk.version} does not take effect until "
                    f"{chunk.effective_date.isoformat()} and is not yet authoritative."
                )
        return warnings
