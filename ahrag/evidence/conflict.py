"""Contradiction detection across authorised sources.

Two detectors, both operating only on chunks the user is authorised to read:

**Version conflict** — two chunks from different documents in the same
``policy_family``. This is the structural case: a live policy and its retired
predecessor both matched the query.

**Numeric conflict** — the same quantity, expressed in the same unit, with
different values, in two chunks of the same family. This is the semantic case,
and it is the one that actually changes an answer: "22 days" versus "26 days"
is the difference between a correct and an incorrect response to an employee.

The design commitment is that conflict is **disclosed, never silently
resolved**. A preferred source is identified (current and higher-authority wins)
but the competing chunk stays in the evidence pack and in the rendered answer.
Silent resolution would be indistinguishable, from the user's side, from the
system simply not having found the other version.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

from ..governance.freshness import FreshnessPolicy
from ..models import Chunk, ConflictReport, ScoredChunk

# Quantity + unit, e.g. "26 days", "10 days", "£450", "20 working days".
_QUANTITY_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>"
    r"days?|working days?|calendar days?|weeks?|months?|years?|hours?|"
    r"percent|%|per cent|basis points?|miles?|nights?"
    r")\b",
    re.IGNORECASE,
)

# Which quantity a sentence is about: the nearest preceding topical noun.
_TOPIC_TERMS = (
    "leave",
    "annual leave",
    "carry over",
    "carry-over",
    "carried over",
    "entitlement",
    "accrue",
    "notice",
    "approval",
    "respond",
    "abroad",
    "attendance",
    "on site",
    "deadline",
    "submitted",
    "timeout",
    "allowance",
)


class ConflictDetector:
    """Finds and reports disagreements between authorised evidence chunks."""

    def __init__(self, freshness: FreshnessPolicy) -> None:
        """Bind the freshness policy used to choose the preferred source."""
        self.freshness = freshness

    def detect(self, items: Sequence[ScoredChunk]) -> list[ConflictReport]:
        """Return every conflict found in the evidence pack.

        Args:
            items: The ACL-filtered evidence pack. Chunks outside the user's
                scope must already be gone — a conflict report naming a document
                the user cannot read would itself be a disclosure.

        Returns:
            Version conflicts first, then numeric conflicts, each naming the
            chunks involved and the preferred source with its justification.
        """
        chunks = [item.chunk for item in items]
        return self._version_conflicts(chunks) + self._numeric_conflicts(chunks)

    def _version_conflicts(self, chunks: Sequence[Chunk]) -> list[ConflictReport]:
        """Detect multiple document versions of the same policy family."""
        by_family: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.policy_family:
                by_family[chunk.policy_family].append(chunk)

        reports: list[ConflictReport] = []
        for family, family_chunks in sorted(by_family.items()):
            doc_ids = {c.doc_id for c in family_chunks}
            if len(doc_ids) < 2:
                continue
            preferred = self.freshness.preferred_in_family(family_chunks)
            versions = sorted(
                {
                    f"{c.title} v{c.version} (effective {c.effective_date.isoformat()})"
                    for c in family_chunks
                }
            )
            reports.append(
                ConflictReport(
                    policy_family=family,
                    kind="version",
                    description=(
                        "The evidence includes more than one version of this topic: "
                        + "; ".join(versions)
                        + ". Both are shown; the answer prefers the current, "
                        "higher-authority version but does not suppress the other."
                    ),
                    chunk_ids=sorted(c.chunk_id for c in family_chunks),
                    preferred_chunk_id=preferred.chunk_id if preferred else None,
                    preference_reason=(
                        self._preference_reason(preferred) if preferred else None
                    ),
                )
            )
        return reports

    def _numeric_conflicts(self, chunks: Sequence[Chunk]) -> list[ConflictReport]:
        """Detect the same quantity stated differently across family versions."""
        by_family: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.policy_family:
                by_family[chunk.policy_family].append(chunk)

        reports: list[ConflictReport] = []
        for family, family_chunks in sorted(by_family.items()):
            if len({c.doc_id for c in family_chunks}) < 2:
                continue

            # topic+unit -> {value -> [chunk_ids]}
            claims: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for chunk in family_chunks:
                for topic, unit, value in _extract_claims(chunk.text):
                    if chunk.chunk_id not in claims[(topic, unit)][value]:
                        claims[(topic, unit)][value].append(chunk.chunk_id)

            for (topic, unit), values in sorted(claims.items()):
                if len(values) < 2:
                    continue
                involved = sorted({cid for ids in values.values() for cid in ids})
                if len({_doc_of(cid) for cid in involved}) < 2:
                    continue
                involved_chunks = [c for c in family_chunks if c.chunk_id in involved]
                preferred = self.freshness.preferred_in_family(involved_chunks)
                rendering = "; ".join(
                    f"{value} {unit} ({', '.join(sorted(ids))})"
                    for value, ids in sorted(values.items())
                )
                reports.append(
                    ConflictReport(
                        policy_family=family,
                        kind="numeric",
                        description=(
                            f"Authorised sources state different values for "
                            f"'{topic}': {rendering}. This disagreement is "
                            "disclosed rather than resolved silently."
                        ),
                        chunk_ids=involved,
                        preferred_chunk_id=preferred.chunk_id if preferred else None,
                        preference_reason=(
                            self._preference_reason(preferred) if preferred else None
                        ),
                    )
                )
        return reports

    def _preference_reason(self, chunk: Chunk) -> str:
        """Explain why ``chunk``'s source is preferred."""
        parts = [
            f"'{chunk.title}' v{chunk.version} takes effect "
            f"{chunk.effective_date.isoformat()}"
        ]
        if not chunk.is_superseded:
            parts.append("is not superseded")
        parts.append(f"authority score {chunk.authority_score}/5")
        return ", ".join(parts) + "."


def _extract_claims(text: str) -> list[tuple[str, str, str]]:
    """Extract ``(topic, unit, value)`` triples from a chunk's sentences.

    Topic is the nearest topical term in the same sentence. Sentences with no
    recognised topic are skipped rather than bucketed under a catch-all: an
    unlabelled "5" colliding with an unlabelled "10" would produce a false
    conflict report, and a false conflict warning erodes trust in the true ones.
    """
    claims: list[tuple[str, str, str]] = []
    for sentence in re.split(r"(?<=[.;:!?])\s+|\n", text):
        lowered = sentence.lower()
        topics = [term for term in _TOPIC_TERMS if term in lowered]
        if not topics:
            continue
        topic = max(topics, key=len)
        for match in _QUANTITY_RE.finditer(sentence):
            unit = _canonical_unit(match.group("unit"))
            value = match.group("value").rstrip("0").rstrip(".") or "0"
            claims.append((topic, unit, value))
    return claims


def _canonical_unit(unit: str) -> str:
    """Normalise unit spelling so '26 days' and '26 day' compare equal."""
    lowered = unit.lower().strip()
    lowered = re.sub(r"^(working|calendar)\s+", "", lowered)
    if lowered in {"%", "per cent"}:
        return "percent"
    return lowered.rstrip("s")


def _doc_of(chunk_id: str) -> str:
    """Recover the parent document ID from a ``{doc_id}::c{n}`` chunk ID."""
    return chunk_id.split("::", 1)[0]
