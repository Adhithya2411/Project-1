"""Structure-aware chunking with stable, human-readable chunk identifiers.

Chunk boundaries follow document structure (Markdown/numbered headings, then
paragraphs) rather than a fixed character window, so that a policy clause or an
error-code section tends to stay intact. Keeping clauses intact matters for this
prototype specifically: the sparse route's whole value proposition is that an
exact identifier and its remediation text land in the *same* retrievable span.

Chunk IDs are ``{doc_id}::c{ordinal:03d}``. They are stable across re-ingestion
of unchanged content and short enough to render inline as ``[chunk_id]``
citations, which is what the generator emits.
"""

from __future__ import annotations

import re
from datetime import date

from ..models import Chunk, DocumentMeta

# A heading is: a Markdown ATX heading, a SHORT numbered line, or a SHORT
# all-caps line.
#
# The length bound on the numbered form is load-bearing. Enterprise runbooks and
# policies are full of numbered *procedure steps* ("2. If p99 latency exceeds
# 6000 ms, raise severity to SEV-2 and page the acquirer liaison."), which are
# lexically identical to numbered headings. Without the bound, every step of
# every procedure starts its own chunk, and a multi-step remediation is
# shattered across chunks that each retrieve poorly and answer nothing.
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"\#{1,6}\s+\S.*"
    r"|\d+(?:\.\d+)*\.?\s+[A-Z][^.!?]{0,58}"
    r"|[A-Z][A-Z \-/&]{6,58}"
    r")\s*$"
)


def chunk_document(
    text: str,
    meta: DocumentMeta,
    target_chars: int = 900,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Split ``text`` into governance-annotated chunks.

    Args:
        text: Full document text.
        meta: Parent document metadata, denormalised onto every chunk.
        target_chars: Soft upper bound on chunk size. Sections shorter than
            this are kept whole; longer sections are split on paragraph
            boundaries with overlap.
        overlap_chars: Characters of trailing context repeated at the start of
            each continuation chunk, so a fact split across a boundary is still
            retrievable from at least one chunk.

    Returns:
        Chunks in document order, each carrying the parent's ACL roles,
        version, effective date, and authority score.

    Raises:
        ValueError: If ``target_chars`` is not positive or ``overlap_chars`` is
            negative or not smaller than ``target_chars``.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be >= 0 and < target_chars")

    sections = _split_sections(text)
    pieces: list[tuple[str | None, str, int, int]] = []
    for heading, body, start in sections:
        for sub_text, sub_start in _split_body(body, start, target_chars, overlap_chars):
            if sub_text.strip():
                pieces.append((heading, sub_text, sub_start, sub_start + len(sub_text)))

    chunks: list[Chunk] = []
    for ordinal, (heading, body, char_start, char_end) in enumerate(pieces):
        payload = f"{heading}\n\n{body}".strip() if heading else body.strip()
        chunks.append(
            Chunk(
                chunk_id=f"{meta.doc_id}::c{ordinal:03d}",
                doc_id=meta.doc_id,
                ordinal=ordinal,
                text=payload,
                heading=heading,
                char_start=char_start,
                char_end=char_end,
                title=meta.title,
                source_uri=meta.source_uri,
                doc_type=meta.doc_type,
                owner=meta.owner,
                acl_roles=list(meta.acl_roles),
                created_date=meta.created_date,
                effective_date=meta.effective_date,
                version=meta.version,
                authority_score=meta.authority_score,
                policy_family=meta.policy_family,
                supersedes=meta.supersedes,
                superseded_by=meta.superseded_by,
            )
        )
    return chunks


def _split_sections(text: str) -> list[tuple[str | None, str, int]]:
    """Group lines into ``(heading, body, char_offset)`` sections."""
    lines = text.split("\n")
    sections: list[tuple[str | None, list[str], int]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    current_start = 0
    offset = 0

    for line in lines:
        line_len = len(line) + 1
        if _HEADING_RE.match(line) and line.strip():
            if current_lines and any(chunk.strip() for chunk in current_lines):
                sections.append((current_heading, current_lines, current_start))
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            current_lines = []
            current_start = offset + line_len
        else:
            if not current_lines:
                current_start = offset
            current_lines.append(line)
        offset += line_len

    if current_lines and any(chunk.strip() for chunk in current_lines):
        sections.append((current_heading, current_lines, current_start))
    if not sections:
        sections = [(None, lines, 0)]
    return [(h, "\n".join(body).strip(), start) for h, body, start in sections]


def _split_body(
    body: str, start: int, target_chars: int, overlap_chars: int
) -> list[tuple[str, int]]:
    """Split one section body into pieces no larger than ``target_chars``."""
    if len(body) <= target_chars:
        return [(body, start)]

    paragraphs = re.split(r"\n\s*\n", body)
    pieces: list[tuple[str, int]] = []
    buffer = ""
    buffer_start = start
    cursor = start

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) > target_chars and buffer:
            pieces.append((buffer, buffer_start))
            tail = buffer[-overlap_chars:] if overlap_chars else ""
            buffer = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
            buffer_start = max(start, cursor - len(tail))
        else:
            if not buffer:
                buffer_start = cursor
            buffer = candidate
        cursor += len(paragraph) + 2

    if buffer.strip():
        pieces.append((buffer, buffer_start))

    # A single paragraph longer than the target still needs hard splitting.
    expanded: list[tuple[str, int]] = []
    for piece, piece_start in pieces:
        if len(piece) <= target_chars * 1.6:
            expanded.append((piece, piece_start))
            continue
        step = target_chars - overlap_chars
        for index in range(0, len(piece), step):
            window = piece[index : index + target_chars]
            if window.strip():
                expanded.append((window, piece_start + index))
    return expanded


def resolve_supersession(metas: list[DocumentMeta]) -> list[DocumentMeta]:
    """Fill in ``superseded_by`` from the declared ``supersedes`` links.

    Only one direction is declared in the manifest; the reverse pointer is
    derived here so that every chunk of a retired document carries its own
    retirement marker. That marker is what the freshness gate and the
    conflict detector key on, and it must be present on the *chunk* rather than
    require a lookup, so it cannot be forgotten downstream.
    """
    by_id = {meta.doc_id: meta for meta in metas}
    updated: dict[str, DocumentMeta] = {m.doc_id: m.model_copy(deep=True) for m in metas}
    for meta in metas:
        if meta.supersedes and meta.supersedes in by_id:
            older = updated[meta.supersedes]
            updated[meta.supersedes] = older.model_copy(
                update={"superseded_by": meta.doc_id}
            )
    return [updated[m.doc_id] for m in metas]


def current_version_of(metas: list[DocumentMeta], policy_family: str) -> DocumentMeta | None:
    """Return the effective current document for ``policy_family``.

    "Current" means: not superseded, and among those the latest effective date,
    breaking ties on authority score. Returns None when the family is unknown.
    """
    family = [m for m in metas if m.policy_family == policy_family]
    if not family:
        return None
    live = [m for m in family if not m.is_superseded] or family
    return max(live, key=lambda m: (m.effective_date, m.authority_score))


def today() -> date:
    """Return today's date. Indirected so tests can patch a fixed clock."""
    return date.today()
