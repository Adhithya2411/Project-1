"""Prompts for the optional LLM generation adapter.

The prompt carries the same rules the deterministic generator enforces
structurally. That parity matters: when the Anthropic adapter is enabled, the
citation verifier in ``generation/base.py`` still runs, so an LLM that ignores
these instructions has its unsupported citations stripped rather than served.
The prompt is the first line of defence, not the only one.
"""

from __future__ import annotations

from typing import Sequence

from ..models import ConflictReport, ScoredChunk

EVIDENCE_ONLY_SYSTEM_PROMPT = """\
You are an enterprise knowledge assistant operating under strict evidence rules.

RULES — these are not stylistic preferences:

1. Answer ONLY from the numbered evidence passages provided in the user message.
   You have no other permitted source. Your own knowledge of policies, error
   codes, products, or organisations is not admissible evidence here, even if
   you are confident it is correct.
2. Cite every factual statement with the chunk id in square brackets, exactly as
   given, e.g. [doc-hr-leave-v2::c002]. A sentence stating a fact without a
   citation is a rule violation.
3. If the evidence does not support an answer, say so plainly and stop. Do not
   fill gaps by inference, analogy, or plausibility.
4. If the evidence contains conflicting versions of the same policy, disclose
   the conflict explicitly. State which version is current and why, and state
   what the other version says. Never silently pick one.
5. If a passage is marked SUPERSEDED, do not present it as current guidance.
   You may cite it for historical context, labelled as such.
6. Do not speculate about documents you cannot see, and do not mention that
   other documents might exist. The evidence you were given is the whole of
   what this user is authorised to read.
7. Be concise and factual. Prefer the exact wording of the source for figures,
   dates, identifiers, and thresholds.
"""


def build_user_prompt(
    query: str,
    evidence: Sequence[ScoredChunk],
    conflicts: Sequence[ConflictReport] = (),
    freshness_warnings: Sequence[str] = (),
) -> str:
    """Render the evidence pack and question into a user prompt.

    Args:
        query: The user's question.
        evidence: The validated, ACL-filtered evidence pack.
        conflicts: Conflicts the detector found, to be disclosed in the answer.
        freshness_warnings: Currency warnings for the packed sources.

    Returns:
        A prompt string in which every passage carries its chunk id, version,
        effective date, authority score, and supersession status — so the model
        has the provenance it needs to comply with rules 4 and 5.
    """
    lines: list[str] = ["EVIDENCE PASSAGES", ""]
    for index, item in enumerate(evidence, start=1):
        chunk = item.chunk
        status = "SUPERSEDED" if chunk.is_superseded else "CURRENT"
        lines.append(
            f"[{index}] chunk_id={chunk.chunk_id}\n"
            f"    source: {chunk.title} (v{chunk.version}, {chunk.doc_type.value})\n"
            f"    effective: {chunk.effective_date.isoformat()} | "
            f"authority: {chunk.authority_score}/5 | status: {status}\n"
            f"    owner: {chunk.owner} | relevance: {item.score:.3f}\n"
            f"    text: {chunk.text}\n"
        )

    if conflicts:
        lines.append("DETECTED CONFLICTS — you must disclose these:")
        for conflict in conflicts:
            lines.append(f"  - {conflict.description}")
            if conflict.preference_reason:
                lines.append(f"    Preferred source: {conflict.preference_reason}")
        lines.append("")

    if freshness_warnings:
        lines.append("FRESHNESS WARNINGS:")
        lines.extend(f"  - {warning}" for warning in freshness_warnings)
        lines.append("")

    lines.append(f"QUESTION: {query}")
    lines.append("")
    lines.append(
        "Answer using only the passages above, citing every factual statement "
        "with its [chunk_id]."
    )
    return "\n".join(lines)


ABSTENTION_TEMPLATE = """\
I can't answer that from the sources you're authorised to read.

{reason}

{suggestion}\
"""
