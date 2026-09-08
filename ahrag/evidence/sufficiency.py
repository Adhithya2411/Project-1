"""The evidence-sufficiency gate.

This is the component that makes abstention a *decision* rather than a failure
mode. It sits between retrieval and generation, and the generator is never
invoked with a pack that failed it.

Checks applied, in order:

1. **ACL validation** — every chunk re-verified against the user's roles.
2. **Minimum top score** — the best chunk must clear a relevance floor.
3. **Minimum mean score** — the pack as a whole must be relevant, not one lucky hit.
4. **Minimum supporting chunks** — enough material to ground an answer.
5. **Source diversity** — comparison and summarisation intents require evidence
   from more than one parent document, because a "comparison" grounded in one
   source is not a comparison.
6. **Authority floor** — policy questions require a source at or above the
   configured authority score.
7. **Freshness** — for freshness-sensitive queries, the top-ranked source must be
   the current effective version of its family.

A failure produces a specific :class:`AbstentionReason`, which is what the
evaluation harness scores for abstention *appropriateness* — a system that
abstains for the wrong reason is not doing the right thing by accident.
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..config import EvidenceParams
from ..governance.acl import AccessControl
from ..governance.freshness import FreshnessPolicy
from ..models import (
    AbstentionReason,
    Chunk,
    DocType,
    Intent,
    RouterFeatures,
    ScoredChunk,
    SufficiencyReport,
    User,
)

logger = logging.getLogger(__name__)


class EvidenceValidator:
    """Applies the sufficiency gate and packs the final evidence set."""

    def __init__(
        self,
        params: EvidenceParams,
        acl: AccessControl,
        freshness: FreshnessPolicy,
    ) -> None:
        """Bind thresholds, the ACL enforcer, and the freshness policy."""
        self.params = params
        self.acl = acl
        self.freshness = freshness

    def pack(
        self,
        candidates: Sequence[ScoredChunk],
        user: User,
        top_k: int,
        freshness_sensitive: bool,
    ) -> list[ScoredChunk]:
        """Select the final evidence pack from reranked candidates.

        Applies freshness and authority multipliers, then drops any superseded
        chunk whose superseding document is *also* present — showing both
        versions of the same clause adds noise without adding information, and
        the conflict detector will still disclose the version split from the
        pack that remains.

        Args:
            candidates: Reranked, ACL-scoped candidates.
            user: Requesting user, for the authorisation re-check.
            top_k: Maximum chunks to pack.
            freshness_sensitive: Whether the query needs current sources. When
                True the freshness multiplier is applied; when False, ordering
                is left to relevance so that a historical question can still
                surface the retired policy that actually answers it.

        Returns:
            The packed evidence, best first.
        """
        authorised = self.acl.filter_scored(candidates, user)
        self.acl.assert_authorised(authorised, user, stage="evidence-packing")

        # Drop candidates the reranker scored at zero. They match nothing in the
        # query and only exist because the first-stage retriever had spare
        # slots. Packing them would dilute the mean-score check into
        # meaninglessness and pad the answer with irrelevant citations.
        authorised = [item for item in authorised if item.score > 0.0]

        adjusted = (
            self.freshness.apply(authorised) if freshness_sensitive else list(authorised)
        )

        if self.params.demote_superseded:
            present_docs = {item.chunk.doc_id for item in adjusted}
            adjusted = [
                item
                for item in adjusted
                if not (
                    item.chunk.is_superseded
                    and item.chunk.superseded_by in present_docs
                    # Keep the retired version when the query is *about* the
                    # difference; the conflict path needs both sides.
                    and not freshness_sensitive
                )
            ]

        return adjusted[:top_k]

    def validate(
        self,
        evidence: Sequence[ScoredChunk],
        user: User,
        features: RouterFeatures,
    ) -> SufficiencyReport:
        """Run every sufficiency check and return a structured verdict.

        Args:
            evidence: The packed evidence.
            user: Requesting user.
            features: Router features, supplying intent and freshness need.

        Returns:
            A :class:`SufficiencyReport`. ``sufficient=False`` means generation
            must not proceed; ``failure_reason`` names the first failed check.
        """
        checks: dict[str, bool] = {}
        details: dict[str, object] = {}

        # 1. ACL — re-verified here even though retrieval already filtered.
        roles = user.role_set
        unauthorised = [
            item.chunk_id for item in evidence if not item.chunk.readable_by(roles)
        ]
        checks["acl_validated"] = not unauthorised
        details["unauthorised_chunk_count"] = len(unauthorised)
        if unauthorised:
            logger.error(
                "Unauthorised chunks reached the sufficiency gate: %s", unauthorised
            )
            return SufficiencyReport(
                sufficient=False,
                checks=checks,
                details=details,
                failure_reason=AbstentionReason.NO_AUTHORISED_EVIDENCE,
                message=(
                    "Evidence failed access-control validation. No answer will be "
                    "generated."
                ),
            )

        if not evidence:
            checks["has_evidence"] = False
            return SufficiencyReport(
                sufficient=False,
                checks=checks,
                details=details,
                failure_reason=AbstentionReason.NO_AUTHORISED_EVIDENCE,
                message=(
                    "No authorised evidence was found for this question within "
                    "your access scope."
                ),
            )
        checks["has_evidence"] = True

        scores = [item.score for item in evidence]
        top_score = max(scores)
        mean_score = sum(scores) / len(scores)
        details["top_score"] = round(top_score, 4)
        details["mean_score"] = round(mean_score, 4)
        details["chunk_count"] = len(evidence)

        # 2/3. Relevance floors.
        checks["min_top_score"] = top_score >= self.params.min_top_score
        checks["min_mean_score"] = mean_score >= self.params.min_mean_score
        if not checks["min_top_score"]:
            return SufficiencyReport(
                sufficient=False,
                checks=checks,
                details=details,
                failure_reason=AbstentionReason.LOW_RELEVANCE,
                message=(
                    f"The best authorised passage scored {top_score:.2f}, below the "
                    f"{self.params.min_top_score:.2f} relevance threshold required "
                    "to ground an answer."
                ),
            )
        if not checks["min_mean_score"]:
            return SufficiencyReport(
                sufficient=False,
                checks=checks,
                details=details,
                failure_reason=AbstentionReason.LOW_RELEVANCE,
                message=(
                    f"The evidence pack averaged {mean_score:.2f}, below the "
                    f"{self.params.min_mean_score:.2f} threshold. One passage is "
                    "marginally on topic but the pack as a whole is not, which is "
                    "the signature of a question this corpus does not answer."
                ),
            )

        # 4. Enough supporting material.
        checks["min_supporting_chunks"] = (
            len(evidence) >= self.params.min_supporting_chunks
        )
        if not checks["min_supporting_chunks"]:
            return SufficiencyReport(
                sufficient=False,
                checks=checks,
                details=details,
                failure_reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                message=(
                    f"Only {len(evidence)} authorised passage(s) were found; at least "
                    f"{self.params.min_supporting_chunks} are required."
                ),
            )

        # 5. Source diversity where the intent demands it.
        distinct_docs = {item.chunk.doc_id for item in evidence}
        details["distinct_documents"] = len(distinct_docs)
        diversity_needed = features.intent.value in self.params.diversity_required_intents
        checks["source_diversity"] = (
            not diversity_needed
            or len(distinct_docs) >= self.params.min_distinct_documents
        )
        if not checks["source_diversity"]:
            return SufficiencyReport(
                sufficient=False,
                checks=checks,
                details=details,
                failure_reason=AbstentionReason.INSUFFICIENT_DIVERSITY,
                message=(
                    f"A {features.intent.value} question needs evidence from at least "
                    f"{self.params.min_distinct_documents} distinct authorised "
                    f"documents, but only {len(distinct_docs)} was available."
                ),
            )

        # 6. Authority floor for policy questions.
        policy_chunks = [
            item for item in evidence if item.chunk.doc_type is DocType.POLICY
        ]
        best_authority = max(
            (item.chunk.authority_score for item in policy_chunks), default=5
        )
        details["best_policy_authority"] = best_authority
        checks["authority_floor"] = (
            not policy_chunks or best_authority >= self.params.min_authority_for_policy
        )
        if not checks["authority_floor"]:
            return SufficiencyReport(
                sufficient=False,
                checks=checks,
                details=details,
                failure_reason=AbstentionReason.LOW_AUTHORITY,
                message=(
                    f"The only authorised policy evidence has authority score "
                    f"{best_authority}/5, below the {self.params.min_authority_for_policy} "
                    "required for a policy answer."
                ),
            )

        # 7. Freshness. A failure here is a warning, not an abstention: the
        # evidence is real and authorised, it is simply not current, and the
        # right response is to answer while disclosing that, not to refuse.
        top_chunk: Chunk = evidence[0].chunk
        freshness_ok = True
        if features.freshness_required and self.params.enforce_current_version_on_freshness:
            freshness_ok = self.freshness.is_current(top_chunk)
        checks["freshness"] = freshness_ok
        details["top_source_is_current"] = self.freshness.is_current(top_chunk)

        # 8. Ambiguity: a very short, vague query with weak, scattered evidence
        # should be clarified rather than answered from a guess.
        ambiguous = (
            features.query_length_tokens <= 4
            and features.semantic_ambiguity >= 0.6
            and top_score < self.params.min_top_score * 2.0
        )
        checks["not_ambiguous"] = not ambiguous
        if ambiguous:
            return SufficiencyReport(
                sufficient=False,
                checks=checks,
                details=details,
                failure_reason=AbstentionReason.AMBIGUOUS_NEEDS_CLARIFICATION,
                message=(
                    "The question is short and ambiguous, and the authorised "
                    "evidence is not clearly on-topic. A clarifying question is "
                    "safer than a guess."
                ),
            )

        return SufficiencyReport(
            sufficient=True,
            checks=checks,
            details=details,
            failure_reason=AbstentionReason.NONE,
            message="Evidence passed all sufficiency checks.",
        )

    def clarifying_question(
        self, features: RouterFeatures, evidence: Sequence[ScoredChunk]
    ) -> str:
        """Compose a clarifying question that does not leak restricted content.

        Suggestions are drawn only from evidence already inside the user's
        scope. When there is none, the question stays generic — never "did you
        mean the Q3 forecast?", which would reveal a document the user cannot
        read.
        """
        if evidence:
            titles = sorted({item.chunk.title for item in evidence})[:3]
            listed = "; ".join(titles)
            return (
                "Could you narrow the question? Within the sources you can access, "
                f"the closest material is: {listed}. Which of these did you mean, "
                "or can you add a date, document name, or identifier?"
            )
        if features.identifiers_found:
            return (
                f"I could not find {', '.join(features.identifiers_found)} in the "
                "sources you are authorised to read. Could you confirm the "
                "identifier, or the system it belongs to?"
            )
        return (
            "I could not find authorised evidence for that question. Could you "
            "rephrase it, or tell me which document or system it relates to?"
        )
