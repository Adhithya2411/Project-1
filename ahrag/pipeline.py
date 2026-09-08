"""The AHRAG engine: end-to-end orchestration of one query.

Execution order, and the reason for it:

1. Resolve the user and compute the **authorised scope** — ACL first, before
   anything else touches the corpus.
2. Normalise the query against conversation history.
3. Run the ACL-scoped **probe** to get first-pass sparse/dense confidence.
4. Extract **router features** from the query, the scope, and the probe.
5. **Route**, applying hard governance constraints before utility maximisation.
6. **Retrieve** under the chosen route, restricted to the authorised pool.
7. **Pack** evidence with freshness and authority applied.
8. **Detect conflicts** and compose freshness warnings.
9. **Validate sufficiency**; abstain or clarify if it fails.
10. **Generate** from the evidence pack only, then verify every citation.
11. **Audit** the whole execution without persisting raw text.

Steps 1 and 5 are what the central claim rests on: authorisation bounds the
candidate set before route selection, so route choice is constrained by
authorised scope rather than merely informed by it.

The engine holds no module-level state. Construct one, or construct several
with different routers — which is exactly what the evaluation harness does.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date
from typing import Sequence

from .audit.logger import AuditLogger
from .config import RouterConfig, Settings, get_settings
from .db import Database
from .evidence.conflict import ConflictDetector
from .evidence.sufficiency import EvidenceValidator
from .generation.base import Generator, build_generator
from .governance.acl import AccessControl
from .governance.freshness import FreshnessPolicy
from .index.store import IndexBundle
from .ingestion.pipeline import IngestionResult, IngestionService
from .models import (
    AbstentionReason,
    AnswerResult,
    AuthorisedScope,
    Citation,
    EvidenceItem,
    Route,
    RouteDecision,
    RetrievalTrace,
    ScoredChunk,
    SufficiencyReport,
    User,
)
from .retrieval.pipeline import RetrievalEngine
from .retrieval.rerank import build_reranker, normalise_query
from .routing.features import FeatureExtractor
from .routing.router import GovernanceAwareRouter, Router

logger = logging.getLogger(__name__)


class UnknownUserError(KeyError):
    """Raised when a query names a user that does not exist."""


class AHRAGEngine:
    """Owns the corpus, indexes, router, and answer pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        config: RouterConfig | None = None,
        db: Database | None = None,
        router: Router | None = None,
        generator: Generator | None = None,
        today: date | None = None,
    ) -> None:
        """Construct an engine.

        Args:
            settings: Operational settings. Defaults to the process settings.
            config: Routing policy. Defaults to ``settings.router_config``.
            db: Database. Defaults to one at ``settings.db_path``.
            router: Routing policy object. Defaults to the governance-aware
                router; the evaluation harness injects the baselines here.
            generator: Generation backend. Defaults to Anthropic-if-configured,
                otherwise extractive.
            today: Fixed reference date for freshness comparisons. Supplying
                one makes evaluation runs reproducible as the real date moves.
        """
        self.settings = settings or get_settings()
        self.config = config or RouterConfig.load(self.settings.router_config)
        self.db = db or Database(self.settings.db_path)

        self.acl = AccessControl([])
        self.index = IndexBundle(self.settings)
        self.freshness = FreshnessPolicy(self.config.evidence, today=today)
        self.features = FeatureExtractor(self.config)
        self.router: Router = router or GovernanceAwareRouter(self.config, self.settings)
        self.generator: Generator = generator or build_generator(self.settings)
        self.audit = AuditLogger(self.db, self.settings)
        self.conflicts = ConflictDetector(self.freshness)
        self.validator = EvidenceValidator(self.config.evidence, self.acl, self.freshness)
        self.ingestion = IngestionService(self.db, self.settings)

        self._reranker = build_reranker(
            self.settings.reranker, self.settings.cross_encoder_model, corpus=None
        )
        self.retrieval = RetrievalEngine(
            self.index, self.acl, self._reranker, self.config, self.settings
        )
        self.refresh_indexes()

    # -- corpus lifecycle ---------------------------------------------------

    def refresh_indexes(self) -> None:
        """Rebuild the ACL snapshot and both search indexes from the database."""
        chunks = self.db.get_chunks()
        self.acl.rebuild(chunks)
        self.index.build(chunks)
        # Refit the lexical reranker's IDF against the new corpus, if it uses one.
        fit = getattr(self._reranker, "fit", None)
        if callable(fit) and chunks:
            fit(chunks)
        logger.info("Engine refreshed: %d chunks indexed", len(chunks))

    def seed(
        self,
        reset: bool = True,
        manifest_path: str | Path | None = None,
    ) -> IngestionResult:
        """Seed a manifest-defined corpus and rebuild the indexes."""
        result = self.ingestion.seed_from_manifest(manifest_path, reset=reset)
        self.refresh_indexes()
        return result

    def ensure_seeded(self) -> IngestionResult | None:
        """Seed only if the corpus is empty. Returns None when already seeded."""
        if self.db.count_chunks() > 0:
            return None
        return self.seed(reset=False)

    def get_user(self, user_id: str) -> User:
        """Resolve a demo user.

        Raises:
            UnknownUserError: If the user does not exist.
        """
        user = self.db.get_user(user_id)
        if user is None:
            known = ", ".join(u.user_id for u in self.db.get_users()) or "none"
            raise UnknownUserError(f"Unknown user {user_id!r}. Known users: {known}")
        return user

    def backend_info(self) -> dict[str, object]:
        """Report which backends are active, for the health endpoint and UI."""
        info: dict[str, object] = dict(self.index.backend_info())
        info["reranker"] = getattr(self._reranker, "name", "unknown")
        info["generator"] = getattr(self.generator, "name", "unknown")
        info["router"] = getattr(self.router, "name", "unknown")
        info["router_version"] = self.config.router_version
        info["verbose_audit"] = self.settings.verbose_audit
        info["documents"] = len(self.db.get_documents())
        return info

    # -- query --------------------------------------------------------------

    def answer(
        self,
        query: str,
        user_id: str,
        history: Sequence[str] | None = None,
        write_audit: bool = True,
    ) -> AnswerResult:
        """Answer one query end to end.

        Args:
            query: The user's question.
            user_id: The selected demo user.
            history: Prior turns, most recent last, used for follow-up handling.
            write_audit: Whether to persist an audit record. The evaluation
                harness disables this so benchmark runs do not flood the log a
                reviewer is inspecting.

        Returns:
            A complete :class:`AnswerResult` including the route decision, the
            evidence pack, the sufficiency verdict, and any disclosed conflicts.

        Raises:
            UnknownUserError: If ``user_id`` is not a known demo user.
            ValueError: If ``query`` is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        started = time.perf_counter()
        timings: dict[str, float] = {}
        query_id = uuid.uuid4().hex[:12]
        user = self.get_user(user_id)

        # 1. ACL first — before routing, before retrieval, before anything.
        stage = time.perf_counter()
        scope = self.acl.scope_for(user)
        timings["acl_scope_ms"] = _ms_since(stage)

        # 2. Query normalisation and follow-up handling.
        normalised = normalise_query(query, list(history or []))

        # 3. ACL-scoped probe.
        stage = time.perf_counter()
        probe = self.retrieval.probe(normalised, scope)
        timings["probe_ms"] = _ms_since(stage)

        # 4. Features.
        stage = time.perf_counter()
        features = self.features.extract(
            normalised, user, scope, history=list(history or []), probe=probe
        )
        timings["features_ms"] = _ms_since(stage)

        # 5. Route.
        stage = time.perf_counter()
        decision = self.router.decide(features, scope)
        timings["routing_ms"] = _ms_since(stage)

        # 6. Retrieve.
        stage = time.perf_counter()
        candidates, trace = self.retrieval.retrieve(decision.route, normalised, user, scope)
        timings["retrieval_ms"] = _ms_since(stage)
        timings.update({f"retrieval.{k}": v for k, v in trace.timings_ms.items()})

        # 7. Pack evidence.
        stage = time.perf_counter()
        evidence = self.validator.pack(
            candidates,
            user,
            top_k=self.config.retrieval.evidence_top_k,
            freshness_sensitive=features.freshness_required,
        )
        timings["evidence_pack_ms"] = _ms_since(stage)

        # 8. Conflicts and freshness warnings.
        conflicts = self.conflicts.detect(evidence)
        freshness_warnings = self.freshness.warnings_for([e.chunk for e in evidence])

        # 9. Sufficiency gate.
        stage = time.perf_counter()
        sufficiency = self.validator.validate(evidence, user, features)
        timings["sufficiency_ms"] = _ms_since(stage)

        route_is_non_generative = (
            decision.route is Route.R0 and self.config.constraints.r0_is_non_generative
        )

        if route_is_non_generative or not sufficiency.sufficient:
            result = self._abstain(
                query_id=query_id,
                decision=decision,
                trace=trace,
                sufficiency=sufficiency,
                evidence=evidence,
                conflicts=conflicts,
                freshness_warnings=freshness_warnings,
                features_route_is_r0=route_is_non_generative,
            )
        else:
            # 10. Generate from evidence only.
            stage = time.perf_counter()
            generated = self.generator.generate(
                normalised, evidence, conflicts, freshness_warnings
            )
            timings["generation_ms"] = _ms_since(stage)
            result = AnswerResult(
                query_id=query_id,
                answer=generated.text,
                abstained=False,
                abstention_reason=AbstentionReason.NONE,
                citations=generated.citations,
                evidence=[_to_evidence_item(item) for item in evidence],
                decision=decision,
                trace=trace,
                sufficiency=sufficiency,
                conflicts=conflicts,
                freshness_warnings=freshness_warnings,
                generator=generated.generator,
                estimated_cost_usd=self.settings.estimate_cost_usd(
                    generated.input_tokens, generated.output_tokens
                ),
            )

        result.total_latency_s = round(time.perf_counter() - started, 4)
        timings["total_ms"] = round(result.total_latency_s * 1000, 3)

        # 11. Audit.
        if write_audit:
            record = self.audit.record(
                user=user,
                query=query,
                decision=decision,
                scope=scope,
                considered=candidates,
                used=evidence,
                citations=[c.chunk_id for c in result.citations],
                timings_ms=timings,
                abstained=result.abstained,
                abstention_reason=result.abstention_reason,
                conflicts_disclosed=len(result.conflicts),
                estimated_cost_usd=result.estimated_cost_usd,
                total_latency_s=result.total_latency_s,
                generator=result.generator,
            )
            result.audit_id = record.audit_id

        return result

    def _abstain(
        self,
        *,
        query_id: str,
        decision: RouteDecision,
        trace: RetrievalTrace,
        sufficiency: SufficiencyReport,
        evidence: Sequence[ScoredChunk],
        conflicts: Sequence,
        freshness_warnings: Sequence[str],
        features_route_is_r0: bool,
    ) -> AnswerResult:
        """Compose an abstention or clarification response.

        The message explains *why* no answer was produced, using only
        information inside the user's scope. It never names a withheld document
        and never hints that one exists — an abstention that leaks the existence
        of a restricted file has failed at the thing it was protecting.
        """
        reason = sufficiency.failure_reason
        if features_route_is_r0 and sufficiency.sufficient:
            reason = AbstentionReason.ROUTE_IS_NON_GENERATIVE

        clarifying: str | None = None
        if reason in {
            AbstentionReason.AMBIGUOUS_NEEDS_CLARIFICATION,
            AbstentionReason.LOW_RELEVANCE,
            AbstentionReason.ROUTE_IS_NON_GENERATIVE,
        }:
            clarifying = self.validator.clarifying_question(decision.features, evidence)

        lines = [
            "I can't answer that from the sources you're authorised to read.",
            "",
            sufficiency.message
            if not (features_route_is_r0 and sufficiency.sufficient)
            else (
                "The router selected R0, which may only clarify or abstain. It "
                "never answers an enterprise factual question from model memory."
            ),
        ]
        if clarifying:
            lines.extend(["", clarifying])

        # Evidence is still shown on abstention — it is authorised material, and
        # showing it lets the user judge the refusal instead of taking it on trust.
        return AnswerResult(
            query_id=query_id,
            answer="\n".join(lines),
            abstained=True,
            abstention_reason=reason,
            clarifying_question=clarifying,
            citations=[],
            evidence=[_to_evidence_item(item) for item in evidence],
            decision=decision,
            trace=trace,
            sufficiency=sufficiency,
            conflicts=list(conflicts),
            freshness_warnings=list(freshness_warnings),
            generator="abstention",
            estimated_cost_usd=0.0,
        )


def _to_evidence_item(item: ScoredChunk) -> EvidenceItem:
    """Convert a scored chunk into the user-facing evidence row."""
    chunk = item.chunk
    return EvidenceItem(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        title=chunk.title,
        source_uri=chunk.source_uri,
        doc_type=chunk.doc_type,
        owner=chunk.owner,
        version=chunk.version,
        effective_date=chunk.effective_date,
        created_date=chunk.created_date,
        authority_score=chunk.authority_score,
        score=round(item.score, 4),
        retriever=item.retriever,
        text=chunk.text,
        acl_status="authorised",
        acl_roles=list(chunk.acl_roles),
        is_superseded=chunk.is_superseded,
        superseded_by=chunk.superseded_by,
        policy_family=chunk.policy_family,
        subquery=item.subquery,
    )


def _ms_since(started: float) -> float:
    """Milliseconds elapsed since a ``perf_counter`` reading."""
    return round((time.perf_counter() - started) * 1000, 3)


__all__ = ["AHRAGEngine", "UnknownUserError", "AuthorisedScope", "Citation"]
