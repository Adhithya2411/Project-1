"""Audit logging.

What is recorded by default: who asked (user id and roles), a SHA-256 hash of
the query, the route chosen and every feature value behind it, the full
candidate-utility table, retrieval timings, the document IDs considered and the
document IDs actually used, how many chunks the ACL withheld, the citations
emitted, any abstention reason, the estimated cost, and a timestamp.

What is **not** recorded by default: the raw query text and any evidence text.
An audit log is itself an information channel — a log that captures a
restricted-finance question verbatim has moved that question into a table with
different access controls than the document it was about. Raw text is stored
only when ``AHRAG_VERBOSE_AUDIT=true``, which is an explicit local-debugging
opt-in and is flagged on every record it affects.

The query hash is salted with nothing and is therefore correlatable across
records by design: that is what makes "the same question was asked 40 times
this week" answerable without storing the question.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Sequence

from ..config import Settings
from ..db import Database
from ..models import (
    AbstentionReason,
    AuditRecord,
    AuthorisedScope,
    Route,
    RouteDecision,
    ScoredChunk,
    User,
)

logger = logging.getLogger(__name__)


class AuditLogger:
    """Builds and persists audit records for query executions."""

    def __init__(self, db: Database, settings: Settings) -> None:
        """Bind to the database and the verbosity setting."""
        self.db = db
        self.settings = settings

    @property
    def verbose(self) -> bool:
        """True when raw query text is being retained (debug opt-in)."""
        return bool(self.settings.verbose_audit)

    @staticmethod
    def hash_query(query: str) -> str:
        """Return the SHA-256 hex digest of a normalised query."""
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()

    def record(
        self,
        *,
        user: User,
        query: str,
        decision: RouteDecision,
        scope: AuthorisedScope,
        considered: Sequence[ScoredChunk],
        used: Sequence[ScoredChunk],
        citations: Sequence[str],
        timings_ms: dict[str, float],
        abstained: bool,
        abstention_reason: AbstentionReason,
        conflicts_disclosed: int,
        estimated_cost_usd: float,
        total_latency_s: float,
        generator: str,
    ) -> AuditRecord:
        """Build and persist one audit record.

        Returns:
            The persisted record. Persistence failures are logged and swallowed:
            an audit write must not take down a query that already completed
            correctly. The failure is visible in the application log.
        """
        record = AuditRecord(
            audit_id=uuid.uuid4().hex,
            timestamp=datetime.now(),
            user_id=user.user_id,
            user_roles=list(user.roles),
            query_hash=self.hash_query(query),
            query_text=query if self.verbose else None,
            route=decision.route,
            fallback_route=decision.fallback_route,
            router_version=decision.router_version,
            router_confidence=decision.confidence,
            features=self._feature_payload(decision),
            candidate_utilities=[u.model_dump(mode="json") for u in decision.utilities],
            timings_ms=timings_ms,
            # Document IDs, not chunk text: enough to reconstruct which sources
            # informed an answer, without copying their content into the log.
            documents_considered=sorted({item.chunk.doc_id for item in considered}),
            documents_used=sorted({item.chunk.doc_id for item in used}),
            acl_pool_size=len(scope.allowed_chunk_ids),
            acl_withheld_count=scope.withheld_count,
            citations=list(citations),
            abstained=abstained,
            abstention_reason=abstention_reason,
            conflicts_disclosed=conflicts_disclosed,
            estimated_cost_usd=estimated_cost_usd,
            total_latency_s=total_latency_s,
            generator=generator,
            verbose=self.verbose,
        )
        try:
            self.db.write_audit(record)
        except Exception as exc:  # noqa: BLE001 - audit must not break the request
            logger.error("Failed to persist audit record %s: %s", record.audit_id, exc)
        return record

    @staticmethod
    def _feature_payload(decision: RouteDecision) -> dict[str, object]:
        """Serialise router features for the log.

        The raw query is excluded at the model level (``RouterFeatures.raw_query``
        is ``exclude=True``), so this dump is safe under the default policy.
        """
        payload = decision.features.model_dump(mode="json")
        payload["reasons"] = decision.reasons
        payload["hard_constraints_applied"] = decision.hard_constraints_applied
        return payload

    def route_distribution(self, limit: int = 500) -> dict[str, int]:
        """Return counts of each route over the most recent ``limit`` records."""
        counts = {route.value: 0 for route in Route}
        for row in self.db.get_audit_records(limit=limit):
            route = str(row.get("route", ""))
            if route in counts:
                counts[route] += 1
        return counts
