"""Domain models for AHRAG.

Every object that crosses a module boundary is defined here as a Pydantic model
so that the FastAPI layer, the Streamlit layer, the evaluation harness, and the
tests all agree on one schema. There is no hidden global state: engines are
constructed explicitly and models are passed by value.
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Route(str, enum.Enum):
    """The five retrieval strategies the router may select between.

    ``R0`` is structurally non-generative: it may clarify, explain the absence
    of authorised evidence, or abstain. It may never answer an enterprise
    factual question from model memory.
    """

    R0 = "R0"  # clarification / evidence-based abstention
    R1 = "R1"  # sparse BM25 only
    R2 = "R2"  # dense vector only
    R3 = "R3"  # hybrid sparse+dense with RRF
    R4 = "R4"  # decomposed iterative hybrid

    @property
    def label(self) -> str:
        """Human-readable name for UI display."""
        return {
            Route.R0: "R0 · Clarify / Abstain",
            Route.R1: "R1 · Sparse (BM25)",
            Route.R2: "R2 · Dense (vector)",
            Route.R3: "R3 · Hybrid (RRF)",
            Route.R4: "R4 · Decomposed iterative hybrid",
        }[self]

    @property
    def is_generative(self) -> bool:
        """True when the route is permitted to produce a factual answer."""
        return self is not Route.R0


class Intent(str, enum.Enum):
    """Coarse query intent used as an interpretable routing feature."""

    LOOKUP = "lookup"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    SUMMARISATION = "summarisation"
    PROCEDURAL = "procedural"
    TEMPORAL = "temporal"


class DocType(str, enum.Enum):
    """Document taxonomy of the seeded enterprise corpus."""

    POLICY = "policy"
    RUNBOOK = "runbook"
    REPORT = "report"
    FINANCE = "finance"
    WIKI = "wiki"
    OTHER = "other"


class AbstentionReason(str, enum.Enum):
    """Why the system declined to answer. Recorded in the audit log."""

    NONE = "none"
    NO_AUTHORISED_EVIDENCE = "no_authorised_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LOW_RELEVANCE = "low_relevance"
    INSUFFICIENT_DIVERSITY = "insufficient_diversity"
    LOW_AUTHORITY = "low_authority"
    AMBIGUOUS_NEEDS_CLARIFICATION = "ambiguous_needs_clarification"
    ROUTE_IS_NON_GENERATIVE = "route_is_non_generative"


# ---------------------------------------------------------------------------
# Corpus / governance
# ---------------------------------------------------------------------------


class DocumentMeta(BaseModel):
    """Metadata for one source document in the enterprise corpus."""

    doc_id: str = Field(description="Stable parent-document identifier.")
    title: str
    source_uri: str = Field(description="Path or URI the document was ingested from.")
    doc_type: DocType = DocType.OTHER
    owner: str = Field(description="Accountable owner (team or role), not a person.")
    acl_roles: list[str] = Field(
        default_factory=list,
        description="Roles permitted to read this document. Empty means nobody.",
    )
    created_date: date
    effective_date: date = Field(
        description="Date from which this version is authoritative."
    )
    version: str = "1.0"
    authority_score: int = Field(
        default=3, ge=1, le=5, description="1 = informal note, 5 = ratified policy."
    )
    policy_family: str | None = Field(
        default=None,
        description=(
            "Groups versions of the same underlying policy/topic. Two documents "
            "sharing a family are candidates for version-conflict detection."
        ),
    )
    supersedes: str | None = Field(default=None, description="doc_id this replaces.")
    superseded_by: str | None = Field(default=None, description="doc_id replacing this.")
    checksum: str | None = None

    @field_validator("acl_roles")
    @classmethod
    def _normalise_roles(cls, value: list[str]) -> list[str]:
        return sorted({role.strip().lower() for role in value if role.strip()})

    @property
    def is_superseded(self) -> bool:
        """True when a newer version of this document exists."""
        return bool(self.superseded_by)


class Chunk(BaseModel):
    """A retrievable text span carrying its parent document's governance metadata.

    Governance metadata is denormalised onto the chunk deliberately: every stage
    downstream of retrieval (rerank, evidence packing, generation, citation
    rendering, audit) can then re-verify authorisation without a join, which
    makes the "never let an unauthorised chunk influence an answer" invariant
    cheap enough to assert repeatedly.
    """

    chunk_id: str = Field(description="Stable, content-derived chunk identifier.")
    doc_id: str
    ordinal: int = Field(description="0-based position of the chunk in its document.")
    text: str
    heading: str | None = None
    char_start: int = 0
    char_end: int = 0

    # Denormalised governance / provenance metadata.
    title: str
    source_uri: str
    doc_type: DocType
    owner: str
    acl_roles: list[str]
    created_date: date
    effective_date: date
    version: str
    authority_score: int
    policy_family: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None

    @property
    def is_superseded(self) -> bool:
        """True when the parent document has been replaced by a newer version."""
        return bool(self.superseded_by)

    def readable_by(self, roles: set[str]) -> bool:
        """Return True when any of ``roles`` grants read access to this chunk."""
        return bool(set(self.acl_roles) & {r.lower() for r in roles})

    def citation_label(self) -> str:
        """Short provenance string used in the UI and in generated citations."""
        return f"{self.title} v{self.version} (eff. {self.effective_date.isoformat()})"


class User(BaseModel):
    """A demo user. Authentication is out of scope; selection stands in for it."""

    user_id: str
    display_name: str
    roles: list[str]
    description: str = ""

    @field_validator("roles")
    @classmethod
    def _normalise(cls, value: list[str]) -> list[str]:
        return sorted({r.strip().lower() for r in value if r.strip()})

    @property
    def role_set(self) -> set[str]:
        """Roles as a set, lowercased."""
        return set(self.roles)


class AuthorisedScope(BaseModel):
    """The result of applying ACLs *before* any retrieval or utility evaluation.

    ``allowed_chunk_ids`` is the only pool downstream retrieval is permitted to
    search. ``withheld_count`` is recorded for the audit trail and feeds the
    router's governance-risk term, but the withheld *content* never leaves this
    boundary.
    """

    user_id: str
    roles: list[str]
    allowed_chunk_ids: list[str]
    allowed_doc_ids: list[str]
    total_chunks: int
    withheld_count: int
    withheld_doc_types: list[str] = Field(
        default_factory=list,
        description=(
            "Document *types* withheld (never titles or text). Used to explain "
            "an abstention without leaking what was withheld."
        ),
    )

    @property
    def is_empty(self) -> bool:
        """True when the user may read nothing in the corpus."""
        return not self.allowed_chunk_ids

    @property
    def restricted_fraction(self) -> float:
        """Share of the corpus withheld from this user, in ``0..1``."""
        if self.total_chunks == 0:
            return 0.0
        return self.withheld_count / self.total_chunks


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class RouterFeatures(BaseModel):
    """Interpretable feature vector consumed by the governance-aware router.

    Every field is either a normalised ``0..1`` score or a categorical label, so
    the "Why this route?" panel can render the exact numbers that produced the
    decision. This is a hand-specified, auditable feature set -- not a learned
    representation. See ``RESEARCH_LIMITATIONS.md``.
    """

    # Surface form
    raw_query: str = Field(default="", exclude=True)
    query_length_tokens: int = 0
    query_length: float = Field(default=0.0, ge=0.0, le=1.0)

    # Lexical / structural cues
    identifier_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    identifiers_found: list[str] = Field(default_factory=list)
    numeric_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    temporal_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    lexical_specificity: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)
    mixed_signal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Both lexical-exact and semantic cues present -> favours fusion.",
    )

    # Reasoning shape
    intent: Intent = Intent.LOOKUP
    intent_scores: dict[str, float] = Field(default_factory=dict)
    comparison_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    hop_signal: float = Field(default=0.0, ge=0.0, le=1.0)
    likely_hop_count: int = 1
    followup_signal: float = Field(default=0.0, ge=0.0, le=1.0)

    # Governance context
    user_role: str = ""
    user_roles: list[str] = Field(default_factory=list)
    authorised_chunk_count: int = 0
    restricted_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_required: bool = False
    doc_type_hints: list[str] = Field(default_factory=list)
    conflict_likelihood: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Probe found >1 version of the same policy family.",
    )

    # First-pass retrieval confidence (ACL-scoped probe)
    sparse_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    dense_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    probe_agreement: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overlap between sparse and dense probe results.",
    )
    unsupported_signal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1 - best probe confidence; drives R0.",
    )

    def display_rows(self) -> list[tuple[str, Any]]:
        """Return ``(name, value)`` pairs for the UI feature table."""
        return [
            ("query_length_tokens", self.query_length_tokens),
            ("identifier_signal", round(self.identifier_signal, 3)),
            ("identifiers_found", ", ".join(self.identifiers_found) or "—"),
            ("numeric_signal", round(self.numeric_signal, 3)),
            ("temporal_signal", round(self.temporal_signal, 3)),
            ("lexical_specificity", round(self.lexical_specificity, 3)),
            ("semantic_ambiguity", round(self.semantic_ambiguity, 3)),
            ("mixed_signal", round(self.mixed_signal, 3)),
            ("intent", self.intent.value),
            ("comparison_signal", round(self.comparison_signal, 3)),
            ("hop_signal", round(self.hop_signal, 3)),
            ("likely_hop_count", self.likely_hop_count),
            ("followup_signal", round(self.followup_signal, 3)),
            ("user_role", self.user_role),
            ("authorised_chunks", self.authorised_chunk_count),
            ("restricted_fraction", round(self.restricted_fraction, 3)),
            ("freshness_required", self.freshness_required),
            ("doc_type_hints", ", ".join(self.doc_type_hints) or "—"),
            ("conflict_likelihood", round(self.conflict_likelihood, 3)),
            ("sparse_confidence", round(self.sparse_confidence, 3)),
            ("dense_confidence", round(self.dense_confidence, 3)),
            ("probe_agreement", round(self.probe_agreement, 3)),
            ("unsupported_signal", round(self.unsupported_signal, 3)),
        ]


class RouteUtility(BaseModel):
    """The full utility breakdown for one candidate route.

    Both admissible and inadmissible routes get a row so the UI can show *why*
    a route was rejected -- including rejection on hard governance grounds,
    where ``admissible`` is False and utility is not meaningful.
    """

    route: Route
    admissible: bool = True
    rejection_reason: str | None = None
    expected_evidence_quality: float = 0.0
    estimated_latency_s: float = 0.0
    estimated_cost_usd: float = 0.0
    estimated_risk: float = 0.0
    latency_penalty: float = 0.0
    cost_penalty: float = 0.0
    risk_penalty: float = 0.0
    utility: float = 0.0


class RouteDecision(BaseModel):
    """The router's output: a choice plus the evidence for that choice."""

    route: Route
    fallback_route: Route
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(
        default_factory=list, description="Plain-English justification, ordered."
    )
    utilities: list[RouteUtility] = Field(default_factory=list)
    features: RouterFeatures
    estimated_latency_s: float = 0.0
    estimated_cost_usd: float = 0.0
    estimated_evidence_risk: float = 0.0
    router_version: str = "unknown"
    hard_constraints_applied: list[str] = Field(default_factory=list)

    def utility_of(self, route: Route) -> RouteUtility | None:
        """Return the utility row for ``route``, or None if absent."""
        for row in self.utilities:
            if row.route is route:
                return row
        return None

    @property
    def rejected(self) -> list[RouteUtility]:
        """Utility rows for every route that was not chosen, best first."""
        others = [u for u in self.utilities if u.route is not self.route]
        return sorted(others, key=lambda u: (u.admissible, u.utility), reverse=True)


# ---------------------------------------------------------------------------
# Retrieval & evidence
# ---------------------------------------------------------------------------


class ScoredChunk(BaseModel):
    """A chunk with its retrieval provenance attached."""

    chunk: Chunk
    score: float = 0.0
    sparse_rank: int | None = None
    dense_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    retriever: str = "unknown"
    subquery: str | None = Field(
        default=None, description="For R4: which sub-query surfaced this chunk."
    )

    @property
    def chunk_id(self) -> str:
        """Convenience accessor for the underlying chunk id."""
        return self.chunk.chunk_id


class RetrievalTrace(BaseModel):
    """Timing and volume telemetry for one retrieval execution."""

    route: Route
    subqueries: list[str] = Field(default_factory=list)
    iterations: int = 1
    sparse_candidates: int = 0
    dense_candidates: int = 0
    fused_candidates: int = 0
    reranked_candidates: int = 0
    acl_scoped_pool: int = 0
    acl_withheld: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    per_iteration: list[dict[str, Any]] = Field(default_factory=list)


class ConflictReport(BaseModel):
    """A disclosed disagreement between authorised sources.

    AHRAG's policy is to *disclose* conflict, never to silently resolve it. The
    preferred chunk is identified, but the competing chunk stays in the evidence
    pack and in the rendered answer.
    """

    policy_family: str
    kind: str = Field(description="'version' or 'numeric'.")
    description: str
    chunk_ids: list[str]
    preferred_chunk_id: str | None = None
    preference_reason: str | None = None


class EvidenceItem(BaseModel):
    """One row of the evidence pack shown to the user and given to the generator."""

    chunk_id: str
    doc_id: str
    title: str
    source_uri: str
    doc_type: DocType
    owner: str
    version: str
    effective_date: date
    created_date: date
    authority_score: int
    score: float
    retriever: str
    text: str
    acl_status: str = Field(
        default="authorised",
        description="Always 'authorised' in a served pack; retained for audit clarity.",
    )
    acl_roles: list[str] = Field(default_factory=list)
    is_superseded: bool = False
    superseded_by: str | None = None
    policy_family: str | None = None
    subquery: str | None = None


class SufficiencyReport(BaseModel):
    """Outcome of the evidence-sufficiency gate that sits before generation."""

    sufficient: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    failure_reason: AbstentionReason = AbstentionReason.NONE
    message: str = ""


# ---------------------------------------------------------------------------
# Answers & audit
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A citation rendered as ``[chunk_id]`` with its provenance."""

    chunk_id: str
    title: str
    version: str
    effective_date: date
    authority_score: int
    acl_status: str = "authorised"
    source_uri: str = ""


class AnswerResult(BaseModel):
    """The complete, user-facing result of one query."""

    query_id: str
    answer: str
    abstained: bool = False
    abstention_reason: AbstentionReason = AbstentionReason.NONE
    clarifying_question: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    decision: RouteDecision
    trace: RetrievalTrace
    sufficiency: SufficiencyReport
    conflicts: list[ConflictReport] = Field(default_factory=list)
    freshness_warnings: list[str] = Field(default_factory=list)
    generator: str = "extractive"
    total_latency_s: float = 0.0
    estimated_cost_usd: float = 0.0
    audit_id: str | None = None


class AuditRecord(BaseModel):
    """A privacy-preserving record of one query execution.

    By default this stores a SHA-256 hash of the query and chunk *identifiers*
    only. Raw query text and raw evidence text are stored solely when
    ``AHRAG_VERBOSE_AUDIT=true`` is set for local debugging.
    """

    audit_id: str
    timestamp: datetime
    user_id: str
    user_roles: list[str]
    query_hash: str
    query_text: str | None = None
    route: Route
    fallback_route: Route
    router_version: str
    router_confidence: float
    features: dict[str, Any] = Field(default_factory=dict)
    candidate_utilities: list[dict[str, Any]] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    documents_considered: list[str] = Field(default_factory=list)
    documents_used: list[str] = Field(default_factory=list)
    acl_pool_size: int = 0
    acl_withheld_count: int = 0
    citations: list[str] = Field(default_factory=list)
    abstained: bool = False
    abstention_reason: AbstentionReason = AbstentionReason.NONE
    conflicts_disclosed: int = 0
    estimated_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    generator: str = "extractive"
    verbose: bool = False
