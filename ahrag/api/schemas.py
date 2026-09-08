"""Request and response schemas for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..models import AnswerResult, DocType


class QueryRequest(BaseModel):
    """A question asked as a particular demo user."""

    query: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1)
    history: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Prior turns, most recent last. Used for follow-up handling.",
    )


class QueryResponse(BaseModel):
    """The full answer payload, including the route decision and evidence."""

    result: AnswerResult


class UserSummary(BaseModel):
    """A demo user as shown in the role selector."""

    user_id: str
    display_name: str
    roles: list[str]
    description: str
    authorised_chunks: int
    total_chunks: int
    withheld_chunks: int


class DocumentSummary(BaseModel):
    """A corpus document as shown in the ingestion panel."""

    doc_id: str
    title: str
    doc_type: DocType
    owner: str
    acl_roles: list[str]
    version: str
    effective_date: str
    created_date: str
    authority_score: int
    policy_family: str | None
    supersedes: str | None
    superseded_by: str | None
    chunk_count: int
    source_uri: str


class IngestRequest(BaseModel):
    """Metadata accompanying an uploaded document."""

    title: str | None = None
    doc_type: DocType = DocType.OTHER
    owner: str = "unassigned"
    acl_roles: list[str] = Field(
        default_factory=list,
        description="Roles permitted to read. Empty defaults to a restrictive set.",
    )
    version: str = "1.0"
    authority_score: int = Field(default=3, ge=1, le=5)
    policy_family: str | None = None
    supersedes: str | None = None
    effective_date: str | None = None


class IngestResponse(BaseModel):
    """Result of ingesting one uploaded document."""

    doc_id: str
    title: str
    chunks: int
    acl_roles: list[str]
    warning: str | None = None


class SeedResponse(BaseModel):
    """Result of re-seeding the demo corpus."""

    documents: int
    chunks: int
    errors: list[str]
    doc_ids: list[str]


class HealthResponse(BaseModel):
    """Which backends are active and how large the corpus is."""

    status: str
    version: str
    backends: dict[str, Any]


class AuditResponse(BaseModel):
    """A page of audit records."""

    records: list[dict[str, Any]]
    route_distribution: dict[str, int]
    verbose_audit_enabled: bool


class EvalRunsResponse(BaseModel):
    """Stored evaluation runs, newest first."""

    runs: list[dict[str, Any]]
