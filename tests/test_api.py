"""HTTP API contract tests."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from ahrag.pipeline import AHRAGEngine


@pytest.fixture
def client(fresh_engine: AHRAGEngine) -> TestClient:
    """A test client over an engine with its own database."""
    from ahrag.api.app import create_app

    return TestClient(create_app(engine=fresh_engine, seed_on_startup=False))


class TestHealthAndMetadata:
    """Introspection endpoints."""

    def test_health_reports_active_backends(self, client: TestClient) -> None:
        """Health names which retrieval and generation backends are live."""
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        for key in ("sparse", "embedder", "vector_store", "reranker", "generator"):
            assert key in payload["backends"]

    def test_users_include_scope_sizes(self, client: TestClient) -> None:
        """Each user reports how much of the corpus they may read."""
        users = client.get("/api/users").json()
        assert len(users) == 5
        for user in users:
            assert user["authorised_chunks"] + user["withheld_chunks"] == user["total_chunks"]

    def test_finance_user_sees_more_than_a_contractor(self, client: TestClient) -> None:
        """Scope size actually varies with role."""
        users = {u["user_id"]: u for u in client.get("/api/users").json()}
        assert (
            users["dan.finance"]["authorised_chunks"]
            > users["erin.contractor"]["authorised_chunks"]
        )

    def test_documents_expose_governance_metadata(self, client: TestClient) -> None:
        """Every document reports its ACL, version, authority, and lineage."""
        documents = client.get("/api/documents").json()
        assert len(documents) == 9
        for doc in documents:
            assert doc["acl_roles"]
            assert 1 <= doc["authority_score"] <= 5
            assert doc["chunk_count"] > 0

    def test_roles_endpoint(self, client: TestClient) -> None:
        """All five demo roles are listed."""
        roles = client.get("/api/roles").json()
        assert {"employee", "manager", "engineering", "hr", "finance"} <= set(roles)


class TestQueryEndpoint:
    """The main query endpoint."""

    def test_returns_route_evidence_and_citations(self, client: TestClient) -> None:
        """A successful query returns the full decision payload."""
        response = client.post(
            "/api/query",
            json={
                "query": "What is the remediation for ERR-5041?",
                "user_id": "alice.employee",
            },
        )
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["decision"]["route"] == "R1"
        assert result["evidence"]
        assert result["citations"]
        assert result["decision"]["reasons"]
        assert len(result["decision"]["utilities"]) == 5

    def test_restricted_query_abstains_without_leaking(
        self, client: TestClient
    ) -> None:
        """An unauthorised finance question returns a refusal with no content."""
        result = client.post(
            "/api/query",
            json={
                "query": "What is the Q3 2026 revenue forecast and gross margin?",
                "user_id": "alice.employee",
            },
        ).json()["result"]
        assert result["abstained"] is True
        body = str(result)
        assert "48.2" not in body
        assert "doc-fin-q3-forecast" not in body

    def test_unknown_user_returns_404(self, client: TestClient) -> None:
        """An unknown user is rejected rather than given a default scope."""
        response = client.post(
            "/api/query", json={"query": "hello", "user_id": "mallory"}
        )
        assert response.status_code == 404

    def test_empty_query_returns_422(self, client: TestClient) -> None:
        """An empty query fails validation."""
        response = client.post(
            "/api/query", json={"query": "", "user_id": "alice.employee"}
        )
        assert response.status_code == 422

    def test_history_is_accepted(self, client: TestClient) -> None:
        """Conversation history is accepted for follow-up handling."""
        response = client.post(
            "/api/query",
            json={
                "query": "what about that?",
                "user_id": "carol.hr",
                "history": ["annual leave carry over rules"],
            },
        )
        assert response.status_code == 200

    def test_conflict_is_surfaced_in_the_payload(self, client: TestClient) -> None:
        """Version conflicts reach the client, not just the server log."""
        result = client.post(
            "/api/query",
            json={
                "query": "What is the current annual leave entitlement now?",
                "user_id": "carol.hr",
            },
        ).json()["result"]
        assert result["conflicts"]


class TestIngestEndpoint:
    """Upload handling."""

    def test_upload_without_acl_is_restricted_and_warns(
        self, client: TestClient
    ) -> None:
        """An ACL-less upload is restricted, and the response says so."""
        response = client.post(
            "/api/ingest",
            files={"file": ("note.md", io.BytesIO(b"# Note\n\nSome content here."), "text/markdown")},
            data={"title": "Note", "doc_type": "other"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["warning"]
        assert "employee" not in payload["acl_roles"]

    def test_upload_with_acl_is_stored(self, client: TestClient) -> None:
        """A declared ACL is honoured."""
        payload = client.post(
            "/api/ingest",
            files={"file": ("p.md", io.BytesIO(b"# P\n\nPolicy content body."), "text/markdown")},
            data={"acl_roles": "hr,finance", "doc_type": "policy"},
        ).json()
        assert payload["acl_roles"] == ["finance", "hr"]
        assert payload["chunks"] > 0

    def test_unsupported_type_returns_415(self, client: TestClient) -> None:
        """An unsupported format is rejected with the supported list."""
        response = client.post(
            "/api/ingest",
            files={"file": ("a.xlsx", io.BytesIO(b"junk"), "application/vnd.ms-excel")},
        )
        assert response.status_code == 415

    def test_invalid_effective_date_returns_422(self, client: TestClient) -> None:
        """A malformed date is a validation error."""
        response = client.post(
            "/api/ingest",
            files={"file": ("a.md", io.BytesIO(b"# A\n\nBody content."), "text/markdown")},
            data={"effective_date": "not-a-date"},
        )
        assert response.status_code == 422


class TestAuditEndpoint:
    """Audit exposure."""

    def test_audit_records_appear_after_a_query(self, client: TestClient) -> None:
        """Queries produce inspectable audit records."""
        client.post(
            "/api/query",
            json={"query": "What is ERR-5041?", "user_id": "alice.employee"},
        )
        payload = client.get("/api/audit").json()
        assert payload["records"]
        assert payload["verbose_audit_enabled"] is False
        assert payload["records"][0]["query_text"] is None

    def test_route_distribution_is_reported(self, client: TestClient) -> None:
        """The audit endpoint reports the observed route mix."""
        client.post(
            "/api/query",
            json={"query": "What is ERR-5041?", "user_id": "alice.employee"},
        )
        distribution = client.get("/api/audit").json()["route_distribution"]
        assert set(distribution) == {"R0", "R1", "R2", "R3", "R4"}
        assert sum(distribution.values()) >= 1

    def test_audit_can_be_cleared(self, client: TestClient) -> None:
        """The demo convenience endpoint empties the log."""
        client.post(
            "/api/query",
            json={"query": "What is ERR-5041?", "user_id": "alice.employee"},
        )
        assert client.delete("/api/audit").status_code == 200
        assert client.get("/api/audit").json()["records"] == []


class TestSeedEndpoint:
    """Corpus reset."""

    def test_seed_rebuilds_the_corpus(self, client: TestClient) -> None:
        """Re-seeding restores the packaged demo corpus."""
        payload = client.post("/api/seed").json()
        assert payload["documents"] == 9
        assert payload["chunks"] > 40
        assert payload["errors"] == []
